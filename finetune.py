# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import pprint
import sys

import numpy as np
import torch
from torch.autograd import Variable

from config import get_finetune_config
from constant import MODEL_DIR, CACHE_PATH
from loaders.data_loader import Dataset, prepare_code_data, load_adj_matrix, preload_data
from loaders.emb_loader import init_emb_and_code_input
from loaders.feature_loader import get_features_for_labels
from loaders.model_loader import load_first_stage_model, load_gan_model
from modules.common_module import get_loss_fn, get_scheduler, AdamW
from modules.icd_modules import ConvLabelAttnGAN
from utils.helper import log, split_codes_by_count, code_to_indices, codes_to_index_labels, get_code_data_list, \
    iterate_minibatch, targets_to_count, all_metrics, metric_string, simple_iterate_minibatch, seed


def load_data(train_notes, train_labels, dev_notes, dev_labels, codes_to_targets, max_note_len):
    log('Preloading data in memory...')
    train_x, train_y, train_masks = preload_data(train_notes, train_labels, codes_to_targets, max_note_len,
                                                 save_path=CACHE_PATH)
    code_data_list = get_code_data_list(train_x, train_y, train_masks)
    dev_x, dev_y, dev_masks = preload_data(dev_notes, dev_labels, codes_to_targets, max_note_len)
    return code_data_list, train_x, train_y, train_masks, dev_x, dev_y, dev_masks


def finetune_on_gan(eval_zero=True, lr=1e-5, batch_size=8, eval_batch_size=16, neg_iters=1, max_note_len=2000,
                    gpu="cuda:0", loss='bce', graph_encoder='conv', l2_ratio=5e-4, top_k=10, finetune_epochs=10,
                    gan_batch_size=64, syn_num=20, C=0., class_margin=False, gan_epoch=30, ndh=256, ngh=512,
                    critic_iters=5, gan_lr=2e-5, reg_ratio=0., decoder='linear', add_zero=False, pool_mode='last'):
    seed()
    pprint.pprint(locals(), stream=sys.stderr)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    train_data = Dataset('train')
    dev_data = Dataset('dev')
    test_data = Dataset('test')

    train_notes, train_labels = train_data.get_data()
    dev_notes, dev_labels = dev_data.get_data()
    test_notes, test_labels = test_data.get_data()

    log(f'Loaded {len(train_notes)} train data, {len(dev_notes)} dev data...')
    n_train_data = len(train_notes)

    train_codes = train_data.get_all_codes()
    dev_codes = dev_data.get_all_codes()
    test_codes = test_data.get_all_codes()

    all_codes = train_codes.union(dev_codes).union(test_codes)
    all_codes = sorted(all_codes)

    codes_to_targets = codes_to_index_labels(all_codes, False)
    extended_codes_to_targets, adj_matrix, codes_to_parents = load_adj_matrix(codes_to_targets)

    eval_code_size = len(codes_to_targets)
    dev_freq_codes, dev_few_shot_codes, dev_zero_shot_codes, codes_counter = \
        split_codes_by_count(train_labels, dev_labels, train_codes, dev_codes)

    test_freq_codes, test_few_shot_codes, eval_zero_shot_codes, _ = \
        split_codes_by_count(train_labels, test_labels, train_codes, test_codes)

    dev_eval_indices = code_to_indices(dev_codes, codes_to_targets)
    dev_freq_indices = code_to_indices(dev_freq_codes, codes_to_targets)
    dev_few_shot_indices = code_to_indices(dev_few_shot_codes, codes_to_targets)
    dev_zero_shot_indices = code_to_indices(dev_zero_shot_codes, codes_to_targets)

    test_eval_indices = code_to_indices(test_codes, codes_to_targets)
    test_freq_indices = code_to_indices(test_freq_codes, codes_to_targets)
    test_few_shot_indices = code_to_indices(test_few_shot_codes, codes_to_targets)
    eval_zero_shot_indices = code_to_indices(eval_zero_shot_codes, codes_to_targets)

    log(f'Developing on {len(dev_eval_indices)} codes, {len(dev_freq_indices)} frequent codes, '
        f'{len(dev_few_shot_indices)} few shot codes and {len(dev_zero_shot_indices)} zero shot codes...')

    target_count = targets_to_count(codes_to_targets, codes_counter)  # if class_margin else None
    zero_shot_indices = np.union1d(dev_zero_shot_indices, eval_zero_shot_indices)
    few_shot_indices = np.union1d(dev_few_shot_indices, test_few_shot_indices)
    syn_indices = zero_shot_indices if eval_zero else few_shot_indices
    log(f'Synthesizing {len(syn_indices)} codes...')

    new_target_count = copy.copy(target_count)
    new_target_count[syn_indices] += syn_num

    word_emb, code_idx_matrix, code_idx_mask = init_emb_and_code_input(extended_codes_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32))  # .to(device)
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32))  # .to(device)  # L x L
    loss_fn = get_loss_fn(loss, reduction='sum')

    # init model
    log(f'Building model on {device}...')
    model = ConvLabelAttnGAN(word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors, loss_fn,
                             eval_code_size=eval_code_size, graph_encoder=graph_encoder,
                             target_count=target_count if class_margin else None, C=C)

    # load stage 1 model
    pretrained_model_path = f"{MODEL_DIR}/{model.pretrain_name}"
    pretrained_state_dict = load_first_stage_model(model_path=pretrained_model_path, device=device)
    model.load_pretrained_state_dict(pretrained_state_dict)

    # set to sparse here
    model.adj_matrix = model.adj_matrix.to_sparse()
    model.to(device)

    _, pretrain_label_emb = model.get_froze_label_emb()
    label_emb = pretrain_label_emb[:, int(pretrain_label_emb.shape[1]) // 2:]

    label_size = label_emb.size(1)
    label_size *= 2
    noise_size = label_size

    code_desc, code_mask, _ = prepare_code_data(model, device, to_torch=True)

    generator, discriminator, label_rnn, gan_model_path = load_gan_model(
        model.pretrain_name, noise_size, model.feat_size, top_k=top_k, pool_mode=pool_mode, epoch=gan_epoch, ngh=ngh,
        ndh=ndh, critic_iters=critic_iters, lr=gan_lr, add_zero=add_zero, reg_ratio=reg_ratio, decoder=decoder
    )

    generator.to(device)
    discriminator.to(device)
    label_rnn.to(device)

    label_real_feats, _ = get_features_for_labels(model, syn_indices)
    model.init_output_fc(pretrain_label_emb)

    def get_rnn_emb(label_indices, labels=None):
        desc_x = code_desc[label_indices][:, :top_k]
        desc_m = code_mask[label_indices][:, :top_k]
        labels_rnn_emb = label_rnn.forward_enc(desc_x, desc_m, None, training=False, pool_mode=pool_mode)
        return torch.cat([labels, labels_rnn_emb], dim=1)

    def generate(labels):
        b = labels.size(0)
        noises = torch.randn(b, noise_size).to(labels.device)
        noises = Variable(noises)

        fake_feats = generator.forward(noises, labels)
        fake_feats = torch.relu(fake_feats)

        return fake_feats

    finetune_params = model.output_fc.parameters()

    optimizer = AdamW(finetune_params, lr=lr, weight_decay=l2_ratio, betas=(0.5, 0.999))
    scheduler = get_scheduler(optimizer, finetune_epochs, ratios=(0.6, 0.9))

    def synthesize(m, n):
        feats = []
        labels = []
        sample_num = n
        with torch.set_grad_enabled(False):
            m.eval()
            for code in syn_indices:
                syn_codes = [code] * sample_num
                syn_codes = torch.LongTensor(syn_codes).to(device)

                label = label_emb[syn_codes]
                label = get_rnn_emb(syn_codes, label)

                gen_feats = generate(labels=label)
                code_feats = gen_feats.data.cpu().numpy()

                if code in label_real_feats:
                    code_feats = np.vstack([code_feats] + label_real_feats[code])

                feats.append(code_feats)
                labels.append(np.ones(len(code_feats)) * code)

        feats = np.vstack(feats)
        labels = np.concatenate(labels)
        return feats.astype(np.float32), labels.astype(int)

    def inf_syn_data_sampler(x, y, b):
        while True:
            batches = simple_iterate_minibatch(x, y, b, shuffle=True)
            for batch in batches:
                yield batch

    code_data_list, train_x, train_y, train_masks, dev_x, dev_y, dev_masks = \
        load_data(train_notes, train_labels, dev_notes, dev_labels, codes_to_targets, max_note_len)

    def inf_data_sampler():
        while True:
            batches = iterate_minibatch(train_x, train_y, train_masks, eval_code_size, batch_size, shuffle=True)
            for batch in batches:
                yield batch

    syn_feats, syn_labels = synthesize(model, syn_num)
    syn_sampler = inf_syn_data_sampler(syn_feats, syn_labels, gan_batch_size)

    real_sampler = inf_data_sampler()
    n_data = len(syn_labels)
    n_batches = n_data // gan_batch_size

    log('Finetuning on GAN generated samples...')

    syn_indices = torch.LongTensor(syn_indices).to(device)
    dev_syn_indices = torch.LongTensor(dev_zero_shot_indices).to(device) \
        if eval_zero else torch.LongTensor(dev_few_shot_indices).to(device)
    test_syn_indices = torch.LongTensor(eval_zero_shot_indices).to(device) \
        if eval_zero else torch.LongTensor(test_few_shot_indices).to(device)

    best_f1 = -1
    best_epoch = -1

    def eval_finetune(eval_x, eva_y, eval_mask, eval_syn_indices):
        eval_losses = []
        y_true = []
        y_score = []

        with torch.set_grad_enabled(False):
            model.eval()
            generator.eval()
            discriminator.eval()
            label_rnn.eval()
            label_rnn.eval()

            for batch in iterate_minibatch(eval_x, eva_y, eval_mask, eval_code_size,
                                           batch_size=eval_batch_size, shuffle=False):
                x, y, mask, _ = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)

                # forward pass
                logits, loss = model.forward(x, y, mask, label_indices=eval_syn_indices)
                probs = torch.sigmoid(logits[:, :eval_code_size])

                # eval stats
                y_true.append(y[:, eval_syn_indices].cpu().numpy())
                y_score.append(probs.cpu().numpy())
                eval_losses.append(loss.mean().data.cpu().numpy())

        y_score = np.vstack(y_score)
        y_true = np.vstack(y_true)
        metircs = all_metrics(y_score, y_true)
        return np.mean(eval_losses), metircs

    gan_model_path = gan_model_path.replace('model', 'npz')
    for epoch in range(finetune_epochs):
        with torch.set_grad_enabled(True):
            model.train()
            generator.eval()
            discriminator.eval()
            label_rnn.eval()

            train_losses = []
            for _ in range(n_batches):
                for _ in range(neg_iters):
                    x, y, mask, _ = next(real_sampler)
                    x, y, mask = x.to(device), y.to(device), mask.to(device)

                    # forward pass
                    optimizer.zero_grad()
                    _, loss = model.forward(x, y, mask, label_indices=syn_indices)
                    loss.backward()
                    optimizer.step()

                sample_indices = next(syn_sampler)

                syn_x = syn_feats[sample_indices]
                syn_y = syn_labels[sample_indices]
                syn_x, syn_y = torch.from_numpy(syn_x), torch.from_numpy(syn_y)
                syn_x, syn_y = syn_x.to(device), syn_y.to(device)

                optimizer.zero_grad()
                finetune_loss = model.forward_final(syn_x, syn_y) / syn_y.size(0)
                finetune_loss.backward()

                optimizer.step()
                train_losses.append(finetune_loss.data.cpu().numpy())

        temp = copy.deepcopy(model.output_fc.weight.data)
        model.output_fc.weight.data.copy_(pretrain_label_emb.data)
        model.output_fc.weight.data[syn_indices] = temp[syn_indices]
        del temp

        dev_loss, dev_metrics = eval_finetune(dev_x, dev_y, dev_masks, dev_syn_indices)
        log(f"Epoch {epoch}, train loss={np.mean(train_losses):.4f}, dev loss={dev_loss:.4f}\n")
        log(f"\t{metric_string(dev_metrics)}\n")

        curr_f1 = dev_metrics['f1_micro']
        if curr_f1 > best_f1 and epoch > finetune_epochs // 2:
            best_f1 = curr_f1
            best_epoch = epoch
            # save the best final code classifier based on dev F1 score
            np.savez(f'{MODEL_DIR}/ft_z{eval_zero}_{gan_model_path}', model.output_fc.weight.data.cpu().numpy())

        if lr >= 1e-4:
            scheduler.step(epoch)

    return best_f1, best_epoch


if __name__ == '__main__':
    finetune_config = get_finetune_config()
    log('Finetuning code classifier with GAN generated features...')
    finetune_on_gan(eval_zero=finetune_config.eval_zero,
                    gpu=finetune_config.gpu,
                    lr=finetune_config.lr,
                    graph_encoder=finetune_config.graph_encoder,
                    syn_num=finetune_config.syn_num,
                    gan_batch_size=finetune_config.gan_batch_size,
                    class_margin=finetune_config.class_margin,
                    C=finetune_config.C,
                    gan_epoch=finetune_config.gan_epoch,
                    finetune_epochs=finetune_config.finetune_epochs,
                    gan_lr=finetune_config.gan_lr,
                    critic_iters=finetune_config.critic_iters,
                    ndh=finetune_config.ndh,
                    ngh=finetune_config.ngh,
                    add_zero=finetune_config.add_zero,
                    batch_size=finetune_config.batch_size,
                    eval_batch_size=finetune_config.eval_batch_size,
                    reg_ratio=finetune_config.reg_ratio,
                    neg_iters=finetune_config.neg_iters,
                    l2_ratio=finetune_config.l2_ratio,
                    decoder=finetune_config.decoder,
                    top_k=finetune_config.top_k,
                    pool_mode=finetune_config.pool_mode)
