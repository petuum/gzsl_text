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


import os
import pprint
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from config import get_gan_config
from constant import CACHE_PATH, MODEL_DIR, FEATURE_DIR
from loaders.data_loader import Dataset, prepare_code_data, preload_data, load_adj_matrix
from loaders.emb_loader import init_emb_and_code_input
from loaders.feature_loader import save_features, load_features, get_nearest_for_zero
from loaders.keywords_loader import load_note_keywords
from loaders.model_loader import load_first_stage_model, get_gan_model_name
from modules.common_module import get_loss_fn
from modules.decoder_module import load_decoder
from modules.generative_module import ConditionalGenerator, ConditionalDiscriminator, calc_gradient_penalty
from modules.icd_modules import ConvLabelAttnGAN
from modules.lm_module import RNNLabelEncoder
from utils.helper import log, split_codes_by_count, code_to_indices, codes_to_index_labels, \
    targets_to_count, simple_iterate_minibatch, get_code_feat_list


def train_generative(lr=1e-4, num_epochs=30, critic_iters=1, max_note_len=2000, gpu="cuda:0", loss='bce',
                     graph_encoder='conv', batch_size=64, C=0., class_margin=False, ndh=256, ngh=512, save_every=10,
                     reg_ratio=0., top_k=10, decoder='linear', add_zero=False, pool_mode='last'):
    pprint.pprint(locals(), stream=sys.stderr)
    gan_hyper = get_gan_model_name(lr, ndh, ngh, critic_iters, reg_ratio, decoder, top_k, add_zero, pool_mode)

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    train_data = Dataset('train')
    dev_data = Dataset('dev')
    test_data = Dataset('test')

    train_notes, train_labels = train_data.get_data()
    dev_notes, dev_labels = dev_data.get_data()
    log(f'Loaded {len(train_notes)} train data, {len(dev_notes)} dev data...')
    n_train_data = len(train_notes)

    train_codes = train_data.get_all_codes()
    dev_codes = dev_data.get_all_codes()
    test_codes = test_data.get_all_codes()

    all_codes = train_codes.union(dev_codes).union(test_codes)
    all_codes = sorted(all_codes)

    codes_to_targets = codes_to_index_labels(all_codes, False)
    extended_codes_to_targets, adj_matrix, _ = load_adj_matrix(codes_to_targets)

    eval_code_size = len(codes_to_targets)

    frequent_codes, few_shot_codes, zero_shot_codes, codes_counter = split_codes_by_count(train_labels, dev_labels,
                                                                                          train_codes, dev_codes)
    eval_indices = code_to_indices(dev_codes, codes_to_targets)
    frequent_indices = code_to_indices(frequent_codes, codes_to_targets)
    few_shot_indices = code_to_indices(few_shot_codes, codes_to_targets)
    zero_shot_indices = code_to_indices(zero_shot_codes, codes_to_targets)

    log(f'Evaluating on {len(eval_indices)} codes, {len(frequent_indices)} frequent codes, '
        f'{len(few_shot_indices)} few shot codes and {len(zero_shot_indices)} zero shot codes...')

    target_count = targets_to_count(codes_to_targets, codes_counter)

    word_emb, code_idx_matrix, code_idx_mask = init_emb_and_code_input(extended_codes_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32))
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32))
    loss_fn = get_loss_fn(loss, reduction='sum')

    # init model
    log(f'Building model on {device}...')
    model = ConvLabelAttnGAN(word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors, loss_fn,
                             eval_code_size=eval_code_size, graph_encoder=graph_encoder,
                             target_count=target_count if class_margin else None, C=C)

    # load stage 1 base feature extractor model
    pretrained_model_path = f"{MODEL_DIR}/{model.pretrain_name}"
    pretrained_state_dict = load_first_stage_model(model_path=pretrained_model_path, device=device)
    model.load_pretrained_state_dict(pretrained_state_dict)

    # set to sparse here
    model.adj_matrix = model.adj_matrix.to_sparse()
    model.to(device)

    _, label_emb = model.get_froze_label_emb()
    clf_emb = label_emb
    graph_label_emb = label_emb[:, int(label_emb.shape[1] // 2):]
    label_emb = graph_label_emb

    feat_path = model.pretrain_name.replace('.model', '.npz')
    if not os.path.exists(f'{FEATURE_DIR}/{feat_path}'):
        log(f'Saving features to {feat_path}...')
        train_x, train_y, train_masks = preload_data(train_notes, train_labels, codes_to_targets, max_note_len,
                                                     save_path=CACHE_PATH)
        train_keywords = load_note_keywords(train_notes)
        save_features(model, train_x, train_y, train_masks, eval_code_size, device, train_keywords, save_path=feat_path)

    fs, ls, kws, kwms = load_features(feat_path)
    code_feat_list, _ = get_code_feat_list(fs, ls)
    if add_zero:
        fs, ls, kws, kwms, sims = get_nearest_for_zero(eval_code_size, code_feat_list, clf_emb, (fs, ls, kws, kwms))
    else:
        sims = None

    fs, ls, kws, kwms = fs.astype(np.float32), ls.astype(int), kws.astype(int), kwms.astype(np.float32)

    if reg_ratio > 0.:
        log(f'Predicting top {top_k} words...')
        kws = kws[:, :top_k]
        kwms = kwms[:, :top_k]

    all_keywords_indices = sorted(np.unique(kws))
    log(f'In total {len(all_keywords_indices)} keywords...')

    word_idx_to_keyword_idx = dict(zip(all_keywords_indices, np.arange(len(all_keywords_indices))))
    all_keywords_indices = torch.LongTensor(all_keywords_indices).to(device)
    kws = np.asarray([[word_idx_to_keyword_idx[w] for w in kw] for kw in kws], dtype=int)

    log('Activating features...')
    fs = np.maximum(fs, 0)

    label_size = label_emb.size(1) * 2  # concat with repr from RNN
    noise_size = label_size

    generator = ConditionalGenerator(noise_size, noise_size, ngh, model.feat_size)
    discriminator = ConditionalDiscriminator(model.feat_size, noise_size, ndh, 1)

    log(f'Encoding label using RNN, label hidden size {label_size}...')
    code_desc, code_mask, word_emb = prepare_code_data(model, device, to_torch=True)
    label_rnn = RNNLabelEncoder(word_emb)
    rnn_params = list(label_rnn.parameters())[1:]

    if reg_ratio > 0.:
        keyword_emb = model.emb.embed.weight.detach()[all_keywords_indices]
        keyword_predictor = load_decoder(decoder, model.feat_size, model.embed_size, keyword_emb)
    else:
        keyword_predictor = nn.Identity()

    generator.to(device)
    discriminator.to(device)
    keyword_predictor.to(device)
    label_rnn.to(device)

    d_params = list(discriminator.parameters()) + rnn_params
    optimizer_d = optim.Adam(d_params, lr=lr, betas=(0.5, 0.999))
    g_params = list(generator.parameters()) + list(keyword_predictor.parameters())
    optimizer_g = optim.Adam(g_params, lr=lr, betas=(0.5, 0.999))

    one = torch.ones([]).to(device)
    mone = one * -1

    n_data = len(fs)

    def inf_data_sampler(b):
        while True:
            batches = simple_iterate_minibatch(fs, ls, b, shuffle=True)
            for batch in batches:
                yield batch

    def get_rnn_emb(label_indices, labels=None):
        desc_x = code_desc[label_indices][:, :20]
        desc_m = code_mask[label_indices][:, :20]
        labels_rnn_emb = label_rnn.forward_enc(desc_x, desc_m, None, training=False, pool_mode=pool_mode)
        return torch.cat([labels, labels_rnn_emb], dim=1)

    def generate(label_indices=None, labels=None):
        if labels is None:
            assert label_indices is not None
            labels = label_emb[label_indices]
            labels = get_rnn_emb(label_indices, labels)
            labels = Variable(labels)

        b = labels.size(0)
        noises = torch.randn(b, noise_size).to(labels.device)
        noises = Variable(noises)
        feats = generator.forward(noises, labels)
        return feats

    log(f'Start training WGAN-GP with {n_data} pretrained features')
    it = 0

    num_batches = n_data // batch_size

    disc_sampler = inf_data_sampler(batch_size)
    gen_sampler = inf_data_sampler(batch_size)

    for epoch in range(num_epochs):
        train_g_losses = []
        train_d_losses = []
        train_r_losses = []
        train_k_losses = []

        # train one epoch
        with torch.set_grad_enabled(True):
            model.eval()
            keyword_predictor.train()
            discriminator.train()
            generator.train()
            label_rnn.train()

            for p in model.parameters():
                p.requires_grad = False

            for _ in range(num_batches):
                # train discriminator
                for p in discriminator.parameters():
                    p.requires_grad = True

                for it_c in range(critic_iters):
                    sampled_idx = next(disc_sampler)

                    real_feats, label_indices = torch.from_numpy(fs[sampled_idx]), torch.from_numpy(ls[sampled_idx])
                    real_feats, label_indices = real_feats.to(device), label_indices.to(device)
                    sim_weight = 1 if sims is None else torch.from_numpy(sims[sampled_idx]).to(device)

                    optimizer_d.zero_grad()

                    labels = label_emb[label_indices]
                    labels = get_rnn_emb(label_indices, labels)

                    real_feats_v = Variable(real_feats)
                    labels_v = Variable(labels)

                    real_logits = discriminator.forward(real_feats_v, labels_v)
                    critic_d_real = (real_logits * sim_weight).mean()
                    critic_d_real.backward(mone, retain_graph=True if add_zero else False)

                    fake_feats = generate(label_indices, labels=labels_v)
                    fake_feats = torch.relu(fake_feats)

                    fake_logits = discriminator.forward(fake_feats.detach(), labels_v)

                    critic_d_fake = (fake_logits * sim_weight).mean()
                    critic_d_fake.backward(one, retain_graph=True if add_zero else False)

                    gp = calc_gradient_penalty(discriminator, real_feats, fake_feats.data, labels)
                    gp.backward()

                    d_cost = critic_d_fake - critic_d_real  # + gp
                    train_d_losses.append(d_cost.data.cpu().numpy())
                    optimizer_d.step()

                # train generator
                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = False  # avoid computation

                sampled_idx = next(gen_sampler)
                real_feats, label_indices = torch.from_numpy(fs[sampled_idx]), torch.from_numpy(ls[sampled_idx])
                real_feats, label_indices = real_feats.to(device), label_indices.to(device)
                sim_weight = 1 if sims is None else torch.from_numpy(sims[sampled_idx]).to(device)

                optimizer_g.zero_grad()

                # Generate a batch of data
                labels = label_emb[label_indices]
                labels = get_rnn_emb(label_indices, labels)

                labels_v = Variable(labels)
                fake_feats = generate(label_indices, labels=labels_v)
                fake_feats = torch.relu(fake_feats)

                recon_loss = F.mse_loss(fake_feats, real_feats, reduction='mean')
                train_r_losses.append(recon_loss.data.cpu().numpy())

                if reg_ratio > 0:
                    keyword_indices, keyword_masks = torch.from_numpy(kws[sampled_idx]), \
                                                     torch.from_numpy(kwms[sampled_idx])
                    keyword_indices, keyword_masks = keyword_indices.to(device), keyword_masks.to(device)
                    keyword_loss = keyword_predictor(fake_feats, keyword_indices, keyword_masks, labels_v)
                    if not isinstance(sim_weight, int):
                        keywords_weight = sim_weight.masked_fill(sim_weight != 1, 0.)
                    else:
                        keywords_weight = sim_weight

                    keyword_loss = torch.mean(keyword_loss * keywords_weight)
                    train_k_losses.append(keyword_loss.data.cpu().numpy())
                else:
                    keyword_loss = 0

                fake_logits = discriminator.forward(fake_feats, labels_v)
                critic_g_fake = (fake_logits * sim_weight).mean()
                g_cost = -critic_g_fake

                train_g_losses.append(g_cost.data.cpu().numpy())
                g_loss = g_cost + reg_ratio * keyword_loss

                g_loss.backward()
                optimizer_g.step()
                it += 1

        log(f'Epoch {epoch}, disc loss={np.mean(train_d_losses):.4f}, '
            f'gen loss={np.mean(train_g_losses):.4f}, '
            f'mse loss={np.mean(train_r_losses):.4f}, '
            f'key loss={np.mean(train_k_losses):.4f}')

        # eval on few / zero shot examples
        with torch.set_grad_enabled(False):
            model.eval()
            discriminator.eval()
            generator.eval()
            label_rnn.eval()
            keyword_predictor.eval()

            sample_num = 100
            dev_scores = []
            for code in few_shot_indices + zero_shot_indices:
                syn_codes = [code] * sample_num
                gen_feats = generate(syn_codes)
                scores = torch.sigmoid(torch.mul(torch.relu(gen_feats), clf_emb[code]).sum(-1))
                dev_scores.append(scores.data.cpu().numpy())

            dev_scores = np.concatenate(dev_scores)
            dev_preds = np.round(dev_scores)

            log(f'\tF/Z: gen probs={np.mean(dev_scores) * 100:.2f}, '
                f'gen acc={np.mean(dev_preds == 1) * 100:.2f} ')

        start_saving = 20
        if (epoch + 1) % save_every == 0 and epoch + 1 >= start_saving:
            gan_model = {'generator': generator.state_dict(),
                         'discriminator': discriminator.state_dict(),
                         'label_rnn': label_rnn.state_dict()}
            torch.save(gan_model, f'{MODEL_DIR}/epoch{epoch + 1}_{gan_hyper}_{model.pretrain_name}')


if __name__ == '__main__':
    gan_config = get_gan_config()
    train_generative(gpu=gan_config.gpu,
                     graph_encoder=gan_config.graph_encoder,
                     class_margin=gan_config.class_margin,
                     C=gan_config.C,
                     num_epochs=gan_config.num_epochs,
                     batch_size=gan_config.batch_size,
                     add_zero=gan_config.add_zero,
                     critic_iters=gan_config.critic_iters,
                     lr=gan_config.lr,
                     ndh=gan_config.ndh,
                     ngh=gan_config.ngh,
                     reg_ratio=gan_config.reg_ratio,
                     decoder=gan_config.decoder,
                     top_k=gan_config.top_k,
                     save_every=gan_config.save_every,
                     pool_mode=gan_config.pool_mode)
