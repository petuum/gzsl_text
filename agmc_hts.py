from absl import flags, app

import copy
import os
import pprint
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from common_module import get_scheduler, AdamW
from data_loader import Dataset, preload_data, load_label_description, load_label_adj_graph, load_word_embedding
from extract_keywords import get_keywords_by_word_emb
from generative_module import ConditionalGenerator, ConditionalDiscriminator, RNNLabelEncoder, LinearKeywordsDecoder
from gan_utils import get_gan_hyper, get_nearest_for_zero, save_features, calc_gradient_penalty, load_gan_model
from helper import log, split_labels_by_count, label_to_indices, labels_to_index_labels, \
    iterate_minibatch, targets_to_count, all_metrics, metric_string, simple_iterate_minibatch, \
    get_label_feat_list, load_feature_extractor
from models import ZAGRNNExtractor


FLAGS = flags.FLAGS

flags.DEFINE_boolean('finetune', False, 'Fine-tuning or train GAN')
flags.DEFINE_boolean('add_zero', False, 'Add zero-shot labels when training GAN')
flags.DEFINE_boolean('eval_zero', False, 'Evaluate zero-shot labels or few-shot labels')

flags.DEFINE_float('C', 2.0, 'Label distribution aware margin')
flags.DEFINE_float('ft_lr', 1e-5, 'Learning rate for fine-tuning')
flags.DEFINE_float('gan_lr', 1e-4, 'Learning rate for GAN')
flags.DEFINE_float('key_lambda', 0., 'Coefficient for keyword reconstruction loss')

flags.DEFINE_integer('max_input_len', 2000, 'Max number of words in input document')
flags.DEFINE_integer('critic_iters', 5, 'Number of critic iterations in WGAN')
flags.DEFINE_integer('ft_epochs', 30, 'Number of epoch for training GAN')
flags.DEFINE_integer('gan_epochs', 100, 'Number of epoch for training GAN')
flags.DEFINE_integer('gan_batch_size', 8, 'Batch size for training GAN')
flags.DEFINE_integer('ft_batch_size', 8, 'Batch size on real data for finetuning')
flags.DEFINE_integer('eval_batch_size', 8, 'Batch size for evaluation')
flags.DEFINE_integer('ngh', 800, 'Number of hidden units in generator')
flags.DEFINE_integer('ndh', 800, 'Number of hidden units in discriminator')
flags.DEFINE_integer('top_k', 30, 'Number of keywords to predict')
flags.DEFINE_integer('syn_num', 256, 'Number of synthetic data for fine-tune')
flags.DEFINE_integer('neg_iters', 5, 'Iterations of negative real data when finetune')

flags.DEFINE_string('graph_encoder', 'gate', 'Graph neural network for encoding label hierarchy')
flags.DEFINE_string('gpu', 'cuda:0', 'Which gpu to use')


def train_generative(lr=1e-4, num_epochs=30, critic_iters=1, batch_size=8, max_input_len=2000, C=0., gpu="cuda:0",
                     graph_encoder='conv', beta1=0.5, ndh=256, ngh=512,  key_lambda=0., top_k=10, add_zero=False):
    pprint.pprint(locals(), stream=sys.stderr)

    gan_hyper = get_gan_hyper(lr, ndh, ngh, critic_iters, key_lambda, top_k, add_zero)
    log(f'{gan_hyper}')
    
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    train_data = Dataset('train')
    dev_data = Dataset('dev')

    train_inputs, train_labels = train_data.get_data()
    dev_inputs, dev_labels = dev_data.get_data()
    log(f'Loaded {len(train_inputs)} train data, {len(dev_inputs)} dev data...')
    n_train_data = len(train_inputs)

    train_label_set = train_data.get_all_labels()
    dev_label_set = dev_data.get_all_labels()

    all_labels = train_label_set.union(dev_label_set)
    all_labels = sorted(all_labels)

    labels_to_targets = labels_to_index_labels(all_labels)
    eval_label_size = len(labels_to_targets)

    _, _, _, labels_counter = split_labels_by_count(train_labels, dev_labels, train_label_set, dev_label_set)
    target_count = targets_to_count(labels_to_targets, labels_counter)

    word_emb = load_word_embedding()
    adj_matrix = load_label_adj_graph(labels_to_targets)
    label_idx_matrix, label_idx_mask = load_label_description(labels_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32))
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32))
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    # init model
    log(f'Building model on {device}...')
    model = ZAGRNNExtractor(word_emb, label_idx_matrix, label_idx_mask, adj_matrix, num_neighbors, loss_fn,
                            eval_label_size=eval_label_size, graph_encoder=graph_encoder,
                            target_count=target_count if C > 0. else None, C=C)

    # load stage 1 model
    pretrained_model_path = f"./models/{model.pretrain_name}"
    pretrained_state_dict = load_feature_extractor(model_path=pretrained_model_path, device=device)
    model.load_pretrained_state_dict(pretrained_state_dict)

    # set to sparse here
    model.adj_matrix.data = model.adj_matrix.data.to_sparse()
    model.to(device)

    _, clf_emb = model.get_froze_label_emb()
    label_emb = clf_emb[:, 200:]

    feat_path = model.pretrain_name.replace('.model', '.npz')

    if os.path.exists(f'./data/feats_{feat_path}'):
        log(f'Loading pretrained features from {feat_path}')
        with np.load(f'./data/feats_{feat_path}') as f:
            fs, ls, kws, kwms = f['arr_0'], f['arr_1'], f['arr_2'], f['arr_3']

        label_feat_list = get_label_feat_list(fs, ls)
        sims = None
        if add_zero:
            fs, ls, kws, kwms, sims = get_nearest_for_zero(eval_label_size, label_feat_list, clf_emb,
                                                           (fs, ls, kws, kwms))
        fs, ls, kws, kwms = fs.astype(np.float32), ls.astype(int), kws.astype(int), kwms.astype(np.float32)

        if key_lambda > 0.:
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
    else:
        train_x, train_y, train_masks = preload_data(train_inputs, train_labels, labels_to_targets, max_input_len)
        train_keywords = get_keywords_by_word_emb(model_path=pretrained_model_path)
        save_features(model, train_x, train_y, train_masks, eval_label_size, device, train_keywords, feat_path)
        return

    label_size = label_emb.size(1) * 2
    noise_size = label_size

    generator = ConditionalGenerator(noise_size, noise_size, ngh, model.feat_size)
    discriminator = ConditionalDiscriminator(model.feat_size, noise_size, ndh, 1)

    log(f'Encoding label size {label_size}...')
    label_rnn = RNNLabelEncoder(word_emb)
    rnn_params = list(label_rnn.parameters())[1:]

    if key_lambda > 0.:
        keyword_emb = model.emb.embed.weight.detach()[all_keywords_indices]
        keyword_predictor = LinearKeywordsDecoder(model.feat_size, model.embed_size, keyword_emb)
    else:
        keyword_predictor = nn.Identity()

    generator.to(device)
    discriminator.to(device)
    keyword_predictor.to(device)
    label_rnn.to(device)

    d_params = list(discriminator.parameters()) + rnn_params
    optimizer_d = optim.Adam(d_params, lr=lr, betas=(beta1, 0.999))

    g_params = list(generator.parameters()) + rnn_params
    optimizer_g = optim.Adam(g_params, lr=lr, betas=(beta1, 0.999))

    one = torch.ones([]).to(device)
    mone = one * -1

    n_data = len(fs)

    def inf_data_sampler(b):
        while True:
            batches = simple_iterate_minibatch(fs, ls, b, shuffle=True)
            for batch in batches:
                yield batch

    def get_rnn_emb(label_indices, labels=None):
        desc_x = label_idx_matrix[label_indices]
        desc_m = label_idx_mask[label_indices]
        labels_rnn_emb = label_rnn.forward_enc(desc_x, desc_m)
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

                    # d_loss.backward()
                    optimizer_d.step()

                # train generator
                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = False  # avoid computation

                sampled_idx = next(gen_sampler)
                real_feats, label_indices = torch.from_numpy(fs[sampled_idx]), torch.from_numpy(ls[sampled_idx])
                real_feats, label_indices = real_feats.to(device), label_indices.to(device)
                sim_weight = 1 if sims is None else torch.from_numpy(sims[sampled_idx]).to(device)

                optimizer_g.zero_grad()

                # Generate a batch of images
                labels = label_emb[label_indices]
                labels = get_rnn_emb(label_indices, labels)

                labels_v = Variable(labels)
                fake_feats = generate(label_indices, labels=labels_v)
                fake_feats = torch.relu(fake_feats)

                if key_lambda > 0:
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
                g_loss = g_cost + key_lambda * keyword_loss

                g_loss.backward()
                optimizer_g.step()
                it += 1

        log(f'Epoch {epoch}, disc loss={np.mean(train_d_losses):.4f}, '
            f'gen loss={np.mean(train_g_losses):.4f}, '
            f'key loss={np.mean(train_k_losses):.4f}')

        if (epoch + 1) % 10 == 0:
            gan_model = {'generator': generator.state_dict(),
                         'discriminator': discriminator.state_dict(),
                         'label_rnn': label_rnn.state_dict()}

            torch.save(gan_model, f'models/epoch{epoch + 1}_{gan_hyper}_{model.name}')


def finetune_on_gan(lr=1e-5, batch_size=8, eval_batch_size=16, max_input_len=2000, gpu="cuda:0",  graph_encoder='gate',
                    l2_ratio=5e-4, beta1=0.5, top_k=10, finetune_epochs=10, gan_batch_size=64, syn_num=20, C=0.,
                    gan_epoch=30, ndh=256, ngh=512, critic_iters=5, gan_lr=2e-5, eval_zero=True,
                    key_lambda=0., neg_iters=5, eval_thresh=5, add_zero=False):

    pprint.pprint(locals(), stream=sys.stderr)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    train_data = Dataset('train')
    dev_data = Dataset('dev')

    train_inputs, train_labels = train_data.get_data()
    dev_inputs, dev_labels = dev_data.get_data()

    log(f'Loaded {len(train_inputs)} train data, {len(dev_inputs)} dev data...')
    n_train_data = len(train_inputs)

    train_label_set = train_data.get_all_labels()
    dev_label_set = dev_data.get_all_labels()

    all_labels = train_label_set.union(dev_label_set)
    all_labels = sorted(all_labels)

    labels_to_targets = labels_to_index_labels(all_labels)
    eval_label_size = len(labels_to_targets)

    eval_label_size = len(labels_to_targets)
    dev_freq_labels, dev_few_shot_labels, dev_zero_shot_labels, labels_counter = \
        split_labels_by_count(train_labels, dev_labels, train_label_set, dev_label_set, eval_thresh)

    dev_eval_indices = label_to_indices(dev_label_set, labels_to_targets)
    dev_freq_indices = label_to_indices(dev_freq_labels, labels_to_targets)
    dev_few_shot_indices = label_to_indices(dev_few_shot_labels, labels_to_targets)
    dev_zero_shot_indices = label_to_indices(dev_zero_shot_labels, labels_to_targets)

    log(f'Developing on {len(dev_eval_indices)} labels, {len(dev_freq_indices)} frequent labels, '
        f'{len(dev_few_shot_indices)} few shot labels and {len(dev_zero_shot_indices)} zero shot labels...')

    target_count = targets_to_count(labels_to_targets, labels_counter)  # if class_margin else None
    syn_indices = dev_zero_shot_indices if eval_zero else dev_few_shot_indices

    log(f'Synthesizing {len(syn_indices)} labels...')

    word_emb = load_word_embedding()
    adj_matrix = load_label_adj_graph(labels_to_targets)
    label_idx_matrix, label_idx_mask = load_label_description(labels_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32))  # .to(device)
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32))  # .to(device)  # L x L
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    # init model
    log(f'Building model on {device}...')
    model = ZAGRNNExtractor(word_emb, label_idx_matrix, label_idx_mask, adj_matrix, num_neighbors, loss_fn,
                            eval_label_size=eval_label_size, graph_encoder=graph_encoder,
                            target_count=target_count if C > 0. else None, C=C)

    # load stage 1 model
    pretrained_model_path = f"./models/{model.pretrain_name}"
    pretrained_state_dict = load_feature_extractor(model_path=pretrained_model_path, device=device)
    model.load_pretrained_state_dict(pretrained_state_dict)

    # set to sparse here
    model.adj_matrix.data = model.adj_matrix.data.to_sparse()
    model.to(device)

    _, pretrain_label_emb = model.get_froze_label_emb()

    raw_label_emb = pretrain_label_emb[:, 200:]
    label_size = raw_label_emb.size(1)
    label_size *= 2
    noise_size = label_size

    generator, discriminator, label_rnn = load_gan_model(model.pretrain_name, noise_size, model.feat_size,
                                                         top_k=top_k, epoch=gan_epoch, ngh=ngh, ndh=ndh,
                                                         critic_iters=critic_iters, lr=gan_lr,
                                                         add_zero=add_zero, key_lambda=key_lambda)
    generator.to(device)
    discriminator.to(device)
    label_rnn.to(device)

    model.init_output_fc(pretrain_label_emb)

    def get_rnn_emb(label_indices, labels=None):
        desc_x = label_idx_matrix[label_indices]
        desc_m = label_idx_mask[label_indices]
        labels_rnn_emb = label_rnn.forward_enc(desc_x, desc_m)
        return torch.cat([labels, labels_rnn_emb], dim=1)

    def generate(label_indices=None, labels=None):
        if labels is None:
            assert label_indices is not None
            labels = raw_label_emb[label_indices]
            labels = Variable(labels)

        b = labels.size(0)

        noises = torch.randn(b, noise_size).to(labels.device)
        noises = Variable(noises)

        fake_feats = generator.forward(noises, labels)
        fake_feats = torch.relu(fake_feats)

        return fake_feats

    finetune_params = model.output_fc.parameters()
    optimizer = AdamW(finetune_params, lr=lr, weight_decay=l2_ratio, betas=(beta1, 0.999))
    scheduler = get_scheduler(optimizer, finetune_epochs, ratios=(0.6, 0.9))

    def synthesize(m, n):
        feats = []
        labels = []
        with torch.set_grad_enabled(False):
            m.eval()
            for code in syn_indices:
                syn_labels = [code] * n
                syn_labels = torch.LongTensor(syn_labels).to(device)

                label = raw_label_emb[syn_labels]
                label = get_rnn_emb(syn_labels, label)

                gen_feats = generate(syn_labels, labels=label)
                label_feats = gen_feats.data.cpu().numpy()

                feats.append(label_feats)
                labels.append(np.ones(len(label_feats)) * code)

        feats = np.vstack(feats)
        labels = np.concatenate(labels)
        return feats.astype(np.float32), labels.astype(int)

    def inf_syn_data_sampler(x, y, b):
        while True:
            batches = simple_iterate_minibatch(x, y, b, shuffle=True)
            for batch in batches:
                yield batch

    train_x, train_y, train_masks = preload_data(train_inputs, train_labels, labels_to_targets, max_input_len)
    dev_x, dev_y, dev_masks = preload_data(dev_inputs, dev_labels, labels_to_targets, max_input_len)

    def inf_data_sampler():
        while True:
            batches = iterate_minibatch(train_x, train_y, train_masks, eval_label_size, batch_size, shuffle=True)
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

            for batch in iterate_minibatch(eval_x, eva_y, eval_mask, eval_label_size,
                                           batch_size=eval_batch_size, shuffle=False):
                x, y, mask, _ = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)

                # forward pass
                logits, loss = model.forward(x, y, mask, label_indices=eval_syn_indices)
                probs = torch.sigmoid(logits[:, :eval_label_size])

                # eval stats
                y_true.append(y[:, eval_syn_indices].cpu().numpy())
                y_score.append(probs.cpu().numpy())
                eval_losses.append(loss.mean().data.cpu().numpy())

        y_score = np.vstack(y_score)
        y_true = np.vstack(y_true)
        metircs = all_metrics(y_score, y_true)
        return np.mean(eval_losses), metircs

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
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_epoch = epoch

    return best_f1, best_epoch


def main(unused_argv):
    gpu = FLAGS.gpu
    graph_encoder = FLAGS.graph_encoder
    max_input_len = FLAGS.max_input_len
    C = FLAGS.C

    gan_lr = FLAGS.gan_lr
    gan_batch_size = FLAGS.gan_batch_size
    gan_epochs = FLAGS.gan_epochs
    critic_iters = FLAGS.critic_iters
    ngh = FLAGS.ngh
    ndh = FLAGS.ndh
    top_k = FLAGS.top_k
    key_lambda = FLAGS.key_lambda
    add_zero = FLAGS.add_zero

    ft_epochs = FLAGS.ft_epochs
    ft_lr = FLAGS.ft_lr
    ft_batch_size = FLAGS.ft_batch_size
    syn_num = FLAGS.syn_num
    neg_iters = FLAGS.neg_iters
    eval_batch_size = FLAGS.eval_batch_size
    eval_zero = FLAGS.eval_zero

    if not FLAGS.finetune:
        train_generative(lr=gan_lr, num_epochs=gan_epochs, critic_iters=critic_iters, batch_size=gan_batch_size,
                         max_input_len=max_input_len, C=C, gpu=gpu, graph_encoder=graph_encoder, ndh=ndh, ngh=ngh,
                         key_lambda=key_lambda, top_k=top_k, add_zero=add_zero)
    else:
        finetune_on_gan(lr=ft_lr, batch_size=ft_batch_size, eval_batch_size=eval_batch_size,
                        max_input_len=max_input_len, gpu=gpu,  graph_encoder=graph_encoder, top_k=top_k,
                        finetune_epochs=ft_epochs, gan_batch_size=gan_batch_size, syn_num=syn_num, C=C,
                        gan_epoch=gan_epochs, ndh=ndh, ngh=ngh, critic_iters=critic_iters, gan_lr=gan_lr,
                        eval_zero=eval_zero, key_lambda=key_lambda, neg_iters=neg_iters, add_zero=add_zero)


if __name__ == '__main__':
    app.run(main)
