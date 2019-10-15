import numpy as np
import torch
import torch.autograd as autograd
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch.autograd import Variable

from generative_module import ConditionalDiscriminator, ConditionalGenerator, RNNLabelEncoder
from helper import iterate_minibatch, log


def load_gan_model(pretrain_model_path, noise_size=100, feat_size=200, epoch=30, ndh=256, ngh=512, critic_iters=5,
                   lr=2e-5, key_lambda=0., top_k=30, add_zero=False):
    gan_hyper = get_gan_hyper(lr, ndh, ngh, critic_iters, key_lambda, top_k, add_zero)

    model_path = f'epoch{epoch}_{gan_hyper}_gan_{pretrain_model_path}'
    log(f'Loading GAN from {model_path}...')
    gan_states = torch.load(f'./models/{model_path}', map_location=lambda storage, loc: storage)

    generator = ConditionalGenerator(noise_size, noise_size, ngh, feat_size)
    discriminator = ConditionalDiscriminator(feat_size, noise_size, ndh, 1)

    word_emb = gan_states['label_rnn']['encoder.weight'].data.numpy()
    label_rnn = RNNLabelEncoder(word_emb)
    label_rnn.load_state_dict(gan_states['label_rnn'])

    generator.load_state_dict(gan_states['generator'])
    discriminator.load_state_dict(gan_states['discriminator'])

    for m in [generator, discriminator, label_rnn]:
        for p in m.parameters():
            p.requires_grad = False

    return generator, discriminator, label_rnn


def save_features(model, train_x, train_y, train_masks, eval_label_size, device, train_keywords, save_path=None):
    k = 30
    thresh = 0.1

    fs = []
    ls = []
    kws = []
    kwms = []
    batches = iterate_minibatch(train_x, train_y, train_masks, eval_label_size, 20, shuffle=False, rtn_indices=True)
    with torch.set_grad_enabled(False):
        model.eval()
        for batch in tqdm.tqdm(batches):
            x, y, mask, indices = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            direct_feats, _ = model.forward_real_feats(x, mask)
            dfs, label_indices = [], []

            for df, labels, data_idx in zip(direct_feats, y, indices):
                idx = labels.nonzero().squeeze(1)
                dfs.append(df[idx])
                label_indices.append(idx)

                if train_keywords is not None:
                    label_keywords = train_keywords[data_idx]
                    for code in idx:
                        m = np.zeros(k)
                        kw = np.zeros(k)
                        code = int(code)

                        if code in label_keywords:
                            keywords, scores = label_keywords[code][:2]
                            keywords = keywords[scores >= thresh][:k]
                            m[: len(keywords)] = scores[scores >= thresh][:k]
                            kw[: len(keywords)] = keywords

                        kws.append(kw)
                        kwms.append(m)

            dfs = torch.cat(dfs, 0)
            label_indices = torch.cat(label_indices, 0)

            fs.append(dfs.data.cpu().numpy())
            ls.append(label_indices.data.cpu().numpy())

    fs = np.vstack(fs)
    ls = np.concatenate(ls)
    kws = np.vstack(kws).astype(int)
    kwms = np.vstack(kwms)

    assert len(fs) == len(ls) == len(kws) == len(kwms)
    if save_path is None:
        save_path = model.pretrain_name.replace('.model', '.npz')

    np.savez(f'./data/feats_{save_path}', fs, ls, kws, kwms)
    return fs, ls, kws, kwms


def get_nearest_for_zero(eval_label_size, label_feat_list, label_emb, data):
    fs, ls, kws, kwms = data
    zeroshot_labels = []
    rest_labels = set(label_feat_list.keys())

    for i in range(eval_label_size):
        if i not in label_feat_list:
            zeroshot_labels.append(i)

    label_emb = label_emb.data.cpu().numpy()
    zeroshot_emb = label_emb[zeroshot_labels]
    cos_dist = cosine_similarity(zeroshot_emb, label_emb)

    n = len(fs)
    zeroshot_neighbors = dict()
    sims = np.ones(n)

    for z, dist in zip(zeroshot_labels, cos_dist):
        z_fs = []
        kw = []
        m = []
        sim = []
        added = 0
        for nei in np.argsort(-dist)[1:]:
            if nei in rest_labels:
                zeroshot_neighbors[z] = nei
                n_aug = len(label_feat_list[nei])
                nei_idx = label_feat_list[nei].l
                z_fs.append(fs[nei_idx])
                kw.append(kws[nei_idx])
                m.append(kwms[nei_idx])
                sim.append(np.ones(n_aug) * dist[nei])
                added += len(nei_idx)
                break

        s = len(fs)
        fs = np.vstack([fs] + z_fs)
        e = len(fs)

        kws = np.vstack([kws] + kw)
        kwms = np.vstack([kwms] + m)
        sims = np.concatenate([sims] + sim)

        data_idx = list(range(s, e))
        ls = np.concatenate([ls, np.ones(len(data_idx)) * z])

    log(f'Added {len(fs) - n} data for zero-shot labels')
    return fs, ls, kws, kwms, sims.astype(np.float32)


def calc_gradient_penalty(discriminator, real_feats, fake_feats, labels, lambda1=10.):
    b = real_feats.size(0)
    alpha = torch.rand(b, 1).to(real_feats.device)
    alpha = alpha.expand(real_feats.size())

    interpolates = alpha * real_feats + ((1 - alpha) * fake_feats)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates, labels)

    ones = torch.ones(disc_interpolates.size()).to(real_feats.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


def get_gan_hyper(lr, ndh, ngh, critic_iters, key_lambda, top_k, add_zero):
    gan_hyper = f'lr{lr}_ndh{ndh}_ngh{ngh}_diter{critic_iters}'

    if key_lambda > 0.:
        gan_hyper += f'_reg{key_lambda}_kw{top_k}'

    if add_zero:
        gan_hyper += '_z'

    return gan_hyper
