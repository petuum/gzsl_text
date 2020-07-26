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
from collections import defaultdict

import numpy as np
import torch
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from constant import FEATURE_DIR
from utils.helper import iterate_minibatch, inf_list_iterator, get_code_feat_list, log


def load_features(feat_path):
    log(f'Loading pretrained features from {feat_path}')
    with np.load(f'{FEATURE_DIR}/{feat_path}') as f:
        # features, labels, keywords, keywords masks
        fs, ls, kws, kwms = f['arr_0'], f['arr_1'], f['arr_2'], f['arr_3']
    return fs, ls, kws, kwms


def get_features_for_labels(model, label_indices):
    feat_path = model.pretrain_name.replace('.model', '.npz')
    feat_path = f'{FEATURE_DIR}/{feat_path}'
    assert os.path.exists(feat_path), f'Features should be at {feat_path}'

    log(f'Loading pretrained features from {feat_path}')
    with np.load(feat_path) as f:
        fs, ls = f['arr_0'], f['arr_1']

    fs = np.maximum(fs, 0)

    label_indices = set(label_indices)
    label_feats = defaultdict(list)
    for x, y in zip(fs, ls):
        if y in label_indices:
            label_feats[y].append(x)

    _, code_centroids = get_code_feat_list(fs, ls)
    centroids = np.zeros((model.eval_code_size, fs.shape[1]))

    for i, code in enumerate(code_centroids):
        centroids[code] = code_centroids[code]

    return label_feats, centroids


def save_features(model, train_x, train_y, train_masks, eval_code_size, device, train_keywords, k=50, thresh=0.1,
                  save_path=None):
    batches = iterate_minibatch(train_x, train_y, train_masks, eval_code_size, 20, shuffle=False, rtn_indices=True)

    fs = []
    ls = []
    kws = []
    kwms = []

    with torch.set_grad_enabled(False):
        model.eval()
        for batch in tqdm.tqdm(batches):
            x, y, mask, indices = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            direct_feats, attn_feats = model.forward_real_feats(x, mask)
            dfs, afs, ascore_s, label_indices = [], [], [], []

            for df, labels, data_idx in zip(direct_feats, y, indices):
                idx = labels.nonzero().squeeze(1)
                dfs.append(df[idx])
                label_indices.append(idx)

                if train_keywords is not None:
                    code_keywords = train_keywords[data_idx]
                    for code in idx:
                        m = np.zeros(k)
                        kw = np.zeros(k)
                        code = int(code)

                        if code in code_keywords:
                            keywords, scores = code_keywords[code][:2]
                            keywords = keywords[scores >= thresh][:k]
                            m[: len(keywords)] = scores[scores >= thresh][:k]
                            kw[: len(keywords)] = keywords

                            if len(keywords) == 0:
                                print("No keywords for", code, max(scores))

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

    np.savez(f'{FEATURE_DIR}/{save_path}', fs, ls, kws, kwms)
    return fs, ls, kws, kwms


def get_nearest_for_zero(eval_code_size, code_feat_list, label_emb, data, step_size=8):
    fs, ls, kws, kwms = data
    zeroshot_codes = []
    rest_codes = set(code_feat_list.keys())

    for i in range(eval_code_size):
        if i not in code_feat_list:
            zeroshot_codes.append(i)

    label_emb = label_emb.data.cpu().numpy()
    zeroshot_emb = label_emb[zeroshot_codes]
    cos_dist = cosine_similarity(zeroshot_emb, label_emb)

    n = len(fs)
    zeroshot_neighbors = dict()
    sims = np.ones(n)

    for z, dist in zip(zeroshot_codes, cos_dist):
        z_fs = []
        kw = []
        m = []
        sim = []
        for nei in np.argsort(-dist)[1:]:
            if nei in rest_codes:
                zeroshot_neighbors[z] = nei
                n_aug = len(code_feat_list[nei])
                nei_idx = code_feat_list[nei].l
                z_fs.append(fs[nei_idx])
                kw.append(kws[nei_idx])
                m.append(kwms[nei_idx])
                sim.append(np.ones(n_aug) * dist[nei])
                break

        s = len(fs)
        fs = np.vstack([fs] + z_fs)
        e = len(fs)

        kws = np.vstack([kws] + kw)
        kwms = np.vstack([kwms] + m)
        sims = np.concatenate([sims] + sim)

        data_idx = list(range(s, e))
        ls = np.concatenate([ls, np.ones(len(data_idx)) * z])

        code_feat_list[z] = inf_list_iterator(data_idx, step_size=step_size)

    log(f'Added {len(fs) - n} data for zero-shot labels')
    return fs, ls, kws, kwms, sims.astype(np.float32)
