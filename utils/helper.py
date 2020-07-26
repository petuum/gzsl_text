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
import random
import sys
from collections import Counter, defaultdict
from typing import Iterator

import numpy as np
import torch

from utils.metrics import all_metrics


def log(s, f=sys.stderr):
    if s[-1] != '\n':
        s += '\n'
    f.write(s)
    f.flush()


def seed(s=8888):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InfListIterator(Iterator):
    def __init__(self, l, step_size=1, shuffle=True):
        self.l = l
        self.step_size = step_size
        self.shuffle = shuffle
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.l)

    def __len__(self):
        return len(self.l)

    def __next__(self):
        if self.idx == len(self):
            if self.shuffle:
                random.shuffle(self.l)
            self.idx = 0

        end = min(self.idx + self.step_size, len(self))
        items = self.l[self.idx: end]
        self.idx = end
        return items


inf_list_iterator = InfListIterator


class CodeDataListIterator(object):
    def __init__(self, l, code):
        self.id_to_data = dict()
        for d in l:
            self.id_to_data[d[0]] = d

        self.code = code
        self.unused = set(self.id_to_data.keys())

    def sample(self, sampled):  # try to sample a unseen example
        rest = self.unused.difference(sampled)

        if len(rest) == 0:  # all data in this code are used, have to re-use
            self.unused = set(self.id_to_data.keys())
            rest = self.unused

        data_id = random.choice(list(rest))
        self.unused.remove(data_id)
        yield self.id_to_data[data_id]


def get_code_data_list(train_notes, train_labels, train_masks):
    code_data_list = defaultdict(list)
    x_id = 0
    for note, label, mask in zip(train_notes, train_labels, train_masks):
        for code in label:
            code_data_list[code].append((x_id, note, label, mask))
        x_id += 1

    code_data_iterator = dict()
    for code in code_data_list:
        code_data_iterator[code] = CodeDataListIterator(code_data_list[code], code)

    return code_data_iterator


def iterate_minibatch(train_notes, train_labels, train_masks, eval_code_size, batch_size=16, shuffle=True,
                      rtn_indices=False):
    assert len(train_notes) == len(train_labels)
    n = len(train_notes)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    num_batches = n // batch_size + (n % batch_size != 0)
    for i in range(num_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        y = np.zeros((len(batch_indices), eval_code_size))
        y_indices = np.zeros((len(batch_indices), eval_code_size), dtype=int) - 1

        for row, b_idx in enumerate(batch_indices):
            c_idx = train_labels[b_idx]
            y[row,  c_idx] = 1
            y_indices[row, :len(c_idx)] = c_idx

        y = np.asarray(y, dtype=np.float32)
        y_indices = np.asarray(y_indices, dtype=int)
        x = train_notes[batch_indices]
        mask = train_masks[batch_indices]
        if rtn_indices:
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), batch_indices
        else:
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(y_indices)


def simple_iterate_minibatch(train_feats, train_labels, batch_size=16, shuffle=True):
    assert len(train_feats) == len(train_labels)
    n = len(train_feats)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    num_batches = n // batch_size + (n % batch_size != 0)
    for i in range(num_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        yield batch_indices


def get_code_feat_list(x, y, step_size=8):
    code_data_list = defaultdict(list)
    code_centroids = dict()

    data_id = 0
    for xx, yy in zip(x, y):
        code_data_list[yy].append(data_id)
        data_id += 1
        if yy in code_centroids:
            code_centroids[yy] += xx
        else:
            code_centroids[yy] = copy.deepcopy(xx)

    for code in code_centroids:
        code_centroids[code] /= len(code_data_list[code])

    code_data_iterator = dict()
    for code in code_data_list:
        code_data_iterator[code] = inf_list_iterator(code_data_list[code], step_size=step_size)

    return code_data_iterator, code_centroids


def codes_to_index_labels(all_codes, truncate=False):
    if truncate:
        truncate_codes = set([code[:3] for code in all_codes])
        truncate_codes_to_labels = dict(zip(truncate_codes, np.arange(len(truncate_codes))))
        codes_to_labels = dict()
        for code in all_codes:
            codes_to_labels[code] = truncate_codes_to_labels[code[:3]]
        return codes_to_labels
    else:
        return dict(zip(sorted(all_codes), np.arange(len(all_codes))))


def log_eval_metrics(epoch, y_score, y_true, frequent_indices, few_shot_indices, zero_shot_indices,
                     train_losses=None, dev_losses=None, fz_only=False):
    if train_losses is not None and dev_losses is not None:
        log(f"Epoch {epoch}, train loss={np.mean(train_losses):.4f}, dev loss={np.mean(dev_losses):.4f}\n")

    if not fz_only:
        metrics = all_metrics(y_score, y_true)
        log(f"\tA: {metric_string(metrics)}\n")

        freq_metircs = all_metrics(y_score[:, frequent_indices], y_true[:, frequent_indices])
        log(f"\tS: {metric_string(freq_metircs)}\n")

    few_metircs = all_metrics(y_score[:, few_shot_indices], y_true[:, few_shot_indices])
    log(f"\tF: {metric_string(few_metircs)}\n")

    zero_metircs = all_metrics(y_score[:, zero_shot_indices], y_true[:, zero_shot_indices])
    log(f"\tZ: {metric_string(zero_metircs)}\n")

    if not fz_only:
        return metrics['f1_micro']
    else:
        return few_metircs['f1_micro'], zero_metircs['f1_micro']


def metric_string(metrics):
    mic_pre, mic_rec, mic_f1, mic_auc = metrics['pre_micro'], metrics['rec_micro'],\
                                        metrics['f1_micro'], metrics['auc_micro']
    mac_pre, mac_rec, mac_f1, mac_auc = metrics['pre_macro'], metrics['rec_macro'],\
                                        metrics['f1_macro'], metrics['auc_macro']
    auc_pr = metrics['auc_pr']

    string = f"mic pre={mic_pre * 100:.2f}, mic rec={mic_rec * 100:.2f}, " \
        f"mic f1={mic_f1 * 100:.2f}, mic auc={mic_auc * 100:.2f}, " \
        f"mac pre={mac_pre * 100:.2f}, mac rec={mac_rec * 100:.2f}, " \
        f"mac f1={mac_f1 * 100:.2f}, mac auc={mac_auc * 100:.2f}, " \
        f"auc-pr={auc_pr * 100:.2f}"

    return string


def label_code_counter(labels):
    code_counter = Counter()

    for l in labels:
        for code in l:
            code_counter[code] += 1
    return code_counter


def split_codes_by_count(train_labels, dev_labels, train_codes, dev_codes, thresh=5):
    frequent_codes = set()
    few_shot_codes = set()
    zero_shot_codes = set()
    code_counter = label_code_counter(train_labels)
    dev_code_counter = label_code_counter(dev_labels)

    for code in dev_codes:
        if dev_code_counter[code] < thresh:
            continue

        if code not in train_codes:
            zero_shot_codes.add(code)
        elif code_counter[code] <= 5:
            few_shot_codes.add(code)
        else:
            frequent_codes.add(code)

    return frequent_codes, few_shot_codes, zero_shot_codes, code_counter


def targets_to_count(codes_to_targets, code_counter):
    target_count = np.zeros(len(codes_to_targets), dtype=np.float32)
    for code in codes_to_targets:
        t = codes_to_targets[code]
        if code in code_counter:
            target_count[t] = code_counter[code]

    log(f'Imbalance ratio={target_count.max()}')
    return target_count


def code_to_indices(codes, codes_to_targets):
    return list(set([codes_to_targets[code] for code in codes]))
