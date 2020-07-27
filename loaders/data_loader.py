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
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils import data

from constant import ICD_CODE_HIERARCHY_PATH, ICD_CODE_DESC_DATA_PATH, SPLIT_DIR, PROCESSED_DIR, VOCAB_PATH
from utils.helper import log


def load_vocab():
    vocab_to_ix = dict()
    with open(VOCAB_PATH) as f:
        i = 0
        for line in f:
            vocab_to_ix[line.strip()] = i
            i += 1
    return vocab_to_ix


def read_hier_codes():
    hier_codes = dict()
    with open(ICD_CODE_HIERARCHY_PATH) as f:
        for line in f:
            code, neighbors = line[:-1].split(':')
            hier_codes[code] = neighbors.split(' ')
    return hier_codes


def load_adj_matrix(codes_to_targets):
    hier_codes = read_hier_codes()

    codes_to_extended_targets = copy.copy(codes_to_targets)
    for code in hier_codes:
        if code not in codes_to_extended_targets:
            codes_to_extended_targets[code] = len(codes_to_extended_targets)

    n = len(codes_to_extended_targets)
    adj_matrix = np.zeros((n, n))
    missed = 0
    codes_to_parents = defaultdict(list)
    for code in codes_to_extended_targets:
        if code in hier_codes:
            for neighbor in hier_codes[code]:
                c_idx = codes_to_extended_targets[code]
                n_idx = codes_to_extended_targets[neighbor]
                if neighbor in code:    # parent
                    codes_to_parents[c_idx].append(n_idx)
                adj_matrix[c_idx, n_idx] = 1
                adj_matrix[n_idx, c_idx] = 1
        else:
            missed += 1

    log(f'Graph has {n} nodes, {missed} codes has no hierarchy...')
    return codes_to_extended_targets, adj_matrix, codes_to_parents


def preload_data(train_notes, train_labels, codes_to_targets, max_note_len=2000, idx_offset=1, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        # if cache exists
        with np.load(save_path, allow_pickle=True) as f:
            return f['arr_0'], f['arr_1'], f['arr_2']

    x = []
    y = []
    mask = []
    row = 0
    for note_path, labels in zip(train_notes, train_labels):
        code_idx = [codes_to_targets[code] for code in labels]
        y.append(code_idx)
        m = np.ones(max_note_len)
        xx = np.zeros(max_note_len)
        note = np.load(note_path) + idx_offset  # shift 1 for padding

        if len(note) < max_note_len:
            m[len(note):] = 0
            xx[:len(note)] = note
        else:
            xx = note[:max_note_len]

        x.append(xx)
        mask.append(m)
        row += 1

    x = np.vstack(x).astype(int)
    y = np.asarray(y)
    mask = np.vstack(mask).astype(np.float32)

    if save_path is not None:
        log(f'Saving training data cache to {save_path}...')
        np.savez(save_path, x, y, mask)

    return x, y, mask


def prepare_code_data(model, device, to_torch=False):
    code_desc = model.code_idx_matrix.data.cpu().numpy()
    code_mask = model.code_idx_mask.data.cpu().numpy()

    if os.path.exists(ICD_CODE_DESC_DATA_PATH):
        with np.load(ICD_CODE_DESC_DATA_PATH) as f:
            all_words_indices, word_emb = f['arr_0'], f['arr_1']
    else:
        all_words_indices = np.sort(np.unique(code_desc))
        word_emb = model.emb.embed.weight[all_words_indices].data.cpu().numpy()
        np.savez(ICD_CODE_DESC_DATA_PATH, all_words_indices, word_emb)

    log(f'In total {len(all_words_indices)} words for labels...')

    word_idx_to_keyword_idx = dict(zip(all_words_indices, np.arange(len(all_words_indices))))
    assert word_idx_to_keyword_idx[0] == 0, "Padding index should match"
    code_desc = np.asarray([[word_idx_to_keyword_idx[w] for w in kw] for kw in code_desc], dtype=int)

    if to_torch:
        code_desc = torch.from_numpy(code_desc).to(device)
        code_mask = torch.from_numpy(code_mask).to(device)

    return code_desc, code_mask, word_emb


class Dataset(data.Dataset):
    def __init__(self, mode, thresh=5):
        names = []
        with open(f'{SPLIT_DIR}/{mode}_thresh{thresh}_hdam_ids.txt') as f:
            for line in f:
                names.append(line[:-1])

        names = sorted(names)
        self.label_paths, self.note_paths = [], []
        self.names = names

        for name in names:
            self.label_paths.append(os.path.join(os.path.join(PROCESSED_DIR, 'label_files/'), f'{name}_labels.txt'))
            self.note_paths.append(os.path.join(os.path.join(PROCESSED_DIR, 'numpy_vectors/'), f'{name}_notes.npy'))

        assert len(self.label_paths) == len(self.note_paths), \
            'len of labels files is {} and is not equal to len of notes files which is {}'.format(len(self.label_paths),
                                                                                                  len(self.note_paths))
        self.mode = mode
        self.labels = get_labels(self.label_paths)

    def get_data(self):
        return np.asarray(self.note_paths), np.asarray(self.labels)

    def get_all_codes(self):
        return set([c for l in self.labels for c in l])

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.label_paths)

    def __getitem__(self, index):
        """ Generate one sample of data """
        # Load data
        X = np.load(self.note_paths[index])
        # get label
        y = self.labels[index]
        return X, y


def get_labels(label_paths):
    labels_dicts = []
    for index in range(len(label_paths)):
        label_path = label_paths[index]
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                labels = f.readlines()
            new_labels = []
            for label in labels:
                label = label.split(',')[1].strip()

                if label == '71970':
                    label = '7197'

                if label == 'V122':
                    continue
                new_labels.append(label)

            labels_dicts.append(new_labels)

    return labels_dicts
