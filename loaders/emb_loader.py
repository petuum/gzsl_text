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
import pickle

from constant import EMBEDDING_PATH
from utils.helper import log
import numpy as np
import torch


def load_word_embedding():
    if os.path.exists(EMBEDDING_PATH):
        with open(EMBEDDING_PATH, 'rb') as f:
            word_embedding = pickle.load(f)
            code_to_idx = pickle.load(f)
        log(f'W emb size {word_embedding.shape}')
        return word_embedding, code_to_idx
    else:
        log(f'Please download ')
        raise ValueError(f'Please download embedding file to {EMBEDDING_PATH}')


def init_emb_and_code_input(codes_to_targets):
    word_emb, code_to_idx = load_word_embedding()

    code_idx_matrix = []
    code_idx_mask = []
    max_code_len = max([len(idx) for idx in code_to_idx.values()])
    log(f'Max code description {max_code_len}')

    targets_to_code = dict((v, k) for k, v in codes_to_targets.items())
    for target in targets_to_code:
        code_idx = code_to_idx[targets_to_code[target]]
        mask = np.zeros(max_code_len)
        mask[: len(code_idx)] = 1
        if len(code_idx) < max_code_len:
            code_idx = code_idx + [0] * (max_code_len - len(code_idx))

        code_idx_matrix.append(code_idx)
        code_idx_mask.append(mask)

    code_idx_mask = np.asarray(code_idx_mask, dtype=np.float32)
    code_idx_mask = torch.from_numpy(code_idx_mask)
    code_idx_matrix = np.asarray(code_idx_matrix, dtype=int)
    code_idx_matrix = torch.from_numpy(code_idx_matrix)
    return word_emb, code_idx_matrix, code_idx_mask
