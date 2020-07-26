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


import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_mask(mask):
    return mask / (mask.sum(1, keepdim=True) + 1e-7)


class LinearKeywordsDecoder(nn.Module):
    def __init__(self, feat_size, emb_size, word_emb, binary=False):
        super(LinearKeywordsDecoder, self).__init__()
        self.register_buffer('word_emb', word_emb)
        self.proj_fc = nn.Linear(feat_size, emb_size, bias=True)
        self.binary = binary
        self.V = word_emb.size(0)

    def predict(self, feats):
        x = self.proj_fc(feats)
        # x = torch.relu(x)
        logits = torch.matmul(x, self.word_emb.t())  # B x V
        return logits

    def forward(self, feats, keyword_indices, keyword_masks, label_emb=None):
        x = self.proj_fc(feats)
        # x = torch.relu(x)

        logits = torch.matmul(x, self.word_emb.t())  # B x V
        b, k = keyword_indices.size()

        targets = torch.zeros(b, self.V).float().to(feats.device)
        index = torch.arange(b)[:, None].to(feats.device)
        keyword_masks = normalize_mask(keyword_masks)

        if self.binary:
            targets[index, keyword_indices] = 1
            loss = F.binary_cross_entropy_with_logits(logits[:, 1:], targets[:, 1:], reduction='none')  # B x V-1
            loss = torch.mean(loss, -1)
        else:
            targets[index, keyword_indices] = keyword_masks
            loss = torch.log_softmax(logits, -1) * targets
            loss = torch.sum(-loss, -1)

        return loss


def load_decoder(decoder, feat_size, embed_size, keyword_emb):
    if decoder == 'linear':
        keyword_predictor = LinearKeywordsDecoder(feat_size, embed_size, keyword_emb)
    else:
        raise ValueError('Wrong decoder')
    return keyword_predictor
