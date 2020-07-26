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


import torch.nn as nn
import torch


class RNNLabelEncoder(nn.Module):
    def __init__(self, word_emb, nlayers=1, dropout=0., rnn_type='GRU', tie_weights=True, freeze_emb=True):
        super(RNNLabelEncoder, self).__init__()

        ntoken, ninp = word_emb.shape

        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)

        rnn_type = rnn_type.upper()
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn = getattr(nn, rnn_type)(ninp, ninp, nlayers, dropout=dropout, batch_first=True)

        self.decoder = nn.Linear(ninp, ntoken, bias=False)
        if word_emb is not None:
            self.init_weights(word_emb)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nlayers = nlayers

        if freeze_emb:
            self.encoder.weight.requires_grad = False

    def init_weights(self, word_emb):
        initrange = 0.1
        self.encoder.weight.data = torch.from_numpy(word_emb)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden_states(self, init_hidden=None):
        if init_hidden is None:
            return init_hidden
        if self.rnn_type == 'LSTM':
            return init_hidden.unsqueeze(0), init_hidden.unsqueeze(0)
        return init_hidden.unsqueeze(0)

    def forward_emb(self, input, mask):
        emb = self.encoder(input)
        emb = torch.mul(emb, mask.unsqueeze(2))
        emb = self.drop(emb)

        lengths = torch.sum(mask, dim=1)
        emb = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        return emb

    def forward_enc(self, input, input_mask, init_hidden=None, training=True, pool_mode='last'):
        enc_emb = self.forward_emb(input, input_mask)
        outputs, hn = self.rnn(enc_emb, self.init_hidden_states(init_hidden))
        outputs, lengthes = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if training:
            return hn

        if pool_mode == 'mean':
            lengthes = lengthes.to(outputs.device)
            hn = torch.sum(outputs, dim=1) / lengthes.unsqueeze(1).float()
            return hn
        elif pool_mode == 'max':
            outputs = outputs.masked_fill(outputs == 0., float('-inf'))
            hn = torch.max(outputs, dim=1)[0]
            return hn

        if self.rnn_type == 'LSTM':
            hn = hn[0]

        return hn.squeeze()
