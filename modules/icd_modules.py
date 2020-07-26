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
from torch.nn.init import xavier_uniform_

from collections import OrderedDict
from utils.helper import log
from modules.common_module import EmbLayer
from modules.graph_module import ConvGraphEncoder, GraphGatedEncoder


def get_graph_encoder(graph_encoder='conv'):
    if graph_encoder == 'conv':
        return ConvGraphEncoder
    elif graph_encoder == 'gate':
        return GraphGatedEncoder
    else:
        return None


class ConvLabelAttnModel(nn.Module):
    def __init__(self, word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors, loss_fn, num_filters=200,
                 kernel_sizes=(10,), eval_code_size=None, label_hidden_size=200, graph_encoder='conv',
                 target_count=None, C=0.):

        super(ConvLabelAttnModel, self).__init__()

        self.name = f"convattn_{graph_encoder}gnn"

        self.n_nodes = len(code_idx_matrix)
        self.register_buffer('code_idx_matrix', code_idx_matrix)
        self.register_buffer('code_idx_mask', code_idx_mask)
        self.register_buffer('adj_matrix', adj_matrix)
        self.register_buffer('num_neighbors', num_neighbors)

        self.loss_fn = loss_fn
        self.eval_code_size = eval_code_size

        self.emb = EmbLayer(word_emb.shape[1], word_emb.shape[0], W=word_emb, freeze_emb=False)
        self.emb_drop = nn.Dropout(p=0.5)
        self.embed_size = self.emb.embed_size

        self.feat_size = self.embed_size
        self.attn_drop = nn.Dropout(p=0.2)

        graph_encoder = get_graph_encoder(graph_encoder)
        if graph_encoder is not None:
            log(f'Using {graph_encoder} for encoding ICD hierarchy...')
            self.graph_label_encoder = graph_encoder(self.embed_size, label_hidden_size, self.n_nodes)
            self.feat_size += label_hidden_size
        else:
            self.graph_label_encoder = None
        self.output_proj = nn.Linear(num_filters, self.feat_size)

        xavier_uniform_(self.output_proj.weight)

        self.conv_modules = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_modules.append(nn.Conv1d(self.embed_size, num_filters, kernel_size=kernel_size))
            xavier_uniform_(self.conv_modules[-1].weight)
        self.proj = nn.Linear(num_filters, self.embed_size)

        xavier_uniform_(self.proj.weight)

        if target_count is not None:
            class_margin = torch.from_numpy(target_count) ** 0.25
            class_margin = class_margin.masked_fill(class_margin == 0, 1)
            self.register_buffer('class_margin', 1.0 / class_margin)
            self.C = C
            self.name += f'_cm{int(self.C)}'
        else:
            self.class_margin = None
            self.C = 0

        self.name += '.model'

    def state_dict_to_save(self):
        state_dict = self.state_dict()
        state_dict_save = OrderedDict()
        for k in state_dict:
            if k in ['code_idx_matrix', 'code_idx_mask', 'adj_matrix', 'num_neighbors']:
                continue
            state_dict_save[k] = state_dict[k]
        return state_dict_save

    def conv_label_embedding(self):
        inp = self.emb(self.code_idx_matrix)  # L x T x E
        inp = torch.mul(inp, self.code_idx_mask.unsqueeze(2))

        label_emb = self.label_conv(inp.transpose(1, 2))  # L x E x T
        label_emb = torch.relu(label_emb)  # .transpose(1, 2)
        label_emb = self.label_pool(label_emb).squeeze()  # L x E
        return label_emb

    def average_label_embedding(self):
        label_emb = self.emb(self.code_idx_matrix).transpose(1, 2).matmul(self.code_idx_mask.unsqueeze(2))
        label_emb = torch.div(label_emb.squeeze(2), torch.sum(self.code_idx_mask, dim=-1).unsqueeze(1))
        return label_emb

    def get_label_embedding(self):
        avg_label_emb = self.average_label_embedding()

        if self.graph_label_encoder is not None:
            graph_label_emb = self.graph_label_encoder(avg_label_emb, self.adj_matrix, self.num_neighbors)
        else:
            graph_label_emb = avg_label_emb

        return avg_label_emb, graph_label_emb

    def conv_attn(self, inp, label_emb):
        outputs = []
        attn_scores = []
        label_emb = label_emb[:self.eval_code_size]

        conv = self.conv_modules[0]
        x = conv(inp.transpose(1, 2))  # B x F x T
        x = torch.relu(x).transpose(1, 2)  # B x T x F

        proj_x = self.proj(x)  # B x T x d
        proj_x = torch.tanh(proj_x)

        attn_score = torch.matmul(proj_x, label_emb.t()).transpose(1, 2)  # B x L x T

        attn_score = F.softmax(attn_score, dim=2)  # B x L x T
        attn_score = self.attn_drop(attn_score)

        attn_output = attn_score.matmul(x)  # B x L x F

        outputs.append(attn_output)
        attn_scores.append(attn_score)

        output = torch.cat(outputs, dim=-1)
        attn_scores = torch.cat(attn_scores, dim=-1)

        return output, attn_scores

    def forward_direct_feats(self, x, label_emb, mask=None):
        x = self.emb(x)
        if mask is not None:
            x = torch.mul(x, mask.unsqueeze(2))

        x = self.emb_drop(x)
        attn_output, attn_scores = self.conv_attn(x, label_emb)
        direct_feats = self.output_proj(attn_output)
        return direct_feats, attn_output, attn_scores

    def forward_loss(self, logits, y):
        # forward loss
        b = y.size(0)
        if self.class_margin is not None and self.C > 0.:
            # LDAM loss: subtract logits for positive classes by margin
            logits_for_loss = logits - y * self.class_margin * self.C
        else:
            logits_for_loss = logits

        loss = self.loss_fn(logits_for_loss, y) / b
        return loss

    def forward(self, x, y, mask=None):
        label_emb, graph_label_emb = self.get_label_embedding()
        proj_output, _, _ = self.forward_direct_feats(x, label_emb, mask)
        proj_output = torch.relu(proj_output)

        proj_output = proj_output[:, :self.eval_code_size]
        graph_label_emb = graph_label_emb[:self.eval_code_size]

        logits = torch.mul(graph_label_emb, proj_output).sum(dim=2)

        # forward loss
        loss = self.forward_loss(logits, y)
        return logits, loss


class ConvLabelAttnGAN(ConvLabelAttnModel):
    def __init__(self, word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors, loss_fn, num_filters=200,
                 kernel_sizes=(10, ), eval_code_size=None, label_hidden_size=200, graph_encoder='conv',
                 noise_size=256, target_count=None, C=0.):

        super(ConvLabelAttnGAN, self).__init__(word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors,
                                               loss_fn, num_filters, kernel_sizes, eval_code_size, label_hidden_size,
                                               graph_encoder, target_count=target_count, C=C)

        self.pretrain_name = self.name
        self.froze_avg_label_emb = None
        self.froze_graph_label_emb = None
        self.noise_size = noise_size
        self.name = 'gan_' + self.name
        self.output_fc = nn.Linear(self.feat_size, self.code_idx_matrix.size(0), bias=False)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def update_class_margin(self, target_count, C):
        assert self.class_margin is not None and self.C > 0
        class_margin = torch.from_numpy(target_count) ** 0.25
        class_margin = 1.0 / class_margin.masked_fill(class_margin == 0, 1)
        self.class_margin = class_margin.to(self.class_margin.device)
        self.C = C

    def init_output_fc(self, label_emb):
        with torch.no_grad():
            self.output_fc.weight.data.copy_(label_emb.data)

    def get_froze_label_emb(self):
        if self.froze_avg_label_emb is None:
            self.froze_avg_label_emb = self.average_label_embedding()
        if self.froze_graph_label_emb is None:
            if self.graph_label_encoder is not None:
                self.froze_graph_label_emb = self.graph_label_encoder(self.froze_avg_label_emb, self.adj_matrix,
                                                                      self.num_neighbors)
            else:
                self.froze_graph_label_emb = self.froze_avg_label_emb
        return self.froze_avg_label_emb, self.froze_graph_label_emb

    def forward_real_feats(self, x, mask=None):
        label_emb, graph_label_emb = self.get_froze_label_emb()
        direct_feats, attn_feats, attn_scores = self.forward_direct_feats(x, label_emb, mask)
        return direct_feats, attn_feats

    def forward_final(self, feats, labels):
        if isinstance(labels, list):
            labels = torch.LongTensor(labels).to(feats.device)

        label_indices = labels
        y = torch.ones_like(labels).to(labels.device).float()

        label_emb = self.output_fc.weight
        logits = torch.mul(label_emb[label_indices], feats).sum(dim=-1)

        if self.class_margin is not None and self.C > 0.:    # subtract logits for positive classes by margin
            logits_for_loss = logits - self.class_margin[label_indices] * self.C
        else:
            logits_for_loss = logits

        loss = self.loss_fn(logits_for_loss, y)
        return loss

    def forward(self, x, y, mask=None, label_indices=None):
        label_emb, graph_label_emb = self.get_froze_label_emb()

        if label_indices is not None:
            label_emb = label_emb[label_indices]
            y = y[:, label_indices]
            if self.class_margin is not None:
                class_margin = self.class_margin[label_indices]
        else:
            class_margin = self.class_margin

        proj_output, _, _ = self.forward_direct_feats(x, label_emb, mask)
        proj_output = torch.relu(proj_output)

        label_emb = self.output_fc.weight
        if label_indices is not None:
            label_emb = label_emb[label_indices]

        proj_output = proj_output[:, :self.eval_code_size]
        label_emb = label_emb[:self.eval_code_size]

        logits = torch.mul(label_emb, proj_output).sum(dim=2)

        if self.class_margin is not None and self.C > 0.:
            logits_for_loss = logits - y * class_margin * self.C
        else:
            logits_for_loss = logits

        # forward loss
        b = y.size(0)
        loss = self.loss_fn(logits_for_loss, y) / b
        return logits, loss

    def load_pretrained_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()
        for k in pretrained_dict:
            if k in model_dict:
                model_dict[k] = pretrained_dict[k]
        self.load_state_dict(model_dict)

        for name, param in self.named_parameters():
            if name in pretrained_dict:
                param.requires_grad = False
                log(f'Freeze {name} in training...')
