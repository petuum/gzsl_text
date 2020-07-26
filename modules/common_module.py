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
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, MultiLabelMarginLoss
from torch.optim.optimizer import Optimizer

EPS = 1e-7


def get_loss_fn(loss='FOCAL', reduction='sum'):
    if loss.upper() == 'FOCAL':
        return FocalLoss2d(reduction=reduction)
    elif loss.upper() == 'BCE':
        return BCEWithLogitsLoss(reduction=reduction)
    elif loss.upper() == 'MULTI':
        return MultiLabelMarginLoss(reduction=reduction)
    else:
        ValueError(loss)


def add_emb_weight_decay(model, weight_decay, emb_weight_decay=1e-2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if name == 'emb.embed.weight':
            decay.append(param)
            continue
        else:
            no_decay.append(param)

    return [{'params': no_decay, 'weight_decay': weight_decay}, {'params': decay, 'weight_decay': emb_weight_decay}]


def get_optimizer(lr, model, freeze_emb=False, opt='ADAM', weight_decay=1e-4):
    if freeze_emb:
        for name, param in model.named_parameters():
            if name == 'emb.embed.weight':
                param.requires_grad = False
                break

    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt == 'ADAM':
        return optim.Adam(params, weight_decay=weight_decay, lr=lr)
    elif opt == 'SGD':
        return optim.SGD(params, weight_decay=weight_decay, lr=lr, momentum=0.9)
    else:
        raise ValueError("optimizer is SGD or ADAM, {} not recognized!".format(opt))


def get_scheduler(optimizer, num_epochs, ratios=(0.6, 0.8)):
    milestones = [int(num_epochs * r) for r in ratios]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)


class EmbLayer(nn.Module):
    def __init__(self, embed_size, vocab_size, pad_idx=0, use_emb=True, W=None, freeze_emb=True):
        super(EmbLayer, self).__init__()
        if use_emb:
            assert W is not None
            W = torch.from_numpy(W)
            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=pad_idx)
            self.embed.weight.data = W.clone()
            embed_size = W.size()[1]
        else:
            # add 1 to include PAD
            self.embed = nn.Embedding(vocab_size + 1, embed_size, padding_idx=pad_idx)

        self.embed_size = embed_size
        if freeze_emb:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        return self.embed(x)


class FocalLoss2d(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='sum', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, logits, targets):
        # inputs and targets are assumed to be Batch x Classes
        assert logits.size(0) == targets.size(0)
        assert logits.size(1) == targets.size(1)

        weight = None
        if self.weight is not None:
            weight = torch.from_numpy(self.weight)

        # compute the negative likelihood
        logpt = - F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight, reduce=False)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1. - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss

        if self.reduction == 'sum':
            return torch.sum(balanced_focal_loss)
        elif self.reduction == 'mean':
            return torch.mean(balanced_focal_loss)
        else:
            return balanced_focal_loss


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), e=1e-6, weight_decay=0.0):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))

        defaults = dict(lr=lr, betas=betas, e=e, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['betas']

                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                state['step'] += 1

                # lr_scheduled = group['lr']
                # update_with_lr = lr_scheduled * update
                # p.data.add_(-update_with_lr)

                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr']  # * math.sqrt(bias_correction2) / bias_correction1
                p.data.add_(-step_size * update)

        return loss
