import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer


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
    def __init__(self, embed_size, vocab_size, pad_idx=0, W=None, freeze_emb=True):
        super(EmbLayer, self).__init__()
        if W is not None:
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
                step_size = group['lr']
                p.data.add_(-step_size * update)

        return loss
