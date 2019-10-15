import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_mask(mask):
    return mask / (mask.sum(1, keepdim=True) + 1e-7)


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
        self.init_weights(word_emb)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        if freeze_emb:
            self.encoder.weight.requires_grad = False

    def init_weights(self, word_emb):
        self.encoder.weight.data = torch.from_numpy(word_emb)

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

    def forward_enc(self, input, input_mask, init_hidden=None):
        enc_emb = self.forward_emb(input, input_mask)
        outputs, hn = self.rnn(enc_emb, self.init_hidden_states(init_hidden))
        outputs, lengthes = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs.masked_fill(outputs == 0., float('-inf'))
        hn = torch.max(outputs, dim=1)[0]
        return hn


class LinearKeywordsDecoder(nn.Module):
    def __init__(self, feat_size, emb_size, word_emb):
        super(LinearKeywordsDecoder, self).__init__()
        self.register_buffer('word_emb', word_emb)
        self.proj_fc = nn.Linear(feat_size, emb_size, bias=True)
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

        targets[index, keyword_indices] = keyword_masks
        loss = torch.log_softmax(logits, -1) * targets
        loss = torch.sum(-loss, -1)

        return loss


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_size, label_size, hidden_size, output_size):
        super(ConditionalGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_size + label_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, noise, label):
        x = torch.cat([noise, label], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, feat_size, label_size, hidden_size, output_size=1):
        super(ConditionalDiscriminator, self).__init__()
        self.fc1 = nn.Linear(feat_size + label_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, feat, label):
        x = torch.cat([feat, label], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x.squeeze()
