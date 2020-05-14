import torch
import torch.nn as nn
import torch.nn.init
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def l1norm(matrix, dim, eps=1e-8):
    """
    l1 normalization
    """
    norm = torch.abs(matrix).sum(dim=dim, keepdim=True) + eps
    matrix = matrix / norm
    return matrix


def l2norm(matrix, dim, eps=1e-8):
    norm = torch.pow(matrix, matrix).sum(dim=dim, keepdim=True).sqrt() + eps
    matrix = matrix / norm
    return matrix


def func_attention(query, context, smooth, norm_func):
    """
    :param query:  (batch_size, query_length, d)
    :param context:  (batch_size, context_length, d)
    :param smooth: temperature parameter in softmax
    :return:
    """
    batch_size = query.size(0)
    query_length = query.size(1)
    context_length = context.size(1)

    # query_transpose: (batch_size, d, query_length)
    query_transpose = torch.transpose(context, 1, 2)

    # score: (batch_size, context_length, query_length)
    score = torch.bmm(context, query_transpose)

    # normalize score(for query)
    if norm_func == 'softmax':
        # score: (batch_size * context_length, query_length)
        score = score.view(batch_size * context_length, query_length)
        score = nn.Softmax()(score)

        # score: (batch_size , context_length, query_length)
        score = score.view(batch_size, context_length, query_length)
    elif norm_func == 'l1norm':
        score = l1norm(score, 2)
    elif norm_func == 'clipped_l1norm':
        score = nn.ReLU()(score)
        score = l1norm(score, 2)
    elif norm_func == 'clipped_leaky_l1norm':
        score = nn.LeakyReLU(0.1)(score)
        score = l1norm(score, 2)
    elif norm_func == 'l2norm':
        score = l2norm(score, 2)
    elif norm_func == 'clipped_l2norm':
        score = nn.ReLU()(score)
        score = l2norm(score, 2)
    elif norm_func == 'clipped_leaky_l2norm':
        score = nn.LeakyReLU(0.1)(score)
        score = l2norm(score, 2)
    elif norm_func == 'no_norm':
        pass
    else:
        raise ValueError("unknown first norm type: ", norm_func)

    # alignment function(softmax): get attention weights(for context)
    # score: (batch_size, query_length, context_length)
    score = torch.transpose(score, 1, 2).contiguous()
    score = score.view(batch_size * query_length, context_length)
    # attn: (batch_size, query_length, context_length)
    attn = nn.Softmax()(score * smooth)
    attn = attn.view(batch_size, query_length, context_length)

    # get weighted context vector
    # weighted_context: (batch_size, query_length, d)
    # (batch_size, query_length, context_length) * (batch_size, context_length, d)
    # -->(batch_size, query_length, d)
    weighted_context = torch.bmm(attn, context)

    return weighted_context, attn


class EncoderText(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional=False, text_norm=True):
        super(EncoderText, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)

        self.use_bi_gru = bidirectional
        self.text_norm = text_norm

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """
        :param x: (batch_size, seq_len)
        :param lengths: (batch_size, )
        :return: (batch_size, seq_len, embed_size)
        """
        # embed the words
        x = self.embed(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)

        # RNN forward propagate RNN
        packed_out, _ = self.gru(packed_x)
        # out: (batch_size, seq_len, num_directions * embed_size)
        out, out_len = pad_packed_sequence(packed_x, batch_first=True)

        if self.use_bi_gru:
            # out: (batch_size, seq_len, embed_size)
            out = (out[:, :, :out.size(2)//2] + out[:, :, out.size(2)//2:]) / 2

        # normalize the text representation vector in the joint embedding space
        if self.text_norm:
            out = l2norm(out, dim=-1)

        # return caption embedding: (batch_size, seq_len, embed_size)
        return out, out_len


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_size, embed_size, use_abs=False, img_norm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.use_abs = use_abs
        self.img_norm = img_norm

        self.fc = nn.Linear(img_size, embed_size)
        self.init_weights()

    def init_weights(self):
        """
        Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, img_features):
        """
        :param img_features: (batch_size, num_regions, row_img_features)
        :return: features: (batch_size, num_regions, img_features)
        """
        # embed precomputed img features into joint embedding space

        features = self.fc(img_features)

        # normalize in the joint embedding space
        if self.img_norm:
            features = l2norm(features, -1)

        if self.use_abs:
            features = torch.abs(features)

        return features


class GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, v):
        """
        :param v: (B, D, N)
        :return:
        """
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star


class CARRN(nn.Module):

    def __init__(self, img_size, hidden_size, vocab_size, word_embed_size, num_layers, use_abs, img_norm):
        super(CRN, self).__init__()
        self.base_img_enc = EncoderImagePrecomp(img_size, hidden_size,
                                                use_abs, img_norm)
        self.base_text_enc = EncoderText(vocab_size, word_embed_size,
                                         hidden_size, num_layers,
                                         bidirectional=False, text_norm=True)


