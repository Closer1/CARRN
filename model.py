import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.backends.cudnn as cudnn
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm

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

    batch_size = query.size(0)
    query_length = query.size(1)
    context_length = context.size(1)

    # query_transpose: (batch_size, d, query_length)
    query_transpose = torch.transpose(query, 1, 2)

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


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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
        out, out_len = pad_packed_sequence(packed_out, batch_first=True)

        if self.use_bi_gru:
            # out: (batch_size, seq_len, embed_size)
            out = (out[:, :, :out.size(2)//2] + out[:, :, out.size(2)//2:]) / 2

        # normalize the text representation vector in the joint embedding space
        if self.text_norm:
            out = l2norm(out, dim=-1)

        # return caption embedding: (batch_size, seq_len, embed_size)
        return out, out_len


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_size, embed_size, use_abs=False, img_norm=True):
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
    """
    from VSRN
    """
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


class CrossAttentionLayer(nn.Module):

    def __init__(self, hidden_size, smooth, norm_func, norm=True, activation_fun='relu'):
        super(CrossAttentionLayer, self).__init__()
        self.norm_func = norm_func
        self.smooth = smooth
        self.fc_img = nn.Linear(hidden_size, hidden_size)
        self.fc_txt = nn.Linear(hidden_size, hidden_size)

        self.norm = norm
        self.activation_fun = activation_fun

    def forward(self, txt_embed, img_embed):
        txt_attn_embed, attn_img = func_attention(txt_embed, img_embed, self.smooth, self.norm_func)
        img_attn_embed, attn_txt = func_attention(img_embed, txt_embed, self.smooth, self.norm_func)
        txt_attn_embed = self.fc_txt(txt_attn_embed)
        img_attn_embed = self.fc_img(img_attn_embed)

        if self.activation_fun == 'relu':
            txt_attn_output = F.relu(txt_attn_embed)
            img_attn_output = F.relu(img_attn_embed)
        elif self.activation_fun == 'gelu':
            txt_attn_output = gelu(txt_attn_embed)
            img_attn_output = gelu(img_attn_embed)
        elif self.activation_fun == 'no_activation_fun':
            txt_attn_output = txt_attn_embed
            img_attn_output = img_attn_embed
        else:
            raise ValueError('Unknown activation function :', self.activation_fun)

        if self.norm:
            txt_attn_output = l2norm(txt_attn_output, -1)
            img_attn_output = l2norm(img_attn_output, -1)

        return txt_attn_output, img_attn_output, attn_img, attn_txt


class CARRNEncoder(nn.Module):

    def __init__(self, img_size, hidden_size, use_abs, vocab_size,
                 word_embed_size, num_layers, bi_gru, smooth,
                 norm_func, norm, activation_func):
        super(CARRNEncoder, self).__init__()
        self.base_img_enc = EncoderImagePrecomp(img_size, hidden_size, use_abs, norm)
        self.base_text_enc = EncoderText(vocab_size, word_embed_size, hidden_size,
                                         num_layers, bi_gru, norm)

        self.GCN_1 = GCN(in_channels=hidden_size, inter_channels=hidden_size)
        self.GCN_2 = GCN(in_channels=hidden_size, inter_channels=hidden_size)
        self.GCN_3 = GCN(in_channels=hidden_size, inter_channels=hidden_size)
        self.GCN_4 = GCN(in_channels=hidden_size, inter_channels=hidden_size)

        self.cross_attn1 = CrossAttentionLayer(hidden_size, smooth, norm_func, norm, activation_func)

        self.cross_attn2 = CrossAttentionLayer(hidden_size, smooth, norm_func, norm, activation_func)

    def forward(self, captions, images, lengths):
        # embed captions and images into joint space
        raw_txt, _ = self.base_text_enc(captions, lengths)
        raw_img = self.base_img_enc(images)

        # object-level cross-attention
        txt_embed, img_embed, object_attn_img, object_attn_txt = self.cross_attn1(raw_txt, raw_img)

        # image object relation reasoning(object alignment)
        # GCN_img_embed : (batch_size, hidden_size, num_regions)

        gcn_img_embed = img_embed.permute(0, 2, 1)
        gcn_img_embed = self.GCN_1(gcn_img_embed)
        gcn_img_embed = self.GCN_2(gcn_img_embed)
        gcn_img_embed = self.GCN_3(gcn_img_embed)
        gcn_img_embed = self.GCN_4(gcn_img_embed)

        gcn_img_embed = gcn_img_embed.permute(0, 2, 1)
        gcn_img_embed = l2norm(gcn_img_embed, 2)

        # relation-level cross-attention(relation alignment)
        txt_embed, img_embed, relation_attn_img, relation_attn_txt = self.cross_attn2(txt_embed, gcn_img_embed)

        return txt_embed, img_embed


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, smooth, norm_func, agg_func, lambda_lse):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, smooth, norm_func)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, captions, cap_lens, smooth, norm_func, agg_func, lambda_lse):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, smooth, norm_func)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    
    def __init__(self, smooth, norm_func, agg_func, lambda_lse, margin=0, alpha=0.5, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.smooth = smooth
        self.norm_func = norm_func
        self.agg_func = agg_func
        self.lambda_lse = lambda_lse
        self.margin = margin
        self.alpha = alpha
        self.max_violation = max_violation

    def forward(self, images, captions, cap_lengths):
        # compute image-sentence score matrix
        t2i_scores = xattn_score_t2i(images, captions, cap_lengths, self.smooth,
                                     self.norm_func, self.agg_func, self.lambda_lse)
        i2t_scores = xattn_score_i2t(images, captions, cap_lengths, self.smooth,
                                     self.norm_func, self.agg_func, self.lambda_lse)

        scores = self.alpha * t2i_scores + (1 - self.alpha) * i2t_scores

        diagonal = scores.diag().view(images.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class CARRN(object):
    def __init__(self, opt):
        # build Models
        self.grad_clip = opt.grad_clip
        self.encoder = CARRNEncoder(opt.img_size, opt.hidden_size, opt.use_abs, opt.vocab_size,
                                    opt.word_embed_size, opt.num_layers, opt.bi_gru,
                                    opt.lambda_softmax, opt.norm_func, opt.norm, opt.activation_func)
        if torch.cuda.is_available():
            self.encoder.cuda()
            cudnn.benchmark = True

        self.criterion = ContrastiveLoss(opt.lambda_softmax, opt.norm_func, opt.agg_func, opt.lambda_lse,
                                         opt.margin, opt.alpha, opt.max_violation)

        params = list(self.encoder.parameters())

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = self.encoder.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.ecoder.load_state_dict(state_dict)

    def train_start(self):
        """
        switch to train mode
        """
        self.encoder.train()

    def val_start(self):
        """
        switch to evaluate mode
        """
        self.enocder.eval()

    def forward_emb(self, images, captions, lengths):
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        txt_embed, img_embed = self.encoder(captions, images, lengths)

        return txt_embed, img_embed

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """
        Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """
        One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, lengths)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
