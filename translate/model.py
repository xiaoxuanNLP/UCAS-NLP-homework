import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from .config import DEVICE


class Embedding(nn.Module):
    def __init__(self, vocab, dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.dim = dim

    def forward(self, x):
        return self.embedding(x)
        # return self.embedding(x) * math.sqrt(self.dim)  有人在embedding处做scale


class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_len):
        super(PositionEmbedding, self).__init__()

        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        absolute_position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        middle_item = torch.exp(
            torch.arange(0., dim, 2, device=DEVICE) * -(math.log(10000.0) / dim))  # 在做乘除法的时候还有幂次操作，要先改成加减操作或着惩处操作

        position[:, 0::2] = torch.sin(absolute_position * middle_item)
        position[:, 1::2] = torch.cos(absolute_position * middle_item)

        position = position.unsqueeze(0)
        self.register_buffer('position', position)  # 被注册进buffer后将不参与更新

    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)


def attention(query, key, value, mask=None, dropout=None):
    dim_k = query.shape[-1]

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)

    attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    return torch.matmul(attn, value), attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert dim % h == 0

        self.dim_k = dim // h  # 虽然叫多头注意力机制，但是隐藏层的维度不能增加，这里就是把维度按照头数拆开
        self.h = h

        self.linears = clones(nn.Linear(dim, dim), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batchsize = query.shape[0]
        query, key, value = [l(x).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batchsize, -1, self.h * self.d_k)

        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class AddNorm(nn.Module):
    def __init__(self, size, dropout):
        super(AddNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, connect_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(dim, connect_dim)
        self.W2 = nn.Linear(connect_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W2(self.dropout(F.relu(self.W1(x))))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.add_normal = clones(AddNorm(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.add_normal[0](x, lambda x: self.attn(x, x, x, mask))
        return self.add_normal[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.attn = attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.add_norm = clones(AddNorm(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory

        x = self.add_norm[0](x, lambda x: self.attn(x, x, x, tgt_mask))
        x = self.add_norm[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.add_norm[2](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, mask):
        return self.encoder(self.src_embed(src), mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

class Generator(nn.Module):
    def __init__(self,vocab,dim):
        super(Generator,self).__init__()
        self.linear = nn.Linear(dim,vocab)

    def forward(self,x):
        return F.log_softmax(self.linear(x),dim=-1)


