import torch
import torch.nn as nn
from torch.nn import init
from .utils import *


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(LSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def forward(self, x):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs


class CRF(nn.Module):
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size

        self.transition = nn.Parameter(tagset_size, tagset_size)

    def forward(self, hiddens, masks):
        score = torch.Tensor(self.tagset_size).fill_(-10000)
        score[:, TAG2ID['Start']] = 0.
        trans = self.transition.unsqueeze(0)
        for hidden, mask in zip(hiddens, masks):
            mask = mask.unsqueeze(1)
            emission = hidden.unsqueeze(2)
            _score = score.unsqueeze(1) + emission + trans  # 这个地方用加不用乘是直接取log了
            _score = log_sum_exp(_score)
            score = _score * mask + score * (1 - mask)
        score = log_sum_exp(score + self.transition[TAG2ID['End']])

        return score

    def score(self, hiddens, labels, masks):
        score = torch.Tensor(hiddens.shape[0]).fill_(0.)
        hiddens = hiddens.unsqueeze(3)
        trans = self.transition.unsqueeze(2)
        for t, (hidden, mask) in enumerate(zip(hiddens, masks)):
            emission = torch.cat([hidden[labels] for hidden, labels in zip(hidden, labels[t + 1])])
            _trans = torch.cat([trans[x] for x in zip(labels[t + 1], labels[t])])
            score += (emission + _trans) * mask

        last_tag = labels.gather(0, masks.sum(0).long().unsqueeze(0)).squeeze(0)
        score += self.transition[TAG2ID['End'], last_tag]
        return score

    def viterbi(self,hiddens,masks):
        # 前向传播
        point = torch.LongTensor()
        score = torch.Tensor(self.tagset_size).fill_(-10000)
        score[:,TAG2ID['Start']] = 0.
        for hidden, mask in zip(hiddens,masks):
            mask = mask.unsqueeze(1)
            _score = score.unsqueeze(1) + self.transition
            _score,_point = _score.max(2)
            score += hidden
            point = torch.cat((point,_point.unsqueeze(1)),1)
            score = _score * mask + score * (1-mask)
        score += self.transition[TAG2ID['End']]
        best_score, best_tag = torch.max(score, 1)

        # 后向传播
        point = point.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b]
            j = masks[:, b].sum().int()
            for _point in reversed(point[b][:j]):
                i = _point[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path

