import torch
import torch.nn as nn
from torch.nn import init
from utils import *


class LSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(LSTM_CRF, self).__init__()
        self.lstm = LSTM(embedding_dim, hidden_dim, dropout, word2id, tag2id)
        self.crf = CRF(len(tag2id))

    def forward(self, x, mask, label):
        hiddens = self.lstm(x, mask)
        hiddens = hiddens.transpose(0, 1)
        mask = mask.transpose(0, 1)
        Z = self.crf(hiddens, mask)
        score = self.crf.score(hiddens, label, mask)
        # print("Z = ", Z)
        # print("score = ", score)
        return torch.mean(Z - score)

    def decode(self, x, mask):
        hiddens = self.lstm(x, mask)
        hiddens = hiddens.transpose(0, 1)
        mask = mask.transpose(0, 1)
        return self.crf.viterbi(hiddens, mask)


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

    def forward(self, x, mask):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        outputs *= mask.unsqueeze(2)
        # print("x = ",x.shape)
        # print("mask = ",mask.shape)
        # print("outputs = ",outputs.shape)
        # print("outputs * mask = ",(outputs*mask.unsqueeze(2)).shape)
        return outputs


class CRF(nn.Module):
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        # self.batch_size = 0
        self.tagset_size = tagset_size

        self.transition = nn.Parameter(torch.randn(tagset_size, tagset_size))  # 第1个维度是目标，第2个维度是出发
        self.transition.data[TAG2ID['Start'], :] = -10000
        self.transition.data[:, TAG2ID['End']] = -10000
        self.transition.data[:, TAG2ID['PAD']] = -10000
        self.transition.data[TAG2ID['PAD'], :] = -10000
        self.transition.data[TAG2ID['PAD'], TAG2ID['End']] = 0
        self.transition.data[TAG2ID['PAD'], TAG2ID['PAD']] = 0

    def forward(self, hiddens, masks):
        # 这里hiddens和masks都是[length batch tags]的shape
        # print("hiddens = ",hiddens.shape)
        # print("hiddens = ",hiddens.shape)
        # print("masks = ",masks.shape)
        score = torch.Tensor(hiddens.shape[1], self.tagset_size).fill_(-10000).cuda()
        score[:, TAG2ID['Start']] = 0.
        trans = self.transition.unsqueeze(0)
        for hidden, mask in zip(hiddens, masks):
            mask = mask.unsqueeze(1)
            emission = hidden.unsqueeze(2)
            _score = score.unsqueeze(1) + emission + trans  # 这个地方用加不用乘是直接取指数加
            _score = log_sum_exp(_score)  # 这个地方，行表示目标，列表示起始状态，因为是viterbi算法，需要记录这个最大的概率从哪来的，所以才将列向量表示为起始
            # print("_score = ",_score.shape)
            # print("mask = ",mask.shape)
            score = _score * (1 - mask) + score * mask
        score = log_sum_exp(score + self.transition[TAG2ID['End']])

        return score

    def score(self, hiddens, labels, masks):
        # score = torch.Tensor(hiddens.shape[1]).fill_(0.).cuda()
        # labels = labels.transpose(0, 1)
        #
        # for t, (hidden, mask) in enumerate(zip(hiddens, masks)):
        #     if t == 0:
        #         _trans = torch.stack(
        #             [self.transition[label_now, TAG2ID['Start']] for label_now in labels[0]])
        #     else:
        #         _trans = torch.stack(
        #             [self.transition[label_now, label_last] for label_now, label_last in zip(labels[t], labels[t - 1])])
        #     # print("_trans = ",_trans)
        #
        #     emission = torch.stack([_h[_label] for _h,_label in zip(hidden,labels[t])])
        #     score = (emission + _trans) * (1-mask) + score * mask
        # # the_last_score = torch.stack([self.transition[TAG2ID['End'], label] for label in labels[-1,:]])
        # the_last_label = labels.gather(0,((1-masks).sum(0)-1).long().unsqueeze(0)).squeeze(0)
        # # print("the_last_label = ",the_last_label)
        # the_last_score = self.transition[TAG2ID['End'],the_last_label]
        # # print("the_last_score = ",the_last_score)
        # score += the_last_score
        #
        # return score
        score = torch.Tensor(hiddens.shape[1]).fill_(0.).cuda()
        hiddens = hiddens.unsqueeze(3)  # [L, B, C, 1]
        trans = self.transition.unsqueeze(2)  # [C, C, 1]
        # masks = masks.transpose(0,1)
        # print("hiddens = ",hiddens.shape)
        # print("masks = ",masks.shape)
        # print("trans = ",trans.shape)
        labels = labels.transpose(0,1)
        label_start = (torch.ones(1,labels.shape[1])*TAG2ID['Start']).long().cuda()
        # print("label_start = ",label_start)
        labels = torch.cat((label_start,labels))
        # print("labels = ",labels.shape)
        # print("labels = ",labels.shape)
        for t, (_h, _mask) in enumerate(zip(hiddens, masks)):
            _emit = torch.cat([_h[labels] for _h, labels in zip(_h, labels[t + 1])])
            _trans = torch.cat([trans[x] for x in zip(labels[t + 1], labels[t])])
            # print("_emit = ",_emit.shape)
            # print("_trans = ",_trans.shape)
            score += (_emit + _trans) * (1-_mask)
        last_tag = labels.gather(0, (1-masks).sum(0).long().unsqueeze(0)).squeeze(0)
        # print("self.transition[TAG2ID['End'], last_tag] = ",self.transition[TAG2ID['End'], last_tag])
        score += self.transition[TAG2ID['End'], last_tag]
        return score


    def viterbi(self, hiddens, masks):
        # 前向传播

        # print("hiddens = ",hiddens.shape)
        # print("masks = ",masks.shape)
        point = torch.LongTensor().cuda()
        score = torch.Tensor(hiddens.shape[1], self.tagset_size).fill_(-10000).cuda()
        score[:, TAG2ID['Start']] = 0.
        for hidden, mask in zip(hiddens, masks):
            mask = mask.unsqueeze(1)
            _score = score.unsqueeze(1) + self.transition
            _score, _point = _score.max(2)
            _score += hidden
            point = torch.cat((point, _point.unsqueeze(1)), 1)
            score = _score * (1 - mask) + score * mask
        score += self.transition[TAG2ID['End']]
        best_score, best_tag = torch.max(score, 1)

        # 后向传播
        point = point.tolist()
        # print("masks = ",masks.shape)
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(hiddens.shape[1]):
            i = best_tag[b]
            j = (1-masks[:, b]).sum().int()
            for _point in reversed(point[b][:j]):
                i = _point[i]
                best_path[b].append(i)
            # print("len(best_path[b]) = ",len(best_path[b]))
            best_path[b].pop()
            best_path[b].reverse()



        # print("best_path = ",best_path)
        return best_path
