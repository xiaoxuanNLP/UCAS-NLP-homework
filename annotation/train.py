# pytorch code for sequence tagging

# 此版本为简单的NER代码，没有使用CRF和训练好的词向量，仅做参考使用。

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils import build_vocab, build_dict, cal_max_length, Config
from model import LSTM,LSTM_CRF
from torch.optim import Adam, SGD
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class NERdataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag2id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file, encoding="utf-8").readlines()
        label = open(label_file, encoding="utf-8").readlines()
        self.corpus = []
        self.label = []
        self.length = []
        self.mask = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            self.length.append(len(corpus_.split()))
            if (len(self.corpus[-1]) > max_length):
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.length[-1] = max_length
                self.mask.append([0 for _ in range(max_length)])
            else:
                self.mask.append([0 for _ in range(len(self.corpus[-1]))])
                while (len(self.corpus[-1]) < max_length):
                    self.mask[-1].append(1)
                    self.corpus[-1].append(word2id['pad'])
                    self.label[-1].append(tag2id['PAD'])

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()
        self.mask = torch.Tensor(self.mask).long()

    def __getitem__(self, item):
        return self.corpus[item], self.label[item], self.length[item], self.mask[item]

    def __len__(self):
        return len(self.label)


def val(config, model):
    # ignore the pad label
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    testset = NERdataset(config.data_dir, 'test', word2id, tag2id, max_length)
    dataloader = DataLoader(testset, batch_size=config.batch_size)
    preds, labels = [], []
    for index, data in enumerate(dataloader):
        optimizer.zero_grad()
        corpus, label, length,mask = data
        corpus, label, length,mask = corpus.cuda(), label.cuda(), length.cuda(),mask.cuda()
        predict = model.decode(corpus,mask)
        # print("predict = ",len(predict[0]))
        # print("label = ",len(label[0]))
        # predict = torch.argmax(output, dim=-1)
        # loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
        leng = []
        for i in label.cpu():
            tmp = []
            for j in i:
                if j.item() < 7:
                    tmp.append(j.item())
            leng.append(tmp)

        for index, i in enumerate(label.tolist()):
            # print("label = ",label)
            labels.extend(i[:len(leng[index])])
            # print("labels = ",labels)

        for index, i in enumerate(predict):
            # print("predict = ",predict)
            preds.extend(i[:len(leng[index])])



    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)
    model.train()
    return precision, recall, f1


def train(config, model, dataloader, optimizer):
    # ignore the pad label
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    best_f1 = 0.0
    for epoch in range(config.epoch):
        for index, data in enumerate(dataloader):
            optimizer.zero_grad()
            corpus, label, length,mask = data
            corpus, label, length,mask = corpus.cuda(), label.cuda(), length.cuda(),mask.cuda()
            output = model(corpus,mask,label)
            loss = output
            # loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
            loss.backward()
            optimizer.step()
            if (index % 200 == 0):
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (index, loss.item()))

        prec, rec, f1 = val(config, model)
        if (f1 > best_f1):
            torch.save(model, config.save_model)


if __name__ == '__main__':
    config = Config()
    word_dict = build_vocab(config.data_dir)
    word2id, tag2id = build_dict(word_dict)
    max_length = cal_max_length(config.data_dir)
    trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
    dataloader = DataLoader(trainset, batch_size=config.batch_size)
    lstm_crf = LSTM_CRF(config.embedding_dim, config.hidden_dim, config.dropout, word2id, tag2id).cuda()
    optimizer = Adam(lstm_crf.parameters(), config.learning_rate)

    train(config, lstm_crf, dataloader, optimizer)
    # a = torch.tensor([[[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
    # b = torch.tensor([[[3],[3],[3]],[[5],[5],[5]],[[7],[7],[7]]])
    # print(a.shape)
    # print(b.shape)
    # print(a+b)