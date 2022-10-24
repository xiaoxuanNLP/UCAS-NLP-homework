import multiprocessing
import os
import re
import signal
from os.path import join
from torch.utils.data import Dataset

import numpy as np
import torch
from numpy.random import choice

DATA_DIR = "./data"


def load_data(file_name):
    file_path = join(DATA_DIR, file_name)
    doc = []
    with open(file_path, "r") as f:
        content = f.read()
        doc = content.split("\n")
    return doc


class MyDataset(Dataset):
    # 这个地方需要把context_size传进来，不然没办法计算len TODO
    def __init__(self, docs,context_size,num_noise_words):
        self.docs = docs
        self.vocabs = self.get_vocab()
        self.word2index,self.index2word = self.get_word2index()
        self.context_size = context_size
        self.num_noise_words = num_noise_words
        self.get_noise_distribution = self.init_noise_distribution()
        self.length = 0
        self.data = []


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 这个的地方如何分batch是一个问题，需要截断或者追加
        return None

    def get_vocab(self):
        vocabs = {}
        for doc in self.docs:
            words = doc.split(" ")
            self.length += len(words) - self.context_size*2 + 1
            for word in words:
                try:
                    vocabs[word] += 1
                except KeyError:
                    vocabs[word] = 1

        return vocabs

    def get_word2index(self):
        word2index = {}
        index2word = {}
        index = 1
        for key in self.vocabs.keys():
            word2index[self.vocabs[key]] = index
            index2word[index] = self.vocabs[key]
            index += 1

        return word2index,index2word

    def init_noise_distribution(self):
        probs = np.zeros(len(self.vocabs)-1)
        for word in self.vocabs.keys():
            probs[self.word2index[word]-1] = self.vocabs[word]

        probs = np.power(probs,0.75)
        probs /= np.sum(probs)

        return lambda :choice(
            probs.shape[0],self.num_noise_words,p=probs
        ).tolist()

    # 这里我令doc的id也是从1开始
    def generate_data(self):
        doc_index = 0
        for doc in self.docs:
            doc_index += 1
            words = doc.split(" ")
            for index in range(len(words)-self.context_size):
                preamble = []
                epilogue = []
                for i in range(self.context_size):
                    preamble = words[index + i]
                    epilogue = words[index + self.context_size + i]

                # 上文、下文、文档、噪音
                self.data.append([preamble,epilogue,doc_index,()])


if __name__ == "__main__":
    a = {1:2,2:4}
    print()
    # for word in a.keys():
        # print("word = ",word)
        # print("freq = ",freq)

