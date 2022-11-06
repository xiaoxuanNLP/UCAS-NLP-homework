import csv
import re
from os.path import join
import json
import pandas as pd

import torch

from data import load_data,MyDataset
from models import DM
from utils import DATA_DIR,MODELS_DIR

def load_model(model_name,vec_dim,num_docs,num_words):
    model_path = join(MODELS_DIR,model_name)

    try:
        checkpoint = torch.load(model_path)
    except AssertionError: # 显存不够往内存里放用CPU
        checkpoint = torch.load(
            model_path,
            map_location=lambda storage, location: storage
        )
    model = DM(vec_dim,num_docs,num_words)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def write_to_file(model_name,vec_dim,data_path,save_name):
    result = []

    dataset = load_data(data_path)
    dataset = MyDataset(dataset, context_size = 2, num_noise_words = 2)

    print("len(dataset.docs),len(dataset.vocabs) = ",len(dataset.docs),len(dataset.vocabs))
    model = load_model(model_name,vec_dim,len(dataset.docs),len(dataset.vocabs))

    for word in dataset.vocabs.keys():
        vec = json.dumps(model.get_word_vector(dataset.word2index[word]))
        result.append([word,vec])

    df = pd.DataFrame(data = result,
                      columns=['word','vec'])

    df.to_csv(save_name)


if __name__ == "__main__":
    write_to_file("zh.sum_contextsize.2_numnoisewords.2_vecdim.100_batchsize.16_lr.0.010000_epoch.18_loss.0.148108.pth",
               100,"zh.txt","zh.vec.csv")

