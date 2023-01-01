import torch
import torch.nn as nn
from torch.autograd import Variable

import sacrebleu
from .config import *
from tqdm import tqdm
from .data_loader import *
from .tokenizer import *
from torch import cuda
from transformers import AdamW
from .model import *
from .loss import LabelSmoothing
from .generate import greedy_search_batch
from .utils import save_training_state
import copy

MAX_LEN = 64

def make_model(src_vocab,tgt_vocab,N=6,dim=512,dim_ff=2048,mulithead=8,dropout=0.1):
    attn = MultiHeadedAttention(mulithead, dim).to(DEVICE)
    ff = FeedForward(dim,dim_ff,dropout).to(DEVICE)
    position = PositionEmbedding(dim,dropout).to(DEVICE)
    model = Transformer(
        Encoder(EncoderLayer(dim,copy.deepcopy(attn),copy.deepcopy(ff),dropout).to(DEVICE),N).to(DEVICE),
        Decoder(DecoderLayer(dim,copy.deepcopy(attn),copy.deepcopy(attn),copy.deepcopy(ff),dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embedding(src_vocab,dim).to(DEVICE), c(position)),
        nn.Sequential(Embedding(tgt_vocab,dim).to(DEVICE), c(position)),
        Generator(dim, tgt_vocab)
    )

    return model

def train(data_path):
    tokenizer_en = Tokenizer_en(data_path+"train.en")
    tokenizer_zh = Tokenizer_zh(data_path+"train.zh")

    model = make_model(tokenizer_en.vocab_size(),tokenizer_zh.vocab_size()).to(DEVICE)

    train_en,train_zh = get_from_file(data_path+"train")
    test_en,test_zh = get_from_file(data_path+"test")
    valid_en,valid_zh = get_from_file(data_path+"valid")

    train_set = MyDataset(train_en,train_zh,tokenizer_en,tokenizer_zh,MAX_LEN,padding=True)
    test_set = MyDataset(test_en,test_zh,tokenizer_en,tokenizer_zh,MAX_LEN,padding=True)
    valid_set = MyDataset(valid_en,valid_zh,tokenizer_en,tokenizer_zh,MAX_LEN,padding=True)

    dataloader_params = {
        "batch_size":batch_size,
        "shuffle":True,
        "num_workers":4
    }

    train_loader = DataLoader(train_set,**dataloader_params)
    test_loader = DataLoader(test_set,**dataloader_params)
    valid_loader = DataLoader(valid_set,**dataloader_params)

    device = "cuda:0" if cuda.is_available() else 'cpu'

    train_param = {
        "lr":LR,
    }
    optimizer = AdamW(**train_param)
    label_smoothing_loss = LabelSmoothing(tokenizer_zh.vocab_size(),pad_tag=0,smoothing=0.1)
    prev_model_file_path = None

    for epoch in range(epochs):
        loss_mean = []
        for step,data in enumerate(train_loader,0):
            model.train()
            optimizer.zero_grad()
            data_en_ids,data_en_mask,data_zh_ids,data_zh_mask = data["data_en_ids"],data["data_en_mask"],data["data_zh_ids"],data["data_zh_mask"]
            data_en_ids = data_en_ids.to(DEVICE)
            data_en_mask = data_en_mask.to(DEVICE)
            data_zh_ids = data_zh_ids.to(DEVICE)
            data_zh_mask = data_zh_mask.to(DEVICE)

            out = model(data_en_ids,data_zh_ids,data_en_mask,data_zh_mask)
            loss = label_smoothing_loss(out,data_zh_ids)
            loss_num = loss.item()

            loss.backward()
            optimizer.step()
            loss_mean.append(loss.item())

            if step % 100 == 0: # 每100步打印一下
                print("epoch_{},step:{},loss:{}".format(epoch,step,loss_num))

        bleu = eval(valid_loader,model,tokenizer_zh)
        print("epoch_{},bleu:{}".format(epoch,bleu))
        state = {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "best_bleu":bleu,
            "optimizer_state_dict":optimizer
        }

        prev_model_file_path = save_training_state(
            epoch,
            sum(loss_mean)/len(loss_mean),
            state,
            False,
            True,
            prev_model_file_path
        )

        loss_mean = []

    bleu = eval(test_loader, model, tokenizer_zh)
    print("test bleu:{}".format(bleu))


def eval(data_loader,model,tokenizer_zh):
    trg = []
    res = []
    with torch.no_grad():
        for step, data in enumerate(data_loader, 0):
            data_en_ids, data_en_mask,tgt= data["data_en_ids"], data["data_en_mask"], data['tgt']
            data_en_ids = data_en_ids.to(DEVICE)
            data_en_mask = data_en_mask.to(DEVICE)
            result = greedy_search_batch(model,data_en_ids,data_en_mask)

            decode_result = [h[0] for h in result]
            translation = [tokenizer_zh.decode(_s) for _s in decode_result]
            trg.extend(tgt)
            res.extend(translation)

    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


if __name__ == "__main__":
    train("./data")