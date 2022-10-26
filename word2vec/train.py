import torch
from torch.optim import Adam
from .data import *
from torch.utils.data import DataLoader
from torch import cuda
from .models import *
import time

VEC_DIM = 100
LR = 0.01
EPOCHS = 10


def train(data_file_name,
          context_size,
          num_noise_words,
          vec_dim,
          save_path,
          generate_plot=True,
          ):
    dataset = load_data(data_file_name)
    dataset = MyDataset(dataset, context_size, num_noise_words)

    train_dataloader_params = {
        "batch_size": 50,
        "shuffle": True,
        "num_workers": 1
    }

    train_loader = DataLoader(dataset, **train_dataloader_params)
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    vocab_size = len(dataset.vocabs)
    doc_size = len(dataset.docs)
    model = DM(VEC_DIM, doc_size,vocab_size)
    model.to(device)

    optimizer = Adam(params=model.parameters(),lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        loss = []

        for step,data in enumerate(train_loader,0):
            model.train()
            optimizer.zero_grad()

            middle,preamble,epilogue,doc_index,noise = \
                data["middle"],data["preamble"],data["epilogue"],data["doc_index"],data["noise"]

            middle, preamble, epilogue, doc_index, noise = \
                middle.to(device),preamble.to(device),epilogue.to(device),doc_index.to(device),noise.to(device)

            x = model(doc_index,) # TODO 这个地方再查一下维度问题和输入格式问题

