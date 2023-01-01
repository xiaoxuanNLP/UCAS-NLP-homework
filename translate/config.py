import torch

LR = 3e-4
batch_size = 100
dim = 512
n_head = 8
n_layers = 6
dim_k = 64
dim_v = 64
dropout = 0.1
epochs = 50
DEVICE = "cuda:0"
ModelSaveName = ("transformer.epoch_{:d}.loss_{:f}")
