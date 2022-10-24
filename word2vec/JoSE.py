# 我想尝试使用JoSE训练一个词向量
# 论文如下：[Spherical Text Embedding](https://arxiv.org/abs/1911.01196)

import time
from sys import float_info, stdout

import fire
import torch
from torch.optim import Adam

