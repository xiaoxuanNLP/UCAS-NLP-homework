import torch
import torch.nn as nn


class NegativeSampling(nn.Module):
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        k = scores.size()[1] - 1
        return -torch.sum(
            self._log_sigmoid(scores[:, 0])
            + torch.sum(self._log_sigmoid(-scores[:,1:]),dim=1)/ k
        ) / scores.size()[0]
