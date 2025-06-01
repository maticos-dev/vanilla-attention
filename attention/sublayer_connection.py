import torch
import torch.nn as nn
import numpy as np

class SublayerConnection(nn.Module):

    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
