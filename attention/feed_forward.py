import torch
import torch.nn as nn
import numpy as np

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True, device=device)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        w1 = self.dropout(self.linear1(inputs).relu())
        w2 = self.linear2(w1)
        return w2
