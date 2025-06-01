import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        std = torch.std(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)

        output = (x - std)/torch.sqrt(var + self.eps)
        return output
