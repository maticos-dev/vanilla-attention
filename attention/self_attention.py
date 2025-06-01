import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)
        self.value_proj = nn.Linear(d_model, d_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query_proj(x) # seq_len * d_k
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = self.softmax(scores)
        output = torch.matmul(attn, V)
        return output
