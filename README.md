# Vanilla Attention (Vaswani et al., 2017)

This repository provides a minimal, **from-scratch PyTorch implementation** of the core building blocks of the **Attention mechanism**, as described in the groundbreaking paper:

> [Attention is All You Need](https://arxiv.org/abs/1706.03762)  
> Ashish Vaswani, et al. (2017)

It is designed to be **educational** and easy to understand, without relying on any high-level abstractions or prebuilt transformer libraries.

---

## ✨ Features

- ✅ Single-head Scaled Dot-Product Attention  
- ✅ Position-wise Feedforward Layer  
- ✅ Residual Connection with Layer Normalization  
- ✅ Fully Modular and Testable Code  
- ✅ Pytest-compatible unit tests

---## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/vanilla-attention.git
cd vanilla-attention
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Build Package
```bash
pip install .
```


### 4. Run Tests (Unittest Style)
```bash
pytest tests/
```

---

## Example Usage
```bash
import torch
from attention import SelfAttention, PositionWiseFeedForward, SublayerConnection

x = torch.rand(100, 5)  # (seq_len, d_model)

attn = SelfAttention(d_model=5, d_k=5, d_v=5)
ffn = PositionWiseFeedForward(d_model=5, d_ff=2048)
residual = SublayerConnection(dropout=0.1)

out = residual(x, attn)
print(out.shape)  # torch.Size([100, 5])
```
