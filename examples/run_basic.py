from attention import SelfAttention, PositionWiseFeedForward, SublayerConnection

d_model = 5 # let d_model = d_k = d_v; effectively creating a single attention head

input_seq = torch.rand(100, d_model).to(device) # shape: (seq_len, d_model)

self_attn = SelfAttention(d_model, 5, 5).to(device)
ffn = PositionWiseFeedForward(d_model, 2048).to(device)
connection = SublayerConnection(0.1).to(device)

result = connection(input_seq, self_attn)
# result is of dimension (100, 5)
