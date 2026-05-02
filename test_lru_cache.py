import torch
import time
from kernel import unified_attention_reference

batch = 16
num_heads = 4
seq_len = 512
head_dim = 64
depth_len = 512
num_key_value_heads = 4

q = torch.randn(batch, num_heads, seq_len, head_dim)
seq_k = torch.randn(batch, num_key_value_heads, seq_len, head_dim)
seq_v = torch.randn(batch, num_key_value_heads, seq_len, head_dim)
depth_k = torch.randn(batch, num_key_value_heads, depth_len, head_dim)
depth_v = torch.randn(batch, num_key_value_heads, depth_len, head_dim)

start = time.time()
for _ in range(100):
    out = unified_attention_reference(q, seq_k, seq_v, depth_k, depth_v, causal=True)
print("Time taken:", time.time() - start)
