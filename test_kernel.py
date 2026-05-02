import torch
from kernel import unified_attention_reference

def test_unified_attention_causality():
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 64
    depth_len = 8

    num_key_value_heads = 2
    q = torch.randn(batch, num_heads, seq_len, head_dim, requires_grad=True)
    seq_k = torch.randn(batch, num_key_value_heads, seq_len, head_dim, requires_grad=True)
    seq_v = torch.randn(batch, num_key_value_heads, seq_len, head_dim, requires_grad=True)
    depth_k = torch.randn(batch, num_key_value_heads, depth_len, head_dim, requires_grad=True)
    depth_v = torch.randn(batch, num_key_value_heads, depth_len, head_dim, requires_grad=True)

    out = unified_attention_reference(q, seq_k, seq_v, depth_k, depth_v, causal=True)

    # 1. Output shape should be (batch, num_heads, seq_len, head_dim)
    assert out.shape == (batch, num_heads, seq_len, head_dim), "Output shape mismatch"

    # 2. Check gradients flow to depth KV
    loss = out.sum()
    loss.backward()

    assert depth_k.grad is not None, "Gradient did not flow to depth_k"
    assert depth_v.grad is not None, "Gradient did not flow to depth_v"

    # 3. Check Causality: if we perturb seq_k at position T,
    # it should only affect output at positions >= T

    q2 = torch.randn(batch, num_heads, seq_len, head_dim)
    seq_k2 = torch.randn(batch, num_key_value_heads, seq_len, head_dim)
    seq_v2 = torch.randn(batch, num_key_value_heads, seq_len, head_dim)
    depth_k2 = torch.randn(batch, num_key_value_heads, depth_len, head_dim)
    depth_v2 = torch.randn(batch, num_key_value_heads, depth_len, head_dim)

    out1 = unified_attention_reference(q2, seq_k2, seq_v2, depth_k2, depth_v2, causal=True)

    # Perturb seq_k2 at pos 5
    seq_k2_perturbed = seq_k2.clone()
    seq_k2_perturbed[:, :, 5, :] += 1.0

    out2 = unified_attention_reference(q2, seq_k2_perturbed, seq_v2, depth_k2, depth_v2, causal=True)

    # Positions < 5 should be identical
    assert torch.allclose(out1[:, :, :5, :], out2[:, :, :5, :], atol=1e-6), "Causality violated: past changed!"
    # Positions >= 5 should be different (with high probability)
    assert not torch.allclose(out1[:, :, 5:, :], out2[:, :, 5:, :], atol=1e-6), "Causality violated: future unchanged!"

    # 4. Check Depth Gate
    depth_gate_0 = torch.zeros(1, num_heads, 1, 1)
    depth_gate_1 = torch.ones(1, num_heads, 1, 1)
    depth_gate_2 = torch.full((1, num_heads, 1, 1), 2.0)

    out_g0 = unified_attention_reference(q2, seq_k2, seq_v2, depth_k2, depth_v2, causal=True, depth_gate=depth_gate_0)
    out_g1 = unified_attention_reference(q2, seq_k2, seq_v2, depth_k2, depth_v2, causal=True, depth_gate=depth_gate_1)
    out_g2 = unified_attention_reference(q2, seq_k2, seq_v2, depth_k2, depth_v2, causal=True, depth_gate=depth_gate_2)

    assert not torch.allclose(out_g0, out_g1, atol=1e-6), "Depth gate 0 should be different from gate 1"
    assert not torch.allclose(out_g1, out_g2, atol=1e-6), "Depth gate 1 should be different from gate 2"

    # Grad check for depth gate
    depth_gate_param = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1))
    out_g = unified_attention_reference(q2, seq_k2, seq_v2, depth_k2, depth_v2, causal=True, depth_gate=depth_gate_param)
    out_g.sum().backward()
    assert depth_gate_param.grad is not None, "Gradient did not flow to depth_gate"

    print("All kernel tests passed!")

if __name__ == "__main__":
    test_unified_attention_causality()
