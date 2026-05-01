import torch
import torch.nn.functional as F

import functools

@functools.lru_cache(maxsize=32)
def _get_mask(seq_len_cache, depth_len_cache, causal_cache, device_str):
    device_cache = torch.device(device_str)
    mask = torch.zeros((seq_len_cache, depth_len_cache + seq_len_cache), dtype=torch.bool, device=device_cache)

    if causal_cache:
        seq_mask = torch.tril(torch.ones((seq_len_cache, seq_len_cache), dtype=torch.bool, device=device_cache))
        # Depth part is fully visible
        mask[:, :depth_len_cache] = True
        # Sequence part is causal
        mask[:, depth_len_cache:] = seq_mask
    else:
        # Depth and sequence parts are fully visible
        mask[:, :] = True

    # SDPA expects a boolean mask of shape (batch, num_heads, seq_len, depth_len + seq_len)
    # or just broadcastable to it, so we can reshape to (1, 1, seq_len, depth_len + seq_len)
    return mask.view(1, 1, seq_len_cache, depth_len_cache + seq_len_cache)

def unified_attention_reference(
    q: torch.Tensor,
    seq_k: torch.Tensor,
    seq_v: torch.Tensor,
    depth_k: torch.Tensor = None,
    depth_v: torch.Tensor = None,
    causal: bool = True
) -> torch.Tensor:
    """
    Computes joint attention over sequence KV and cross-layer depth KV.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        seq_k: (batch, num_heads, seq_len, head_dim)
        seq_v: (batch, num_heads, seq_len, head_dim)
        depth_k: (batch, num_heads, depth_len, head_dim) - optional
        depth_v: (batch, num_heads, depth_len, head_dim) - optional
        causal: bool, whether to apply a causal mask to the sequence attention

    Returns:
        output: (batch, num_heads, seq_len, head_dim)
    """
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
        The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
        to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    batch, num_heads, seq_len, head_dim = q.shape
    num_key_value_heads = seq_k.shape[1]
    n_rep = num_heads // num_key_value_heads

    seq_k = repeat_kv(seq_k, n_rep)
    seq_v = repeat_kv(seq_v, n_rep)

    has_depth = depth_k is not None and depth_v is not None

    if not has_depth:
        # Standard attention over sequence only
        return F.scaled_dot_product_attention(
            q, seq_k, seq_v, is_causal=causal
        )

    depth_k = repeat_kv(depth_k, n_rep)
    depth_v = repeat_kv(depth_v, n_rep)

    depth_len = depth_k.shape[2]

    # Concatenate depth and sequence keys/values along the sequence dimension
    # (batch, num_heads, depth_len + seq_len, head_dim)
    k_combined = torch.cat([depth_k, seq_k], dim=2)
    v_combined = torch.cat([depth_v, seq_v], dim=2)

    # We need to construct a custom mask since the depth KV is not causal
    # but the sequence KV is causal.
    # The mask should allow:
    # 1. Any query to attend to any depth KV token (fully visible)
    # 2. Query at pos `i` to attend to seq KV token `j` only if `j <= i` (causal)

    mask = _get_mask(seq_len, depth_len, causal, str(q.device))

    # Run scaled dot product attention
    out = F.scaled_dot_product_attention(
        q, k_combined, v_combined, attn_mask=mask
    )

    return out

from abc import ABC, abstractmethod

class AttentionKernel(ABC):
    @abstractmethod
    def __call__(
        self,
        q: torch.Tensor,
        seq_k: torch.Tensor,
        seq_v: torch.Tensor,
        depth_k: torch.Tensor = None,
        depth_v: torch.Tensor = None,
        causal: bool = True
    ) -> torch.Tensor:
        pass

class PyTorchSDPAKernel(AttentionKernel):
    def __call__(
        self,
        q: torch.Tensor,
        seq_k: torch.Tensor,
        seq_v: torch.Tensor,
        depth_k: torch.Tensor = None,
        depth_v: torch.Tensor = None,
        causal: bool = True
    ) -> torch.Tensor:
        return unified_attention_reference(q, seq_k, seq_v, depth_k, depth_v, causal)

def get_attention_kernel(use_triton: bool = False) -> AttentionKernel:
    """
    Returns the appropriate attention kernel interface.
    """
    if use_triton:
        if not torch.cuda.is_available():
            print("Warning: Triton requested but CUDA is not available. Falling back to PyTorch SDPA reference.")
            return PyTorchSDPAKernel()
        raise NotImplementedError("Triton kernel is not implemented for CPU execution.")
    return PyTorchSDPAKernel()
