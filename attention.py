import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from kernel import get_attention_kernel, apply_rotary_pos_emb
from config import ReMoDAConfig
from depth_policy import get_depth_policy
from cache import ReMoDACache

class ReMoDAAttention(nn.Module):
    def __init__(self, config: ReMoDAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # In RT, KV projection inputs are RMS Normalized
        self.kv_norm = nn.RMSNorm(self.hidden_size) if config.use_rt_kv else nn.Identity()
        self.attn_fn = get_attention_kernel(use_triton=False)
        self.depth_policy = get_depth_policy(config.depth_selection_policy)

        if config.use_depth_gate:
            self.depth_gate = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        else:
            self.depth_gate = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kv_cache: ReMoDACache,
        output_states: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, seq_len, _ = hidden_states.size()

        # Query is always projected from the block input
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Determine what to project K and V from
        # Standard: project from `hidden_states` (input to the block)
        # RT/ReMoDA: project from `output_states` (output of the MLP residual block)
        kv_input = output_states if (self.config.use_rt_kv and output_states is not None) else hidden_states
        kv_input = self.kv_norm(kv_input)

        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if self.config.use_rope and cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        current_layer_kv = (k, v)

        # Depth KV retrieval (MoDA)
        depth_k, depth_v = None, None
        if self.config.use_moda:
            depth_k, depth_v = self.depth_policy.select_kvs(kv_cache, layer_idx, self.config.depth_slots)

        # Unified Attention Kernel (handles sequence and depth KV)
        attn_output = self.attn_fn(
            q=q,
            seq_k=k,
            seq_v=v,
            depth_k=depth_k,
            depth_v=depth_v,
            causal=True,
            depth_gate=self.depth_gate
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, current_layer_kv
