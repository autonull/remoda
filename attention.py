import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from kernel import get_attention_kernel
from config import ReMoDAConfig

class ReMoDAAttention(nn.Module):
    def __init__(self, config: ReMoDAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # In RT, KV projection inputs are RMS Normalized
        self.kv_norm = nn.RMSNorm(self.hidden_size) if config.use_rt_kv else nn.Identity()
        self.attn_fn = get_attention_kernel(use_triton=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        output_states: Optional[torch.Tensor] = None
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
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        current_layer_kv = (k, v)

        # Depth KV retrieval (MoDA)
        depth_k, depth_v = None, None
        if self.config.use_moda and layer_idx > 0 and len(kv_cache) > 0:
            # Policy: last-n layers
            slots_to_fetch = min(self.config.depth_slots, layer_idx)
            if self.config.depth_selection_policy == "last-n":
                selected_kvs = kv_cache[-slots_to_fetch:]
            else:
                # Fallback to last-n if policy unknown
                selected_kvs = kv_cache[-slots_to_fetch:]

            # Concatenate the selected historical layers along the sequence (depth) dimension
            depth_k = torch.cat([kv[0] for kv in selected_kvs], dim=2)
            depth_v = torch.cat([kv[1] for kv in selected_kvs], dim=2)

        # Unified Attention Kernel (handles sequence and depth KV)
        attn_output = self.attn_fn(
            q=q,
            seq_k=k,
            seq_v=v,
            depth_k=depth_k,
            depth_v=depth_v,
            causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, current_layer_kv
