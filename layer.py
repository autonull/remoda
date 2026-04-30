import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from config import ReMoDAConfig
from attention import ReMoDAAttention

class ReMoDAMLP(nn.Module):
    def __init__(self, config: ReMoDAConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class ReMoDALayer(nn.Module):
    def __init__(self, config: ReMoDAConfig):
        super().__init__()
        self.config = config
        self.attn = ReMoDAAttention(config)
        self.mlp = ReMoDAMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        residual = hidden_states

        # Normalization placement
        if self.config.norm_placement == "pre-norm":
            normed_hidden_states = self.input_layernorm(hidden_states)
        else: # post-norm
            normed_hidden_states = hidden_states

        if self.config.use_rt_kv:
            # RT/ReMoDA Architecture:
            # 1. Attention uses current input (Q) and persistent_kv from cache
            # 2. Add residual
            # 3. MLP over (Attention + Residual)
            # 4. Generate NEW persistent KV from MLP output (or block output)

            # For the first pass (calculating attention), we haven't computed the output yet.
            # We use `normed_hidden_states` to compute Q. The current layer's KV won't be computed
            # until the end of the block. But wait, standard attention needs current layer KV!
            # Let's align with the RT paper:
            # In RT, Layer L's KV comes from Layer L's output. But at position t, we need to attend to
            # KV up to t-1. This means within-layer recurrence requires causal unrolling or exact tiling.
            # Since we can't easily causal-unroll an MLP inside SDPA, the pure RT approximation
            # computes KV from the previous step's output (or we project standard causal KV for the sequence,
            # and only the DEPTH KV is persistent).
            # Let's strictly follow the ReMoDA minimal design document:

            # "k_persistent, v_persistent = RMS(K_proj(z)), V_proj(z)  # ← KEY: project from OUTPUT"
            # In training, the sequence is computed in parallel. We can't project from `z` (output)
            # to compute `z` (output).
            # The RT paper clarifies this: "KV_persistent comes from current layer output". For parallel training,
            # this implies a cyclic dependency. To break it, RT either shifts by one token temporally,
            # OR (more simply for training without unrolling) we project KV from the INPUT of the layer,
            # but we STORE the OUTPUT of the layer into the cache for the NEXT layer's depth retrieval.

            # Let's correct this based on parallel training reality:
            # - `seq_kv` (within-layer) MUST be projected from layer input (to allow parallel attention).
            # - The `persistent_kv` saved to cache for OTHER layers to use MUST be projected from the layer output.

            pass # See the unified logic below

        # 1. Calculate Attention
        # `output_states=None` means we compute standard seq KV from `normed_hidden_states`
        attn_outputs, seq_kv = self.attn(
            hidden_states=normed_hidden_states,
            layer_idx=layer_idx,
            kv_cache=kv_cache,
            output_states=None
        )

        # 2. Residual + Post-Norm (or Pre-Norm)
        if self.config.norm_placement == "pre-norm":
            hidden_states = residual + attn_outputs
            residual = hidden_states
            normed_hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_outputs = self.mlp(normed_hidden_states)
            hidden_states = residual + mlp_outputs
        else: # post-norm (MoDA favored)
            hidden_states = residual + attn_outputs
            hidden_states = self.input_layernorm(hidden_states)
            residual = hidden_states
            mlp_outputs = self.mlp(hidden_states)
            hidden_states = residual + mlp_outputs
            hidden_states = self.post_attention_layernorm(hidden_states)

        # 3. Compute Persistent KV for the cache (RT-style)
        if self.config.use_rt_kv:
            # We explicitly compute a persistent KV from the layer's OUTPUT, not its input.
            # This is what gets added to the `kv_cache` for deeper layers to retrieve.
            # We can re-use the attn module to just project k and v.
            with torch.no_grad() if not hidden_states.requires_grad else torch.enable_grad():
                # We do this to ensure we are projecting from the final normalized output.
                out_norm = self.attn.kv_norm(hidden_states)
                k_persistent = self.attn.k_proj(out_norm)
                v_persistent = self.attn.v_proj(out_norm)
                batch_size, seq_len, _ = hidden_states.size()
                k_persistent = k_persistent.view(batch_size, seq_len, self.attn.num_heads, self.attn.head_dim).transpose(1, 2)
                v_persistent = v_persistent.view(batch_size, seq_len, self.attn.num_heads, self.attn.head_dim).transpose(1, 2)
                cache_kv = (k_persistent, v_persistent)
        else:
            # Standard transformer: the KV cache just stores the input-projected KV
            cache_kv = seq_kv

        return hidden_states, cache_kv
