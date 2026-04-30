import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
from kernel import get_attention_kernel

@dataclass
class ReMoDAConfig:
    vocab_size: int = 50257
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    max_position_embeddings: int = 1024

    # Ablation Toggles
    use_rt_kv: bool = True               # True = ReMoDA/RT (project KV from output), False = Std Transformer (project KV from input)
    use_moda: bool = True                # True = ReMoDA/MoDA (cross-layer depth attention), False = Std Self-Attention
    depth_selection_policy: str = "last-n"  # "last-n" or "last-n+random-1"
    norm_placement: str = "post-norm"    # "pre-norm" or "post-norm"
    depth_slots: int = 1                 # How many historical layers to retrieve

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

class ReMoDAModel(nn.Module):
    def __init__(self, config: ReMoDAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # RoPE would typically go here, omitting for minimal purity or using absolute.
        # Let's use learned absolute position embeddings for simplicity.
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([ReMoDALayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids) + self.position_embeddings(positions)

        # Forward pass through layers
        kv_cache = []
        for i, layer in enumerate(self.layers):
            hidden_states, layer_cache_kv = layer(hidden_states, i, kv_cache)
            kv_cache.append(layer_cache_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return loss, logits

class ReMoDADecisionTransformer(nn.Module):
    """
    A Decision Transformer-like wrapper around ReMoDA.
    Takes states, actions, and returns-to-go, embeds them, and predicts the next action.
    """
    def __init__(self, config: ReMoDAConfig, state_dim: int, action_dim: int):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_size = config.hidden_size

        # Embeddings for continuous/discrete state, action, and return-to-go (RTG)
        self.embed_state = nn.Linear(state_dim, self.hidden_size)
        self.embed_action = nn.Embedding(action_dim, self.hidden_size) if action_dim > 0 else None # Assuming discrete actions for CartPole
        self.embed_rtg = nn.Linear(1, self.hidden_size)

        # We reuse ReMoDAModel structure but bypass embed_tokens
        self.model = ReMoDAModel(config)

        # Action predictor head
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, action_dim)
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Embed inputs
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        rtg_embeddings = self.embed_rtg(returns_to_go)

        # Time embeddings
        time_embeddings = self.model.position_embeddings(timesteps)

        # Add time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        # Interleave tokens: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Stack shape: (batch_size, seq_length, 3, hidden_size)
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        )

        # Reshape to (batch_size, seq_length * 3, hidden_size)
        hidden_states = stacked_inputs.reshape(batch_size, 3 * seq_length, self.hidden_size)

        # Forward pass through layers (bypassing embed_tokens since we just created hidden_states)
        kv_cache = []
        for i, layer in enumerate(self.model.layers):
            hidden_states, layer_cache_kv = layer(hidden_states, i, kv_cache)
            kv_cache.append(layer_cache_kv)

        hidden_states = self.model.norm(hidden_states)

        # Reshape back to (batch_size, seq_length, 3, hidden_size)
        hidden_states = hidden_states.reshape(batch_size, seq_length, 3, self.hidden_size)

        # We predict the action based on the state representation
        # The state representation is the 2nd token in the tuple (RTG, State, Action)
        state_preds = hidden_states[:, :, 1]

        action_preds = self.predict_action(state_preds)

        return action_preds

class ReMoDAForSequenceClassification(nn.Module):
    def __init__(self, config: ReMoDAConfig, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # We use the ReMoDAModel but without the lm_head and with a different classification head
        self.model = ReMoDAModel(config)
        self.score = nn.Linear(config.hidden_size, num_labels, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # Pass through the base model
        # We ignore the lm_head logits returned by ReMoDAModel
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.model.embed_tokens(input_ids) + self.model.position_embeddings(positions)

        kv_cache = []
        for i, layer in enumerate(self.model.layers):
            hidden_states, layer_cache_kv = layer(hidden_states, i, kv_cache)
            kv_cache.append(layer_cache_kv)

        hidden_states = self.model.norm(hidden_states)

        # We only care about the last token representation for sequence classification
        pooled_logits = hidden_states[:, -1, :]
        logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits
