import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from config import ReMoDAConfig
from layer import ReMoDALayer
from cache import ReMoDACache

class ReMoDAModel(nn.Module):
    def __init__(self, config: ReMoDAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # RoPE Setup
        if config.use_rope:
            self.head_dim = config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            # Let's use learned absolute position embeddings for simplicity if RoPE is disabled.
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([ReMoDALayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.embed_tokens.weight = self.lm_head.weight

    def _get_cos_sin(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        cos, sin, position_ids = None, None, None
        if self.config.use_rope:
            cos, sin = self._get_cos_sin(seq_len, input_ids.device)
        else:
            positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embeddings(position_ids)

        # Forward pass through layers
        kv_cache = ReMoDACache()
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, i, kv_cache, cos=cos, sin=sin, position_ids=position_ids)

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

        if not self.config.use_rope:
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

        cos, sin, position_ids = None, None, None
        if self.config.use_rope:
            # DT has interleaved tokens, so we need to expand timesteps
            # timesteps is (batch, seq_len)
            # each timestep has 3 tokens.
            # position_ids for RoPE: (batch, 3 * seq_len)
            position_ids = timesteps.repeat_interleave(3, dim=1)
            # We use the max possible position to get cos/sin
            max_pos = position_ids.max().item() + 1
            cos, sin = self.model._get_cos_sin(int(max_pos), states.device)

        # Forward pass through layers (bypassing embed_tokens since we just created hidden_states)
        kv_cache = ReMoDACache()
        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states, i, kv_cache, cos=cos, sin=sin, position_ids=position_ids)

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

        hidden_states = self.model.embed_tokens(input_ids)

        cos, sin, position_ids = None, None, None
        if self.config.use_rope:
            cos, sin = self.model._get_cos_sin(seq_len, input_ids.device)
        else:
            positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.model.position_embeddings(position_ids)

        kv_cache = ReMoDACache()
        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states, i, kv_cache, cos=cos, sin=sin, position_ids=position_ids)

        hidden_states = self.model.norm(hidden_states)

        # We only care about the last token representation for sequence classification
        pooled_logits = hidden_states[:, -1, :]
        logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits
