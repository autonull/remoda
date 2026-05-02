import torch
import torch.nn as nn
from pgra import PGRALayer

class BaselineActorCritic(nn.Module):
    """
    Standard Actor-Critic with an LSTM memory backbone.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init_state(self, batch_size, device):
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (hx, cx)

    def forward(self, x, state):
        features = self.encoder(x)
        hx, cx = self.rnn(features, state)

        logits = self.actor(hx)
        value = self.critic(hx)

        return logits, value, (hx, cx)


class PGRAActorCritic(nn.Module):
    """
    Actor-Critic augmented with a PGRALayer for episodic memory.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # We replace the LSTM with the PGRALayer
        self.pgra = PGRALayer(d_model=hidden_dim, n_heads=4, d_k=hidden_dim//4, d_v=hidden_dim//4)

        # After PGRA, we want to maintain a fast representation
        self.post_pgra = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init_state(self, batch_size, device):
        return self.pgra.init_state(batch_size, device)

    def forward(self, x, state, external_signal=None):
        features = self.encoder(x)

        pgra_out, new_state = self.pgra.forward_step(features, state, external_signal)

        h = self.post_pgra(pgra_out)

        logits = self.actor(h)
        value = self.critic(h)

        return logits, value, new_state
