import torch
import torch.nn as nn

class DifferentiableSaliency(nn.Module):
    """
    Determines WHEN the network should trigger a massive one-shot update.
    Uses a soft gate with sparsity regularization.
    """
    def __init__(self, d_model, tau=2.0, lambda_sparsity=1e-3):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.tau = tau  # Controls sharpness (→ hard gate as tau → ∞)
        self.lambda_sparsity = lambda_sparsity

    def compute(self, x_t, external_signal=None):
        if external_signal is not None:
            # External signal overrides learned gating (e.g., in RL)
            # Make sure it has shape (B, 1) if x_t is (B, ...)
            if external_signal.dim() == 1:
                external_signal = external_signal.unsqueeze(-1)
            return external_signal

        logits = self.proj(x_t)
        P_t = torch.sigmoid(self.tau * logits)  # Smooth ∈ [0, 1]
        return P_t

    def sparsity_penalty(self, P_t):
        # Encourages rare, high-magnitude triggers
        return self.lambda_sparsity * (P_t.mean() + self._entropy(P_t))

    def _entropy(self, p):
        # Binary entropy for a batch of probabilities
        p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()


class DeltaRuleUpdate(nn.Module):
    """
    Computes the error-driven residual update for the memory matrix.
    """
    def __init__(self):
        super().__init__()

    def compute_delta(self, P_t, M_prev, K_t, V_t, E_prev, V_sal_prev, S_prev):
        # M_prev: (B, H, d_v, d_k)
        # K_t: (B, H, d_k)
        # V_t: (B, H, d_v)
        # E_prev: (B, H, d_k)
        # V_sal_prev: (B, H, d_v)
        # S_prev: (B, 1) or scalar

        # P_t could be shape (B, 1) or similar. Needs to broadcast appropriately.
        # Ensure P_t shapes match for broadcasting
        # P_t typically comes out of Saliency as (B, 1).

        # Check if we can skip update
        # P_t and V_sal_prev are tensors.
        if (P_t < 1e-4).all() and (V_sal_prev.abs() < 1e-4).all():
            return torch.zeros_like(M_prev), S_prev

        # Batch matrix multiplication: (B, H, d_v, d_k) @ (B, H, d_k, 1) -> (B, H, d_v, 1)
        B, H, d_v, d_k = M_prev.shape

        # reshape for bmm
        M_prev_flat = M_prev.view(B * H, d_v, d_k)
        E_prev_flat = E_prev.view(B * H, d_k, 1)
        K_t_flat = K_t.view(B * H, d_k, 1)

        pred_retro = torch.bmm(M_prev_flat, E_prev_flat).view(B, H, d_v)
        pred_pro = torch.bmm(M_prev_flat, K_t_flat).view(B, H, d_v)

        # Error signals
        err_retro = V_t - pred_retro
        err_pro = V_sal_prev - pred_pro

        # Smoothed saliency trace for prospective conditioning
        # S_t will have the same shape as P_t (e.g., B, 1)
        S_t = 0.9 * S_prev + 0.1 * P_t

        # Reshape P_t and S_t for broadcasting to (B, H, 1, 1)
        if P_t.dim() == 2: # (B, 1)
            P_t_expanded = P_t.view(B, 1, 1, 1)
        else:
            P_t_expanded = P_t.view(-1, 1, 1, 1)

        if S_t.dim() == 2:
            S_t_expanded = S_t.view(B, 1, 1, 1)
        else:
            S_t_expanded = S_t.view(-1, 1, 1, 1)

        # Error-driven residual updates
        # torch.einsum('bhv,bhk->bhvk', err, vec) -> out
        retro_update = torch.einsum('bhv,bhk->bhvk', err_retro, E_prev) * P_t_expanded
        pro_update   = torch.einsum('bhv,bhk->bhvk', err_pro, K_t) * P_t_expanded * S_t_expanded

        return retro_update + pro_update, S_t


class PGRALayer(nn.Module):
    """
    Plateau-Gated Rapid Adaptation Layer.
    Can be dropped into a Transformer/Mamba or an RL Actor-Critic Network.
    """
    def __init__(self, d_model, n_heads, d_k, d_v, eps_decay=0.01, lr_alpha=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.eps_decay = eps_decay
        self.lr_alpha = lr_alpha

        # Projections
        self.W_q = nn.Linear(d_model, n_heads * d_k)
        self.W_k = nn.Linear(d_model, n_heads * d_k)
        self.W_v = nn.Linear(d_model, n_heads * d_v)
        self.W_o = nn.Linear(n_heads * d_v, d_model)

        # Saliency and Update
        self.saliency = DifferentiableSaliency(d_model)
        self.update_rule = DeltaRuleUpdate()

    def forward_step(self, x_t, state, external_signal=None):
        """
        Processes a single timestep.
        x_t: (B, d_model)
        state: tuple of (M_prev, E_prev, V_sal_prev, S_prev)
        external_signal: optional (B, 1) for RL
        """
        B = x_t.shape[0]
        M_prev, E_prev, V_sal_prev, S_prev = state

        # 1. Projections
        Q_t = self.W_q(x_t).view(B, self.n_heads, self.d_k)
        K_t = self.W_k(x_t).view(B, self.n_heads, self.d_k)
        V_t = self.W_v(x_t).view(B, self.n_heads, self.d_v)

        # 2. Trigger Plateau Saliency
        P_t = self.saliency.compute(x_t, external_signal)

        # 3. Compute Memory Update
        Delta_M, S_t = self.update_rule.compute_delta(P_t, M_prev, K_t, V_t, E_prev, V_sal_prev, S_prev)

        # 4. Stable update + normalization
        M_t = (1 - self.eps_decay) * M_prev + self.lr_alpha * Delta_M

        # Frobenius normalization to prevent norm drift
        norm = torch.norm(M_t, dim=(-2, -1), keepdim=True)
        M_t = M_t / (norm + 1e-6)

        # 5. Output Retrieval
        # Detached retrieval for slow-weight training
        M_t_det = M_t.detach()

        # BMM for retrieval
        M_t_det_flat = M_t_det.view(B * self.n_heads, self.d_v, self.d_k)
        Q_t_flat = Q_t.view(B * self.n_heads, self.d_k, 1)
        O_t_flat = torch.bmm(M_t_det_flat, Q_t_flat).view(B, self.n_heads, self.d_v)

        # Residual bypass + Output projection
        out = self.W_o(O_t_flat.flatten(-2)) + self.W_o(V_t.flatten(-2))

        # 6. Trace advancement
        # Reshape P_t for broadcasting to (B, H, d_k / d_v)
        P_t_expanded = P_t.view(B, 1, 1)

        E_t = 0.9 * E_prev + K_t
        V_sal_t = 0.9 * V_sal_prev + P_t_expanded * V_t

        new_state = (M_t, E_t, V_sal_t, S_t)
        return out, new_state

    def init_state(self, batch_size, device):
        """
        Initializes the recurrent state.
        """
        M_0 = torch.zeros(batch_size, self.n_heads, self.d_v, self.d_k, device=device)
        E_0 = torch.zeros(batch_size, self.n_heads, self.d_k, device=device)
        V_sal_0 = torch.zeros(batch_size, self.n_heads, self.d_v, device=device)
        S_0 = torch.zeros(batch_size, 1, device=device)
        return (M_0, E_0, V_sal_0, S_0)
