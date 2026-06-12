import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Building blocks for MLPs and output heads for noise samplers.
"""

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.LayerNorm(h)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def reparam(mu, logvar, Tz=1.0):
    """
    Reparameterization for a diagonal Gaussian with parameters (mu, logvar).
    logvar = log(variance), so std = exp(0.5 * logvar).
    """
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(0.5 * logvar) * Tz

def softplus_floor(x, floor):
    # smooth lower bound: >= floor, with gradient
    return floor + F.softplus(x - floor)  # log(1 + exp(x - floor)) + floor

def sigmoid_box(x, lo, hi, temp=1.0):
    return lo + (hi - lo) * torch.sigmoid(temp * x)

class DiagGaussianHead(nn.Module):
    """Outputs (mu, log_sigma) for R^3."""
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even: 2 * D"
        self.D = out_dim//2
        self.mlp = MLP(in_dim, out_dim, hidden=hidden)
    def forward(self, x):
        out = self.mlp(x)
        mu, log_sigma = out[..., :self.D], out[..., self.D:]
        return mu, log_sigma

class DeterministicHead(nn.Module):
    """
    Deterministic output head:
        c -> r_next_pred
    """
    def __init__(self, in_dim, out_dim, hidden=(128, 128)):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.mlp(x)

class FullGaussianHead(nn.Module):
    """
    Outputs mean mu in R^D and Cholesky factor L of a full covariance:
        Sigma = L @ L.T

    We parameterize:
      - unconstrained lower-triangular off-diagonal entries directly
      - diagonal entries through softplus + eps to ensure positivity
    """
    def __init__(self, in_dim, D, hidden=(128, 128), min_diag=1e-4):
        super().__init__()
        self.D = D
        self.min_diag = min_diag
        self.n_tril = D * (D + 1) // 2
        self.mlp = MLP(in_dim, out_dim=D + self.n_tril, hidden=hidden)

        # indices for lower triangle
        tril_idx = torch.tril_indices(row=D, col=D, offset=0)
        self.register_buffer("tril_row", tril_idx[0])
        self.register_buffer("tril_col", tril_idx[1])

    def forward(self, x):
        """
        x: (..., in_dim)

        returns:
            mu: (..., D)
            L:  (..., D, D), lower-triangular with positive diagonal
        """
        out = self.mlp(x)
        mu = out[..., :self.D]
        raw_tril = out[..., self.D:]   # (..., n_tril)

        batch_shape = raw_tril.shape[:-1]
        L = torch.zeros(*batch_shape, self.D, self.D,
                        device=raw_tril.device, dtype=raw_tril.dtype)

        L[..., self.tril_row, self.tril_col] = raw_tril

        # enforce positive diagonal
        diag_idx = torch.arange(self.D, device=raw_tril.device)
        raw_diag = L[..., diag_idx, diag_idx]
        pos_diag = F.softplus(raw_diag) + self.min_diag
        pos_diag = pos_diag.to(dtype=L.dtype)
        L[..., diag_idx, diag_idx] = pos_diag

        return mu, L