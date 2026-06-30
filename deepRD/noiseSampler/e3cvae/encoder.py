import torch
import torch.nn as nn
from e3nn import o3
from .messagepassing import E3MessageLayer

def extract_0e(x, irreps):
    """
    x: [N, irreps.dim]
    returns all 0e scalar channels
    """
    irreps = o3.Irreps(irreps)
    parts = []

    for (mul, ir), sl in zip(irreps, irreps.slices()):
        if ir.l == 0 and ir.p == 1:
            parts.append(x[:, sl])

    if not parts:
        raise ValueError("No 0e scalar channels found.")

    return torch.cat(parts, dim=-1)

class E3InvariantEncoder(nn.Module):
    def __init__(self, zdim, hidden_irreps="32x0e + 16x1o + 8x2e", lag2=False):
        super().__init__()

        # Base: 5 vector inputs (v_n, v_nm1, r_n, r_nm1, r_next); lag2 adds v_nm2 and r_nm2
        n_vecs = 7 if lag2 else 5
        self.irreps_in = o3.Irreps(f"{n_vecs}x1o + {n_vecs}x0e")
        self.irreps_hidden = o3.Irreps(hidden_irreps)

        self.layers = nn.ModuleList([
            E3MessageLayer(self.irreps_in, self.irreps_hidden, lmax=2),
            E3MessageLayer(self.irreps_hidden, self.irreps_hidden, lmax=2),
        ])

        scalar_dim = extract_0e(
            torch.zeros(1, self.irreps_hidden.dim),
            self.irreps_hidden,
        ).shape[-1]

        self.readout = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
        )

        self.mu_head = nn.Linear(128, zdim)
        self.logvar_head = nn.Linear(128, zdim)

    def forward(self, h, edge_index, edge_vec, edge_radial, batch_index):
        """
        h: [B*2, irreps_enc_in.dim]
        batch_index: [B*2], tells which graph/sample each node belongs to
        """

        for layer in self.layers:
            h = layer(h, edge_index, edge_vec, edge_radial)

        scalars = extract_0e(h, self.irreps_hidden)

        B = int(batch_index.max().item()) + 1
        pooled = scalars.new_zeros(B, scalars.shape[-1])
        pooled.index_add_(0, batch_index, scalars)

        # For exactly two beads, sum and mean only differ by factor 2.
        pooled = pooled / 2.0

        h_graph = self.readout(pooled)

        z_mu = self.mu_head(h_graph)
        z_logvar = self.logvar_head(h_graph)

        return z_mu, z_logvar
