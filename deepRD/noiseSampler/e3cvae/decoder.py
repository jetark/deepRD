import torch
import torch.nn as nn
from e3nn import o3
from .messagepassing import E3MessageLayer


class E3EquivariantDecoder(nn.Module):
    def __init__(self, zdim, hidden_irreps="32x0e + 16x1o + 8x2e", isotropic=False, lag2=False):
        super().__init__()

        self.zdim = zdim
        self.isotropic = isotropic

        # Base: 4 vector inputs (v_n, v_nm1, r_n, r_nm1); lag2 adds v_nm2 and r_nm2
        n_vecs = 6 if lag2 else 4
        self.irreps_in = o3.Irreps(f"{n_vecs}x1o + {n_vecs + zdim}x0e")
        self.irreps_hidden = o3.Irreps(hidden_irreps)

        self.layers = nn.ModuleList([
            E3MessageLayer(self.irreps_in, self.irreps_hidden, lmax=2),
            E3MessageLayer(self.irreps_hidden, self.irreps_hidden, lmax=2),
        ])

        self.vector_head = o3.Linear(self.irreps_hidden, "1x1o")

        # Isotropic: one shared log-sigma per node.
        # Axial: two log-scales (parallel and perpendicular to bond).
        n_sigma = 1 if isotropic else 2
        self.sigma_head = o3.Linear(self.irreps_hidden, f"{n_sigma}x0e")

    def forward(self, h, edge_index, edge_vec, edge_radial):
        for layer in self.layers:
            h = layer(h, edge_index, edge_vec, edge_radial)

        mu = self.vector_head(h)          # [B*2, 3]
        log_sigma = self.sigma_head(h)    # [B*2, 1] (isotropic) or [B*2, 2] (axial)

        return mu, log_sigma
