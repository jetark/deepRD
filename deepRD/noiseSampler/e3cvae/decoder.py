import torch
import torch.nn as nn
from e3nn import o3
from .messagepassing import E3MessageLayer


class E3EquivariantDecoder(nn.Module):
    def __init__(self, zdim, hidden_irreps="32x0e + 16x1o + 8x2e"):
        super().__init__()

        self.zdim = zdim

        # 4 vector inputs: v_n, v_nm1, r_n, r_nm1
        # plus scalar features and z
        self.irreps_in = o3.Irreps(f"4x1o + {4 + zdim}x0e")
        self.irreps_hidden = o3.Irreps(hidden_irreps)

        self.layers = nn.ModuleList([
            E3MessageLayer(self.irreps_in, self.irreps_hidden, lmax=2),
            E3MessageLayer(self.irreps_hidden, self.irreps_hidden, lmax=2),
        ])

        self.vector_head = o3.Linear(self.irreps_hidden, "1x1o")

        # Two invariant log scales per node:
        #   column 0: parallel to the bond axis
        #   column 1: perpendicular to the bond axis
        self.sigma_head = o3.Linear(self.irreps_hidden, "2x0e")

    def forward(self, h, edge_index, edge_vec, edge_radial):
        for layer in self.layers:
            h = layer(h, edge_index, edge_vec, edge_radial)

        mu = self.vector_head(h)          # [B*2, 3]
        log_sigma = self.sigma_head(h)    # [B*2, 2]

        return mu, log_sigma
