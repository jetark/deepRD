import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import NormActivation


def radial_embedding(edge_vec, num_basis=16, r_cut=5.0):
    "Embedding for the edge vector."
    r = edge_vec.norm(dim=-1, keepdim=True)
    centers = torch.linspace(0.0, r_cut, num_basis, device=edge_vec.device)
    widths = (r_cut / num_basis)
    return torch.exp(-((r - centers) ** 2) / (widths ** 2))

class RadialMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class E3MessageLayer(nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        lmax=2,
        radial_dim=16,
        radial_hidden=64,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
        )

        self.radial = RadialMLP(
            in_dim=radial_dim,
            hidden_dim=radial_hidden,
            out_dim=self.tp.weight_numel,
        )

        self.self_connection = o3.Linear(self.irreps_in, self.irreps_out)

        self.activation = NormActivation(
            self.irreps_out,
            scalar_nonlinearity=F.silu,
            normalize=True,
            epsilon=1e-8,
        )

    def forward(self, h, edge_index, edge_vec, edge_radial):
        """
        h:           [num_nodes_total, irreps_in.dim]
        edge_index:  [2, num_edges_total], source -> target
        edge_vec:    [num_edges_total, 3]
        edge_radial: [num_edges_total, radial_dim]
        """

        src, dst = edge_index

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )

        weights = self.radial(edge_radial)

        messages = self.tp(h[src], sh, weights)

        out = h.new_zeros(h.shape[0], self.irreps_out.dim)
        out.index_add_(0, dst, messages)

        out = out + self.self_connection(h)
        out = self.activation(out)

        return out