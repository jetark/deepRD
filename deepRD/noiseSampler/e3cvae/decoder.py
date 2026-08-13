import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import NormActivation
import torch.nn.functional as F
from .messagepassing import E3MessageLayer
from .axial_covariance import bond_unit_from_edge_vec


class E3EquivariantDecoder(nn.Module):
    def __init__(self, zdim, hidden_irreps="32x0e + 16x1o + 8x2e", isotropic=False, lag2=False,
                 n_layers=2, radial_num_basis=16, radial_hidden=64, r_cut=2.0,
                 nonlinear_head=False, n_extra_scalars=0, lmax=2,
                 add_linear_response_mean=False, gate_scale=0.05):
        super().__init__()

        self.zdim = zdim
        self.isotropic = isotropic
        self.n_layers = int(n_layers)
        self.nonlinear_head = bool(nonlinear_head)
        self.add_linear_response_mean = bool(add_linear_response_mean)
        self.gate_scale = float(gate_scale)

        # Base: 4 vector inputs (v_n, v_nm1, r_n, r_nm1); lag2 adds v_nm2 and r_nm2.
        # Scalars: one norm per vector + zdim latent + n_extra_scalars (e.g. dx).
        n_vecs = 6 if lag2 else 4
        self.n_vecs = n_vecs
        self.irreps_in = o3.Irreps(f"{n_vecs}x1o + {n_vecs + zdim + n_extra_scalars}x0e")
        self.irreps_hidden = o3.Irreps(hidden_irreps)

        mlk = dict(lmax=lmax, radial_dim=radial_num_basis, radial_hidden=radial_hidden)
        layers = [E3MessageLayer(self.irreps_in, self.irreps_hidden, **mlk)]
        for _ in range(self.n_layers - 1):
            layers.append(E3MessageLayer(self.irreps_hidden, self.irreps_hidden, **mlk))
        self.layers = nn.ModuleList(layers)

        # Optional nonlinear vector readout: an equivariant NormActivation block on
        # the hidden irreps before the linear projection to 1x1o, so mu is a
        # nonlinear (magnitude-gated) function of the final hidden features rather
        # than a bare linear map. Backward-compatible: off by default.
        if self.nonlinear_head:
            self.head_mix = o3.Linear(self.irreps_hidden, self.irreps_hidden)
            self.head_act = NormActivation(self.irreps_hidden, scalar_nonlinearity=F.silu,
                                           normalize=True, epsilon=1e-8)
        self.vector_head = o3.Linear(self.irreps_hidden, "1x1o")

        # Isotropic: one shared log-sigma per node.
        # Axial: two log-scales (parallel and perpendicular to bond).
        n_sigma = 1 if isotropic else 2
        self.sigma_head = o3.Linear(self.irreps_hidden, f"{n_sigma}x0e")

        # Axial linear-response mean term: mu_lin = sum_k g_k,par(c)*u_k,par + g_k,perp(c)*u_k,perp
        # where u_k ranges over the RAW (un-message-passed) conditioning vectors
        # (v_n, v_nm1, r_n, r_nm1, ...) and g_k,{par,perp} are invariant scalar gates
        # read off the same final hidden state sigma_head uses. This gives the small,
        # velocity-dependent dissipation/friction correction (the physical quantity
        # Fix A patches in at inference time, see tests/e3_precision/fd_sampler.py) a
        # direct, un-gated (no NormActivation) path to mu, mirroring how sigma_head
        # already decomposes the noise covariance along the bond axis (axial_covariance.py)
        # instead of forcing it to emerge from generic equivariant message passing.
        # Backward-compatible: off by default, adds only 2*n_vecs extra output scalars.
        # Gates are bounded to +-gate_scale via tanh: the calibrated physical friction
        # slope is ~0.005-0.007 (fd_gains_*.json), but an unconstrained o3.Linear head
        # was empirically found to reach |gate|~0.5-0.7 on in-distribution data (~100x
        # the physical scale) -- exactly the "instantaneous friction via mean is a
        # positive feedback loop" failure mode already documented for the joint-FD loss
        # (PROJECT_CONTEXT.md Sec 6.1, Fix B), and it reproduced as OOD rollout
        # divergence (13/60 short native rollouts -> NaN) before this bound was added.
        if self.add_linear_response_mean:
            self.gate_head = o3.Linear(self.irreps_hidden, f"{2 * n_vecs}x0e")

    def forward(self, h, edge_index, edge_vec, edge_radial):
        h0 = h  # raw input: leading n_vecs*3 entries are the un-gated conditioning vectors

        for layer in self.layers:
            h = layer(h, edge_index, edge_vec, edge_radial)

        h_head = h
        if self.nonlinear_head:
            h_head = self.head_act(self.head_mix(h))

        mu = self.vector_head(h_head)     # [B*2, 3]
        log_sigma = self.sigma_head(h)    # [B*2, 1] (isotropic) or [B*2, 2] (axial)

        if self.add_linear_response_mean:
            num_graphs = h0.shape[0] // 2
            bond_unit = bond_unit_from_edge_vec(edge_vec, num_graphs)          # [B*2, 3]
            raw_vecs = h0[:, : self.n_vecs * 3].reshape(-1, self.n_vecs, 3)    # [B*2, n_vecs, 3]

            v_par_scalar = (raw_vecs * bond_unit[:, None, :]).sum(dim=-1, keepdim=True)
            v_par = v_par_scalar * bond_unit[:, None, :]                      # [B*2, n_vecs, 3]
            v_perp = raw_vecs - v_par

            gates = torch.tanh(self.gate_head(h)) * self.gate_scale           # [B*2, 2*n_vecs], bounded
            g_par = gates[:, : self.n_vecs, None]
            g_perp = gates[:, self.n_vecs :, None]

            mu_lin = (g_par * v_par + g_perp * v_perp).sum(dim=1)             # [B*2, 3]
            mu = mu + mu_lin

        return mu, log_sigma
