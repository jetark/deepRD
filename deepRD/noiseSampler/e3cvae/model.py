import torch
import torch.nn as nn
from .encoder import E3InvariantEncoder
from .decoder import E3EquivariantDecoder
from .axial_covariance import bond_unit_from_edge_vec, sample_axial_gaussian
from .tools import append_z_to_decoder_features

class E3DimerCVAE(nn.Module):
    def __init__(self, zdim, hidden_irreps="32x0e + 16x1o + 8x2e", isotropic=False, lag2=False,
                 n_dec_layers=2, dec_nonlinear_head=False, n_enc_layers=2,
                 add_dx_scalar=False, radial_num_basis=16, r_cut=2.0, lmax=2,
                 add_linear_response_mean=False, gate_scale=0.05):
        super().__init__()
        self.zdim = zdim
        self.hidden_irreps = hidden_irreps
        self.isotropic = isotropic
        self.lag2 = lag2
        self.n_dec_layers = int(n_dec_layers)
        self.dec_nonlinear_head = bool(dec_nonlinear_head)
        self.add_linear_response_mean = bool(add_linear_response_mean)
        # featurisation (must be mirrored by build_dimer_graph_batch at train/eval/rollout)
        self.add_dx_scalar = bool(add_dx_scalar)
        self.radial_num_basis = int(radial_num_basis)
        self.r_cut = float(r_cut)
        n_extra = 1 if self.add_dx_scalar else 0
        self.lmax = int(lmax)
        self.encoder = E3InvariantEncoder(zdim=zdim, hidden_irreps=hidden_irreps, lag2=lag2,
                                          n_layers=n_enc_layers, radial_num_basis=radial_num_basis,
                                          n_extra_scalars=n_extra, lmax=lmax)
        self.decoder = E3EquivariantDecoder(zdim=zdim, hidden_irreps=hidden_irreps,
                                            isotropic=isotropic, lag2=lag2,
                                            n_layers=n_dec_layers,
                                            nonlinear_head=dec_nonlinear_head,
                                            radial_num_basis=radial_num_basis, r_cut=r_cut,
                                            n_extra_scalars=n_extra, lmax=lmax,
                                            add_linear_response_mean=add_linear_response_mean,
                                            gate_scale=gate_scale)

    def graph_featurisation(self):
        """Kwargs for build_dimer_graph_batch so eval/rollout match training."""
        return dict(add_dx_scalar=self.add_dx_scalar,
                    radial_num_basis=self.radial_num_basis, r_cut=self.r_cut)

    def reparameterize(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, batch):
        """
        batch should contain:
            encoder node features h_enc
            decoder node features h_dec_base
            edge_index
            edge_vec
            edge_radial
            batch_index
            target r_next, shape [B*2, 3]
        """

        z_mu, z_logvar = self.encoder(
            h=batch["h_enc"],
            edge_index=batch["edge_index"],
            edge_vec=batch["edge_vec"],
            edge_radial=batch["edge_radial"],
            batch_index=batch["batch_index"],
        )

        z = self.reparameterize(z_mu, z_logvar)

        z_node = z.repeat_interleave(2, dim=0)

        h_dec = append_z_to_decoder_features(
            batch["h_dec_base"],
            z_node,
        )

        mu, log_sigma = self.decoder(
            h=h_dec,
            edge_index=batch["edge_index"],
            edge_vec=batch["edge_vec"],
            edge_radial=batch["edge_radial"],
        )

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "log_sigma_para": log_sigma[:, 0:1],
            "log_sigma_perp": log_sigma[:, 1:2],
            "z_mu": z_mu,
            "z_logvar": z_logvar,
        }

    @torch.no_grad()
    def sample_torch(self, batch, Tr=1.0, Tz=1.0):
        B = batch["num_graphs"]
        z = torch.randn(
            B,
            self.zdim,
            device=batch["edge_vec"].device,
            dtype=batch["edge_vec"].dtype,
        ) * Tz
        z_node = z.repeat_interleave(2, dim=0)

        h_dec = append_z_to_decoder_features(
            batch["h_dec_base"],
            z_node,
        )

        mu, log_sigma = self.decoder(
            h=h_dec,
            edge_index=batch["edge_index"],
            edge_vec=batch["edge_vec"],
            edge_radial=batch["edge_radial"],
        )

        if self.isotropic:
            eps = torch.randn_like(mu)
            r_next = mu + torch.exp(log_sigma) * Tr * eps
        else:
            bond_unit = batch.get("bond_unit_node")
            if bond_unit is None:
                bond_unit = bond_unit_from_edge_vec(batch["edge_vec"], B)
            r_next = sample_axial_gaussian(
                mu,
                bond_unit,
                log_sigma[:, 0:1],
                log_sigma[:, 1:2],
                noise_scale=Tr,
            )

        return r_next, mu, log_sigma
