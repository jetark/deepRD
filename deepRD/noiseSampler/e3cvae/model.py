import torch
import torch.nn as nn
from .encoder import E3InvariantEncoder
from .decoder import E3EquivariantDecoder

class E3DimerCVAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.zdim = zdim
        self.encoder = E3InvariantEncoder(zdim=zdim)
        self.decoder = E3EquivariantDecoder(zdim=zdim)

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
            "z_mu": z_mu,
            "z_logvar": z_logvar,
        }

    @torch.no_grad()
    def sample_torch(self, batch):
        B = batch["num_graphs"]
        z = torch.randn(B, self.zdim, device=batch["edge_vec"].device)
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

        eps = torch.randn_like(mu)
        r_next = mu + torch.exp(log_sigma) * eps

        return r_next, mu, log_sigma