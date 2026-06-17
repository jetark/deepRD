import torch
from e3nn import o3


@torch.no_grad()
def test_decoder_equivariance(model, batch):
    R = o3.rand_matrix(device=batch["edge_vec"].device)

    z = torch.randn(batch["num_graphs"], model.zdim, device=batch["edge_vec"].device)

    batch_rot = rotate_batch_vectors(batch, R)

    z_node = z.repeat_interleave(2, dim=0)

    h_dec = append_z_to_decoder_features(batch["h_dec_base"], z_node)
    h_dec_rot = append_z_to_decoder_features(batch_rot["h_dec_base"], z_node)

    mu, sig = model.decoder(
        h_dec, batch["edge_index"], batch["edge_vec"], batch["edge_radial"]
    )

    mu_rot, sig_rot = model.decoder(
        h_dec_rot, batch_rot["edge_index"], batch_rot["edge_vec"], batch_rot["edge_radial"]
    )

    expected_mu_rot = mu @ R.T

    rel_err = (mu_rot - expected_mu_rot).norm() / (expected_mu_rot.norm() + 1e-8)
    sig_err = (sig_rot - sig).abs().max()

    print("decoder vector equivariance error:", rel_err.item())
    print("decoder scalar invariance error:", sig_err.item())


@torch.no_grad()
def test_encoder_invariance(model, batch):
    R = o3.rand_matrix(device=batch["edge_vec"].device)
    batch_rot = rotate_batch_vectors(batch, R)

    z_mu, z_logvar = model.encoder(
        batch["h_enc"],
        batch["edge_index"],
        batch["edge_vec"],
        batch["edge_radial"],
        batch["batch_index"],
    )

    z_mu_rot, z_logvar_rot = model.encoder(
        batch_rot["h_enc"],
        batch_rot["edge_index"],
        batch_rot["edge_vec"],
        batch_rot["edge_radial"],
        batch_rot["batch_index"],
    )

    print("z_mu invariance error:", (z_mu - z_mu_rot).abs().max().item())
    print("z_logvar invariance error:", (z_logvar - z_logvar_rot).abs().max().item())