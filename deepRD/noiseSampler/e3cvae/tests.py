import torch
from e3nn import o3

from .losses import e3_cvae_axial_loss
from .tools import append_z_to_decoder_features, build_dimer_graph_batch, rotate_batch_vectors


def make_random_dimer_batch(batch_size=4, device="cpu", dtype=torch.float32):
    """
    Build a random graph batch for smoke/equivariance tests.
    """
    q1 = torch.randn(batch_size, 3, device=device, dtype=dtype)
    dq = torch.randn(batch_size, 3, device=device, dtype=dtype)
    q2 = q1 + dq
    fields = {
        "v1": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "v2": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "r1": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "r2": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "v1_prev": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "v2_prev": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "r1_prev": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "r2_prev": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "r1_next": torch.randn(batch_size, 3, device=device, dtype=dtype),
        "r2_next": torch.randn(batch_size, 3, device=device, dtype=dtype),
    }
    return build_dimer_graph_batch(q1=q1, q2=q2, **fields)


def smoke_forward_loss_backward_sample(model, batch=None, beta=1.0):
    """
    One forward pass, axial ELBO, backward pass, and sampling call.
    """
    if batch is None:
        device = next(model.parameters()).device
        batch = make_random_dimer_batch(device=device)

    model.train()
    outputs = model(batch)
    loss, nll, kl = e3_cvae_axial_loss(outputs, batch, beta=beta)
    loss.backward()

    model.eval()
    with torch.no_grad():
        r_sample, mu, log_sigma = model.sample_torch(batch)

    assert outputs["mu"].shape == batch["r_next"].shape
    assert outputs["log_sigma"].shape == (batch["num_graphs"] * 2, 2)
    assert r_sample.shape == batch["r_next"].shape
    assert mu.shape == batch["r_next"].shape
    assert log_sigma.shape == (batch["num_graphs"] * 2, 2)
    return {"loss": loss.item(), "nll": nll.item(), "kl": kl.item()}


@torch.no_grad()
def test_decoder_equivariance(model, batch, atol=1e-5, rtol=1e-5):
    """
    Rotating all vector inputs should rotate decoder mu and leave sigmas invariant.
    """
    R = o3.rand_matrix(device=batch["edge_vec"].device)
    z = torch.randn(
        batch["num_graphs"],
        model.zdim,
        device=batch["edge_vec"].device,
        dtype=batch["edge_vec"].dtype,
    )
    z_node = z.repeat_interleave(2, dim=0)

    batch_rot = rotate_batch_vectors(batch, R)

    h_dec = append_z_to_decoder_features(batch["h_dec_base"], z_node)
    h_dec_rot = append_z_to_decoder_features(batch_rot["h_dec_base"], z_node)

    mu, log_sigma = model.decoder(
        h_dec, batch["edge_index"], batch["edge_vec"], batch["edge_radial"]
    )
    mu_rot, log_sigma_rot = model.decoder(
        h_dec_rot,
        batch_rot["edge_index"],
        batch_rot["edge_vec"],
        batch_rot["edge_radial"],
    )

    expected_mu_rot = mu @ R.T
    torch.testing.assert_close(mu_rot, expected_mu_rot, atol=atol, rtol=rtol)
    torch.testing.assert_close(log_sigma_rot, log_sigma, atol=atol, rtol=rtol)


@torch.no_grad()
def test_encoder_invariance(model, batch, atol=1e-5, rtol=1e-5):
    """
    Rotating all vector inputs should leave invariant latent posterior params fixed.
    """
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

    torch.testing.assert_close(z_mu_rot, z_mu, atol=atol, rtol=rtol)
    torch.testing.assert_close(z_logvar_rot, z_logvar, atol=atol, rtol=rtol)
