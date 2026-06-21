import math
import torch


def bond_unit_from_edge_vec(edge_vec: torch.Tensor, num_graphs: int, eps: float = 1e-12):
    """
    Build node-level bond unit vectors from the dimer edge vectors.

    edge_vec: [2*B, 3], ordered as first B edges 0->1 with dq,
              next B edges 1->0 with -dq.
    returns:  [2*B, 3], ordered like flattened nodes
              [g0 bead0, g0 bead1, g1 bead0, g1 bead1, ...].
    """
    dq = edge_vec[:num_graphs]  # [B, 3]
    e = dq / dq.norm(dim=-1, keepdim=True).clamp_min(eps)
    return e.repeat_interleave(2, dim=0)


def axial_vector_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    bond_unit: torch.Tensor,
    log_sigma_para: torch.Tensor,
    log_sigma_perp: torch.Tensor,
):
    """
    Per-node Gaussian NLL for covariance aligned with the bond axis.

    Covariance:
        Sigma = sigma_para^2 ee^T + sigma_perp^2 (I - ee^T)

    y, mu:          [B*2, 3]
    bond_unit:      [B*2, 3]
    log_sigma_*:    [B*2] or [B*2, 1]
    returns:        [B*2]
    """
    log_sigma_para = log_sigma_para.squeeze(-1)
    log_sigma_perp = log_sigma_perp.squeeze(-1)

    diff = y - mu
    d_para = (diff * bond_unit).sum(dim=-1)       # [B*2]
    diff_para = d_para[:, None] * bond_unit       # [B*2, 3]
    diff_perp = diff - diff_para                  # [B*2, 3]

    para2 = d_para.pow(2)
    perp2 = diff_perp.pow(2).sum(dim=-1)

    inv_para = torch.exp(-2.0 * log_sigma_para)
    inv_perp = torch.exp(-2.0 * log_sigma_perp)

    return 0.5 * (
        para2 * inv_para
        + perp2 * inv_perp
        + 2.0 * log_sigma_para
        + 4.0 * log_sigma_perp
        + 3.0 * math.log(2.0 * math.pi)
    )


def sample_axial_gaussian(
    mu: torch.Tensor,
    bond_unit: torch.Tensor,
    log_sigma_para: torch.Tensor,
    log_sigma_perp: torch.Tensor,
    noise_scale: float = 1.0,
):
    """
    Sample vectors from the axial Gaussian.

    mu:             [B*2, 3]
    bond_unit:      [B*2, 3]
    log_sigma_*:    [B*2] or [B*2, 1]
    returns:        [B*2, 3]
    """
    sigma_para = torch.exp(log_sigma_para).squeeze(-1) * noise_scale
    sigma_perp = torch.exp(log_sigma_perp).squeeze(-1) * noise_scale

    eps_para = torch.randn(mu.shape[0], device=mu.device, dtype=mu.dtype)
    eps = torch.randn_like(mu)
    eps_perp = eps - (eps * bond_unit).sum(dim=-1, keepdim=True) * bond_unit

    noise = (
        sigma_para[:, None] * eps_para[:, None] * bond_unit
        + sigma_perp[:, None] * eps_perp
    )
    return mu + noise
