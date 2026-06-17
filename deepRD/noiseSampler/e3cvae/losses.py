import math
import torch

def isotropic_vector_nll(y, mu, log_sigma):
    """
    y, mu: [B*2, 3]
    log_sigma: [B*2, 1]
    """
    diff2 = (y - mu).pow(2).sum(dim=-1, keepdim=True)
    inv_var = torch.exp(-2.0 * log_sigma)

    nll = 0.5 * (
        diff2 * inv_var
        + 6.0 * log_sigma
        + 3.0 * math.log(2.0 * math.pi)
    )

    return nll.squeeze(-1)

def standard_gaussian_kl(z_mu, z_logvar):
    return -0.5 * torch.sum(
        1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp(),
        dim=-1,
    )

def axial_vector_nll(y, mu, bond_unit, log_sigma_para, log_sigma_perp):
    """
    y, mu: [B*2, 3]
    bond_unit: [B*2, 3]
        For bead 1 and bead 2, you can use the same bond direction
        or signed directions depending on your convention.
    log_sigma_para, log_sigma_perp: [B*2]
    """
    diff = y - mu

    d_para = (diff * bond_unit).sum(dim=-1)
    diff_para = d_para[:, None] * bond_unit
    diff_perp = diff - diff_para

    para2 = d_para.pow(2)
    perp2 = diff_perp.pow(2).sum(dim=-1)

    inv_para = torch.exp(-2.0 * log_sigma_para)
    inv_perp = torch.exp(-2.0 * log_sigma_perp)

    nll = 0.5 * (
        para2 * inv_para
        + perp2 * inv_perp
        + 2.0 * log_sigma_para
        + 4.0 * log_sigma_perp
        + 3.0 * math.log(2.0 * math.pi)
    )

    return nll