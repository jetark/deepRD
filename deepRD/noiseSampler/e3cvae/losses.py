import math
import torch
from .axial_covariance import axial_vector_nll, bond_unit_from_edge_vec

def isotropic_vector_nll(y, mu, log_sigma):
    """
    y, mu: [B*2, 3]
    log_sigma: [B*2, 1]
    returns per-node 3D vector NLL: [B*2]
    """
    diff2 = (y - mu).pow(2).sum(dim=-1, keepdim=True)
    inv_var = torch.exp(-2.0 * log_sigma)

    nll = 0.5 * (
        diff2 * inv_var
        + 6.0 * log_sigma
        + 3.0 * math.log(2.0 * math.pi)
    )

    return nll.squeeze(-1)

def standard_gaussian_kl(z_mu, z_logvar, free_bits=0.0):
    kl_per_dim = -0.5 * (1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=-1)

def e3_cvae_axial_loss(outputs, batch, beta=1.0, free_bits=0.0):
    """
    Graph-level E3 CVAE loss using the axial vector Gaussian decoder.

    Each graph has two 3D bead targets. The per-node axial NLL is a full 3D
    vector likelihood (1 parallel + 2 perpendicular dimensions), and the graph
    NLL sums both bead terms, so each graph contributes a 6D observation
    likelihood before averaging across the batch.

    outputs:
        "mu":             [B*2, 3]
        "log_sigma":      [B*2, 2], columns are parallel/perpendicular
        "z_mu":           [B, zdim]
        "z_logvar":       [B, zdim]
    batch:
        "r_next":         [B*2, 3]
        "edge_vec":       [2*B, 3]
        "num_graphs":     int
        optional "bond_unit_node": [B*2, 3]
    """
    mu = outputs["mu"]
    log_sigma = outputs["log_sigma"]
    y = batch["r_next"]
    B = batch["num_graphs"]

    bond_unit = batch.get("bond_unit_node")
    if bond_unit is None:
        bond_unit = bond_unit_from_edge_vec(batch["edge_vec"], B)

    nll_node = axial_vector_nll(
        y,
        mu,
        bond_unit,
        log_sigma[:, 0:1],
        log_sigma[:, 1:2],
    )
    nll_graph = nll_node.reshape(B, 2).sum(dim=-1)  # [B], two 3D beads per graph
    kl_graph = standard_gaussian_kl(outputs["z_mu"], outputs["z_logvar"], free_bits=free_bits)

    nll = nll_graph.mean()
    kl = kl_graph.mean()
    loss = nll + beta * kl
    return loss, nll, kl


def e3_cvae_isotropic_loss(outputs, batch, beta=1.0, free_bits=0.0):
    """
    Graph-level E3 CVAE loss with isotropic Gaussian decoder (one sigma per node).

    outputs:
        "mu":         [B*2, 3]
        "log_sigma":  [B*2, 1]
        "z_mu":       [B, zdim]
        "z_logvar":   [B, zdim]
    batch:
        "r_next":     [B*2, 3]
        "num_graphs": int
    """
    mu = outputs["mu"]
    log_sigma = outputs["log_sigma"]
    y = batch["r_next"]
    B = batch["num_graphs"]

    nll_node = isotropic_vector_nll(y, mu, log_sigma)
    nll_graph = nll_node.reshape(B, 2).sum(dim=-1)
    kl_graph = standard_gaussian_kl(outputs["z_mu"], outputs["z_logvar"], free_bits=free_bits)

    nll = nll_graph.mean()
    kl = kl_graph.mean()
    loss = nll + beta * kl
    return loss, nll, kl
