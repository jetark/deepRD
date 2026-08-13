import math
import torch
from .axial_covariance import axial_vector_nll, bond_unit_from_edge_vec


def _batch_cov(a, b):
    """Batch covariance of two [B] tensors (scalar)."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean()


def _batch_cov_vec(a, b):
    """Batch covariance of two [B, D] tensors, summed over D (isotropic)."""
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    return (a * b).sum(dim=-1).mean()


def fluctuation_dissipation_penalty(outputs, batch):
    """
    Match the velocity-noise cross-covariance (the memory-friction term of the
    generalized Langevin equation) between the decoded conditional-mean noise mu
    and the target noise r_next, over the current batch.

    The plain axial NLL is nearly blind to this small correlation (benchmark
    corr(dvx_n, dr_par) ~ -0.06), yet the stationary velocity variance of the
    reduced dynamics is very sensitive to it: under-learning it makes rollouts
    run hot. This penalty adds an explicit, physically-motivated constraint.

    Returns a scalar penalty = sum over {relative-parallel, relative-perp,
    COM-isotropic} channels of (cov_model - cov_target)^2.

    Requires batch["h_dec_base"] (v_n is its first packed vector) and
    batch["bond_unit_node"]; both are produced by build_dimer_graph_batch.
    """
    B = batch["num_graphs"]
    mu = outputs["mu"].reshape(B, 2, 3)
    y = batch["r_next"].reshape(B, 2, 3)
    v_n = batch["h_dec_base"][:, 0:3].reshape(B, 2, 3)      # first packed decoder vector
    e = batch["bond_unit_node"].reshape(B, 2, 3)[:, 0, :]   # per-graph bond unit

    dv = v_n[:, 0] - v_n[:, 1]
    dmu = mu[:, 0] - mu[:, 1]
    dy = y[:, 0] - y[:, 1]
    sv = v_n[:, 0] + v_n[:, 1]
    smu = mu[:, 0] + mu[:, 1]
    sy = y[:, 0] + y[:, 1]

    dvx = (dv * e).sum(-1)
    dmu_par = (dmu * e).sum(-1)
    dy_par = (dy * e).sum(-1)
    dv_perp = dv - dvx[:, None] * e
    dmu_perp = dmu - dmu_par[:, None] * e
    dy_perp = dy - dy_par[:, None] * e

    c_par = (_batch_cov(dvx, dmu_par) - _batch_cov(dvx, dy_par)) ** 2
    c_perp = (_batch_cov_vec(dv_perp, dmu_perp) - _batch_cov_vec(dv_perp, dy_perp)) ** 2
    c_com = (_batch_cov_vec(sv, smu) - _batch_cov_vec(sv, sy)) ** 2
    return c_par + c_perp + c_com


def _centered_sq(a):
    """Total variance of a [n, D] tensor: sum over D of per-dim variance (mean-0)."""
    a = a - a.mean(dim=0, keepdim=True)
    return (a * a).sum(dim=-1).mean()


def joint_fd_penalty(
    outputs,
    batch,
    nbins=8,
    dx_range=(0.0, 3.0),
    w_par=3.0,
    w_perp=1.0,
    w_com=1.0,
    w_var=1.0,
    w_cov=1.0,
    min_bin=32,
    eps=1e-8,
):
    """
    Binned (in bond length dx) fluctuation-dissipation moment-matching loss.

    For each dx-bin and each channel {axial-parallel, relative-perp, COM}, match
    the model to the batch's TRUE r_next on BOTH:
      * fluctuation:  Var(channel) — the model value combines the batch spread of
        the decoded mean mu with the ANALYTIC sampled variance sum(sigma^2) from
        the axial decoder (par/perp), so the sigma head is pushed to carry the
        residual kick magnitude the mean cannot.
      * dissipation:  Cov(dv_channel, channel-mean) — the velocity-noise coupling
        (memory friction), matched via the decoded mean as in
        fluctuation_dissipation_penalty.

    Both terms use relative-squared error (normalized by the detached target
    magnitude) so fluctuation (~1e-4 scale) and dissipation (~1e-4 cov) terms are
    dimensionless and comparably weighted. Per-channel weights up-weight the axial
    (parallel) channel, the one that resisted the warm-start attempt.

    Binning in dx makes the learned friction/kick STATE-dependent (not a global
    mean), matching the slow-bond-mode structure.
    """
    B = batch["num_graphs"]
    mu = outputs["mu"].reshape(B, 2, 3)
    y = batch["r_next"].reshape(B, 2, 3)
    log_sig_par = outputs["log_sigma_para"].reshape(B, 2)
    log_sig_perp = outputs["log_sigma_perp"].reshape(B, 2)
    sig_par2 = torch.exp(2.0 * log_sig_par)     # [B,2] per-node axial variance
    sig_perp2 = torch.exp(2.0 * log_sig_perp)   # [B,2] per-node perp variance
    v_n = batch["h_dec_base"][:, 0:3].reshape(B, 2, 3)
    e = batch["bond_unit_node"].reshape(B, 2, 3)[:, 0, :]
    dx = batch["edge_vec"][:B].norm(dim=-1)

    dv = v_n[:, 0] - v_n[:, 1]
    sv = v_n[:, 0] + v_n[:, 1]
    dmu = mu[:, 0] - mu[:, 1]
    smu = mu[:, 0] + mu[:, 1]
    dy = y[:, 0] - y[:, 1]
    sy = y[:, 0] + y[:, 1]

    dvx = (dv * e).sum(-1)
    dmu_par = (dmu * e).sum(-1)
    dy_par = (dy * e).sum(-1)
    dmu_perp = dmu - dmu_par[:, None] * e
    dy_perp = dy - dy_par[:, None] * e
    dv_perp = dv - dvx[:, None] * e

    # analytic sampled variance per graph (relative = node1+node2 independent)
    var_noise_par = sig_par2.sum(-1)                 # relative axial: s1^2 + s2^2
    var_noise_perp = 2.0 * sig_perp2.sum(-1)         # 2 perp dims, both nodes
    var_noise_com = (sig_par2 + 2.0 * sig_perp2).sum(-1)  # full 3D, both nodes

    edges = torch.linspace(dx_range[0], dx_range[1], nbins + 1, device=dx.device)
    idx = torch.bucketize(dx, edges[1:-1])

    loss = dx.new_zeros(())
    total_w = 0.0
    for b in range(nbins):
        m = idx == b
        n = int(m.sum())
        if n < min_bin:
            continue

        def rel_var(model_val, tgt_val):
            tgt = tgt_val.detach()
            return ((model_val - tgt_val) / (tgt + eps)) ** 2

        # fluctuation: total channel variance (mean spread + analytic sampled var)
        vp_m = dmu_par[m].var(unbiased=False) + var_noise_par[m].mean()
        vp_t = dy_par[m].var(unbiased=False)
        vperp_m = _centered_sq(dmu_perp[m]) + var_noise_perp[m].mean()
        vperp_t = _centered_sq(dy_perp[m])
        vcom_m = _centered_sq(smu[m]) + var_noise_com[m].mean()
        vcom_t = _centered_sq(sy[m])
        var_term = (
            w_par * rel_var(vp_m, vp_t)
            + w_perp * rel_var(vperp_m, vperp_t)
            + w_com * rel_var(vcom_m, vcom_t)
        )

        # dissipation: velocity-noise covariance (via decoded mean), matched as a
        # CORRELATION difference — denominator is the stable velocity*channel std
        # product (detached), not the tiny cov itself, so per-bin sampling noise
        # in the ~1e-4 covariance does not blow the term up.
        def _tot_std(a):
            if a.dim() == 1:
                a = a[:, None]
            return torch.sqrt(_centered_sq(a) + eps)

        def corr_diff(cov_m, cov_t, dv_series, tgt_series):
            denom = (_tot_std(dv_series) * _tot_std(tgt_series)).detach()
            return ((cov_m - cov_t) / (denom + eps)) ** 2

        cp_m = _batch_cov(dvx[m], dmu_par[m])
        cp_t = _batch_cov(dvx[m], dy_par[m])
        cperp_m = _batch_cov_vec(dv_perp[m], dmu_perp[m])
        cperp_t = _batch_cov_vec(dv_perp[m], dy_perp[m])
        ccom_m = _batch_cov_vec(sv[m], smu[m])
        ccom_t = _batch_cov_vec(sv[m], sy[m])

        cov_term = (
            w_par * corr_diff(cp_m, cp_t, dvx[m], dy_par[m])
            + w_perp * corr_diff(cperp_m, cperp_t, dv_perp[m], dy_perp[m])
            + w_com * corr_diff(ccom_m, ccom_t, sv[m], sy[m])
        )

        loss = loss + n * (w_var * var_term + w_cov * cov_term)
        total_w += n

    return loss / max(total_w, 1.0)

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
