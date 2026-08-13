import numpy as np
import torch
import matplotlib.pyplot as plt

from deepRD.noiseSampler.cvae import losses
from deepRD.noiseSampler.cvae.transforms import (
    build_local_frame,
    compute_dx_dvx,
    minimal_image_rel,
    to_xyz,
)

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_tensor(x, device="cpu", dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def split_dimer_trajectory(q, v, r):
    """
    Convert alternating bead trajectories [n_traj, 2*T, 3] into bead-wise
    concatenated arrays [n_traj, T, 6].
    """
    q = to_tensor(q)
    v = to_tensor(v)
    r = to_tensor(r)
    q_pair = torch.cat((q[:, 0::2], q[:, 1::2]), dim=-1)
    v_pair = torch.cat((v[:, 0::2], v[:, 1::2]), dim=-1)
    r_pair = torch.cat((r[:, 0::2], r[:, 1::2]), dim=-1)
    return q_pair, v_pair, r_pair


def infer_dimer_context(q_pair, v_pair, n_samples, boxsize=5.0, r_pair=None):
    """
    Infer the state time slice corresponding to flattened one-step samples.
    Dimer conditionings usually lose some timesteps e.g, q <- q[:, 1:-1].
    This helper also handles T-1 and T-3 style datasets.
    """
    q_pair = to_tensor(q_pair)
    v_pair = to_tensor(v_pair)
    if r_pair is not None:
        r_pair = to_tensor(r_pair)
    n_traj, T, _ = q_pair.shape
    if n_samples % n_traj != 0:
        raise ValueError("n_samples must be divisible by number of trajectories.")

    T_eff = n_samples // n_traj
    if T_eff == T - 1:
        sl = slice(0, -1)
    elif T_eff == T - 2:
        sl = slice(1, -1)
    elif T_eff == T - 3:
        sl = slice(2, -1)
    else:
        raise ValueError(
            f"Could not infer dimer time slice: T={T}, T_eff={T_eff}."
        )

    q_state = q_pair[:, sl, :]
    v_state = v_pair[:, sl, :]
    r_state = r_pair[:, sl, :] if r_pair is not None else None
    start = 0 if sl.start is None else sl.start
    stop = T if sl.stop is None else sl.stop
    prev_context_arrays = None
    if start > 0:
        prev_sl = slice(start - 1, stop - 1)
        q_prev = q_pair[:, prev_sl, :]
        v_prev = v_pair[:, prev_sl, :]
        r_prev = r_pair[:, prev_sl, :] if r_pair is not None else None
        q1_prev, q2_prev = q_prev[..., :3], q_prev[..., 3:]
        v1_prev, v2_prev = v_prev[..., :3], v_prev[..., 3:]
        dx_prev, dvx_prev = compute_dx_dvx(
            q1_prev, q2_prev, v1_prev, v2_prev, boxsize=boxsize
        )
        R_prev, _ = build_local_frame(q1_prev, q2_prev, boxsize=boxsize)
        prev_context_arrays = {
            "q_state_prev": q_prev,
            "v_state_prev": v_prev,
            "r_state_prev": r_prev,
            "q1_prev": q1_prev,
            "q2_prev": q2_prev,
            "v1_prev": v1_prev,
            "v2_prev": v2_prev,
            "R_prev": R_prev,
            "dx_prev": dx_prev,
            "dvx_prev": dvx_prev,
        }
    q1, q2 = q_state[..., :3], q_state[..., 3:]
    v1, v2 = v_state[..., :3], v_state[..., 3:]

    dx, dvx = compute_dx_dvx(q1, q2, v1, v2, boxsize=boxsize)
    R, _ = build_local_frame(q1, q2, boxsize=boxsize)

    flat = lambda x: x.reshape(-1, *x.shape[2:])
    context = {
        "q_state": flat(q_state),
        "v_state": flat(v_state),
        "q1": flat(q1),
        "q2": flat(q2),
        "v1": flat(v1),
        "v2": flat(v2),
        "R": flat(R),
        "dx": dx.reshape(-1),
        "dvx": dvx.reshape(-1),
        "T_eff": T_eff,
        "time_slice": sl,
    }
    if r_state is not None:
        context["r_state"] = flat(r_state)
    if prev_context_arrays is not None:
        context.update({
            "q_state_prev": flat(prev_context_arrays["q_state_prev"]),
            "v_state_prev": flat(prev_context_arrays["v_state_prev"]),
            "q1_prev": flat(prev_context_arrays["q1_prev"]),
            "q2_prev": flat(prev_context_arrays["q2_prev"]),
            "v1_prev": flat(prev_context_arrays["v1_prev"]),
            "v2_prev": flat(prev_context_arrays["v2_prev"]),
            "R_prev": flat(prev_context_arrays["R_prev"]),
            "dx_prev": prev_context_arrays["dx_prev"].reshape(-1),
            "dvx_prev": prev_context_arrays["dvx_prev"].reshape(-1),
        })
        if prev_context_arrays["r_state_prev"] is not None:
            context["r_state_prev"] = flat(prev_context_arrays["r_state_prev"])
    return context

def scaler_roundtrip_error(scaler, x, n=20000):
    """
    Compute the maximum absolute error of a scaler round-trip on a random subset of x.
    This is a sanity check to ensure the scaler is not losing precision.
    """
    x_np = to_numpy(x)
    if len(x_np) > n:
        idx = np.random.choice(len(x_np), n, replace=False)
        x_np = x_np[idx]
    y = scaler.inverse_transform(scaler.transform(x_np))
    return float(np.max(np.abs(y - x_np)))


@torch.no_grad()
def evaluate_standard_cvae(
    model,
    r_ph,
    c_ph,
    device="cpu",
    Tr=1.0,
    Tz=1.0,
):
    """
    One-step evaluation for a diagonal-Gaussian CVAE with standard Gaussian prior.

    Inputs are physical model-space arrays. For local-frame models, these are
    local-frame physical values; convert to global with `model_space_to_global`.
    """
    model.eval()
    r_ph_np = to_numpy(r_ph).astype(np.float32)
    c_ph_np = to_numpy(c_ph).astype(np.float32)

    # Normalize inputs with model scalers.
    r_norm = torch.as_tensor(
        model.scaler_r.transform(r_ph_np),
        dtype=torch.float32,
        device=device,
    )
    c_norm = torch.as_tensor(
        model.scaler_c.transform(c_ph_np),
        dtype=torch.float32,
        device=device,
    )

    # Forward pass through model to get reconstructions
    dec_out, q, p = model(r_norm, c_norm)
    rec_mu_norm, rec_log_sig = dec_out
    q_mu, q_logv = q
    p_mu, p_logv = p

    r_rec_norm = (
        rec_mu_norm
        + torch.exp(rec_log_sig) * torch.randn_like(rec_mu_norm) * Tr
    )

    # Sample from the prior for generation diagnostics (normalized space).
    z_prior = torch.randn(
        c_norm.shape[0], model.zdim, device=device, dtype=c_norm.dtype
    ) * Tz
    gen_mu_norm, gen_log_sig = model.decode(z_prior, c_norm)
    r_gen_norm = gen_mu_norm + torch.exp(gen_log_sig) * torch.randn_like(gen_mu_norm) * Tr

    # For calibration diagnostics, also decode the zero latent vector.
    z0 = torch.zeros_like(z_prior)
    z0_mu_norm, z0_log_sig = model.decode(z0, c_norm)
    z0_sample_norm = z0_mu_norm + torch.exp(z0_log_sig) * torch.randn_like(z0_mu_norm) * Tr

    # Compute losses from the reconstruction distribution for diagnostics.
    total, nll, kl = losses.elbo_loss(
        r_norm,
        dec_out,
        q,
        p,
        beta=1.0,
        per_sample=False,
    )
    nll_ps = losses.gaussian_nll_diag(
        r_norm,
        rec_mu_norm,
        rec_log_sig,
        per_sample=True,
    )
    kl_ps = losses.kl_diag(q_mu, q_logv, p_mu, p_logv, per_sample=True)

    def inv(x):
        """Inverse transform a tensor to the physical space using the model's scaler."""
        return model.scaler_r.inverse_transform(to_numpy(x))

    return {
        "r_true_model": r_ph_np,
        "c_model": c_ph_np,
        "r_true_norm": to_numpy(r_norm),
        "r_rec_mu_norm": to_numpy(rec_mu_norm),
        "r_rec_mu_phys": inv(rec_mu_norm),
        "r_rec_phys": inv(r_rec_norm),
        "r_gen_model": inv(r_gen_norm),
        "gen_mu_model": inv(gen_mu_norm),
        "r_z0_model": inv(z0_sample_norm),
        "z0_mu_model": inv(z0_mu_norm),
        "gen_mu_norm": to_numpy(gen_mu_norm),
        "z0_mu_norm": to_numpy(z0_mu_norm),
        "rec_log_sig": to_numpy(rec_log_sig),
        "gen_log_sig": to_numpy(gen_log_sig),
        "z0_log_sig": to_numpy(z0_log_sig),
        "q_mu": to_numpy(q_mu),
        "q_logv": to_numpy(q_logv),
        "kl_per_sample": to_numpy(kl_ps),
        "nll_per_sample": to_numpy(nll_ps),
        "loss": float(total.item()),
        "nll": float(nll.item()),
        "kl": float(kl.item()),
        "Tr": Tr,
        "Tz": Tz,
    }


def model_space_to_global(x_model, context, cond_type):
    """
    Convert model-space dimer outputs to global bead-wise physical vectors.
    Local-frame models are rotated back with the benchmark frame.
    """
    x_t = to_tensor(x_model)
    if cond_type.startswith("local_"):
        R = context["R"].to(dtype=x_t.dtype, device=x_t.device)
        x_t = to_xyz(R, x_t)
    return to_numpy(x_t)


def attach_global_outputs(results, context, cond_type):
    output_keys = {
        "r_true": "r_true_model",
        "r_rec_mu": "r_rec_mu_phys",
        "r_rec": "r_rec_phys",
        "r_gen": "r_gen_model",
        "gen_mu": "gen_mu_model",
        "z0_mu": "z0_mu_model",
        "r_z0": "r_z0_model",
    }
    for global_key, source_key in output_keys.items():
        if source_key in results:
            results[f"{global_key}_global"] = model_space_to_global(
                results[source_key], context, cond_type
            )
    return results


def rel_com_channels(r_global, context):
    r = to_tensor(r_global)
    q1 = context["q1"].to(dtype=r.dtype, device=r.device)
    q2 = context["q2"].to(dtype=r.dtype, device=r.device)
    v1 = context["v1"].to(dtype=r.dtype, device=r.device)
    v2 = context["v2"].to(dtype=r.dtype, device=r.device)
    rel_pos = minimal_image_rel(q1, q2, boxsize=5.0, boundary_type="periodic")
    dx = torch.linalg.norm(rel_pos, dim=-1, keepdim=True).clamp_min(1e-12)
    e = rel_pos / dx

    r1, r2 = r[..., :3], r[..., 3:]
    r_rel = r2 - r1
    r_com = 0.5 * (r1 + r2)
    v_rel = v2 - v1
    v_com = 0.5 * (v1 + v2)

    r1_par = torch.sum(r1 * e, dim=-1)
    r2_par = torch.sum(r2 * e, dim=-1)
    rel_par = torch.sum(r_rel * e, dim=-1)
    com_par = torch.sum(r_com * e, dim=-1)
    v_rel_par = torch.sum(v_rel * e, dim=-1)
    rel_perp = r_rel - rel_par.unsqueeze(-1) * e
    com_perp = r_com - com_par.unsqueeze(-1) * e
    r1_perp = r1 - r1_par.unsqueeze(-1) * e
    r2_perp = r2 - r2_par.unsqueeze(-1) * e
    v_rel_perp = v_rel - v_rel_par.unsqueeze(-1) * e

    return {
        "r1_x": r1[:, 0].cpu().numpy(),
        "r1_y": r1[:, 1].cpu().numpy(),
        "r1_z": r1[:, 2].cpu().numpy(),
        "r2_x": r2[:, 0].cpu().numpy(),
        "r2_y": r2[:, 1].cpu().numpy(),
        "r2_z": r2[:, 2].cpu().numpy(),
        "||r1||": torch.linalg.norm(r1, dim=-1).cpu().numpy(),
        "||r2||": torch.linalg.norm(r2, dim=-1).cpu().numpy(),
        "r1_parallel": r1_par.cpu().numpy(),
        "r2_parallel": r2_par.cpu().numpy(),
        "||r1_perp||": torch.linalg.norm(r1_perp, dim=-1).cpu().numpy(),
        "||r2_perp||": torch.linalg.norm(r2_perp, dim=-1).cpu().numpy(),
        "r_rel_parallel": rel_par.cpu().numpy(),
        "||r_rel_perp||": torch.linalg.norm(rel_perp, dim=-1).cpu().numpy(),
        "r_com_parallel": com_par.cpu().numpy(),
        "||r_com_perp||": torch.linalg.norm(com_perp, dim=-1).cpu().numpy(),
        "r_rel · v_rel": torch.sum(r_rel * v_rel, dim=-1).cpu().numpy(),
        "r_com · v_com": torch.sum(r_com * v_com, dim=-1).cpu().numpy(),
        "r_rel_parallel * Delta v_x": (
            rel_par * to_tensor(context["dvx"], dtype=r.dtype).to(device=r.device)
        ).cpu().numpy(),
        "||r_rel_perp|| * ||v_rel_perp||": (
            torch.linalg.norm(rel_perp, dim=-1)
            * torch.linalg.norm(v_rel_perp, dim=-1)
        ).cpu().numpy(),
        "r1_norm": torch.linalg.norm(r1, dim=-1).cpu().numpy(),
        "r2_norm": torch.linalg.norm(r2, dim=-1).cpu().numpy(),
        "r_norm": torch.linalg.norm(r, dim=-1).cpu().numpy(),
        "rel_norm": torch.linalg.norm(r_rel, dim=-1).cpu().numpy(),
        "com_norm": torch.linalg.norm(r_com, dim=-1).cpu().numpy(),
        "rel_parallel": rel_par.cpu().numpy(),
        "rel_perp_norm": torch.linalg.norm(rel_perp, dim=-1).cpu().numpy(),
        "com_parallel": com_par.cpu().numpy(),
        "com_perp_norm": torch.linalg.norm(com_perp, dim=-1).cpu().numpy(),
    }


COMPONENT_ALIASES = {
    "r1_norm": "||r1||",
    "r2_norm": "||r2||",
    "r1_perp_norm": "||r1_perp||",
    "r2_perp_norm": "||r2_perp||",
    "rel_parallel": "r_rel_parallel",
    "rel_perp_norm": "||r_rel_perp||",
    "com_parallel": "r_com_parallel",
    "com_perp_norm": "||r_com_perp||",
    "r_rel_dot_v_rel": "r_rel · v_rel",
    "r_com_dot_v_com": "r_com · v_com",
    "r_rel_parallel_x_dvx": "r_rel_parallel * Delta v_x",
    "rel_perp_norm_x_v_rel_perp_norm": "||r_rel_perp|| * ||v_rel_perp||",
}


ONE_STEP_COMPONENTS = (
    "r1_x",
    "r1_y",
    "r1_z",
    "r2_x",
    "r2_y",
    "r2_z",
    "||r1||",
    "||r2||",
    "r1_parallel",
    "r2_parallel",
    "||r1_perp||",
    "||r2_perp||",
    "r_rel_parallel",
    "||r_rel_perp||",
    "r_com_parallel",
    "||r_com_perp||",
    "r_rel · v_rel",
    "r_com · v_com",
    "r_rel_parallel * Delta v_x",
    "||r_rel_perp|| * ||v_rel_perp||",
)


def canonical_component(component):
    return COMPONENT_ALIASES.get(component, component)


def component_values(r_global, context, component):
    values = rel_com_channels(r_global, context)
    component = canonical_component(component)
    if component not in values:
        available = ", ".join(ONE_STEP_COMPONENTS)
        raise KeyError(f"Unknown component '{component}'. Available components: {available}")
    return values[component]


def summary_table(results):
    true = results["r_true_global"]
    gen = results["r_gen_global"]
    rec = results["r_rec_mu_global"]

    rows = []
    labels = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
    for j, label in enumerate(labels):
        rows.append({
            "coord": label,
            "true_mean": true[:, j].mean(),
            "gen_mean": gen[:, j].mean(),
            "true_std": true[:, j].std(),
            "gen_std": gen[:, j].std(),
            "gen_mean_err": gen[:, j].mean() - true[:, j].mean(),
            "gen_std_err": gen[:, j].std() - true[:, j].std(),
            "rec_mu_mae": np.mean(np.abs(rec[:, j] - true[:, j])),
            "rec_mu_rmse": np.sqrt(np.mean((rec[:, j] - true[:, j]) ** 2)),
        })
    return rows


def scalar_metrics_table(results):
    """
    Compact scalar metrics for reconstruction and generation. Reconstruction errors
    are meaningful pointwise; generation metrics compare marginal moments.
    """
    true_g = results["r_true_global"]
    rec_mu_g = results["r_rec_mu_global"]
    rec_g = results["r_rec_global"]
    gen_g = results["r_gen_global"]

    true_n = results["r_true_norm"]
    rec_mu_n = results["r_rec_mu_norm"]
    z0_resid = (
        (results["r_true_norm"] - results["z0_mu_norm"])
        / (np.exp(results["z0_log_sig"]) + 1e-12)
    )

    rows = [
        {
            "metric": "ELBO beta=1",
            "value": results["loss"],
            "space": "normalized",
        },
        {"metric": "NLL", "value": results["nll"], "space": "normalized"},
        {"metric": "KL", "value": results["kl"], "space": "normalized"},
        {
            "metric": "recon_mu MAE",
            "value": np.mean(np.abs(rec_mu_g - true_g)),
            "space": "physical global",
        },
        {
            "metric": "recon_mu RMSE",
            "value": np.sqrt(np.mean((rec_mu_g - true_g) ** 2)),
            "space": "physical global",
        },
        {
            "metric": "recon MAE",
            "value": np.mean(np.abs(rec_g - true_g)),
            "space": "physical global",
        },
        {
            "metric": "gen marginal mean abs err",
            "value": np.mean(np.abs(gen_g.mean(axis=0) - true_g.mean(axis=0))),
            "space": "physical global",
        },
        {
            "metric": "gen marginal std abs err",
            "value": np.mean(np.abs(gen_g.std(axis=0) - true_g.std(axis=0))),
            "space": "physical global",
        },
        {
            "metric": "recon_mu normalized MAE",
            "value": np.mean(np.abs(rec_mu_n - true_n)),
            "space": "normalized",
        },
        {
            "metric": "standardized residual mean",
            "value": np.mean(z0_resid),
            "space": "normalized z=0",
        },
        {
            "metric": "standardized residual std",
            "value": np.std(z0_resid),
            "space": "normalized z=0",
        },
    ]

    for name, sig in (
        ("posterior sigma", np.exp(results["rec_log_sig"])),
        ("prior-sample sigma", np.exp(results["gen_log_sig"])),
        ("z=0 sigma", np.exp(results["z0_log_sig"])),
    ):
        rows.extend([
            {"metric": f"{name} mean", "value": sig.mean(), "space": "normalized"},
            {"metric": f"{name} median", "value": np.median(sig), "space": "normalized"},
            {"metric": f"{name} q05", "value": np.quantile(sig, 0.05), "space": "normalized"},
            {"metric": f"{name} q95", "value": np.quantile(sig, 0.95), "space": "normalized"},
        ])

    return rows


def print_scalar_summary(results):
    log_sig = results["gen_log_sig"]
    sigma = np.exp(log_sig)
    print(f"ELBO(beta=1): {results['loss']:.4f}")
    print(f"NLL:          {results['nll']:.4f}")
    print(f"KL:           {results['kl']:.4f}")
    print(
        "Generated decoder sigma, normalized space: "
        f"mean={sigma.mean():.4f}, median={np.median(sigma):.4f}, "
        f"q05={np.quantile(sigma, 0.05):.4f}, q95={np.quantile(sigma, 0.95):.4f}"
    )


def finite_summary(results):
    keys = [
        "r_true_global",
        "r_rec_mu_global",
        "r_rec_global",
        "r_gen_global",
        "rec_log_sig",
        "gen_log_sig",
        "q_mu",
        "q_logv",
    ]
    return [{"array": key, "finite": bool(np.isfinite(results[key]).all())} for key in keys]


def plot_coordinate_marginals(
    results,
    bins=120,
    xlim=None,
    recon_key="r_rec_global",
    recon_label="recon",
):
    true = results["r_true_global"]
    rec = results[recon_key]
    gen = results["r_gen_global"]
    labels = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        vals = [true[:, j], gen[:, j], rec[:, j]]
        lo = min(v.min() for v in vals)
        hi = max(v.max() for v in vals)
        if xlim is not None:
            lo, hi = xlim
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(true[:, j], bins=edges, density=True, histtype="step", lw=2, label="true")
        ax.hist(gen[:, j], bins=edges, density=True, histtype="step", lw=1.8, label="generated")
        ax.hist(rec[:, j], bins=edges, density=True, histtype="step", lw=1.3, label=recon_label)
        ax.set_title(labels[j])
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    return fig, axes


def plot_channel_marginals(
    results,
    context,
    bins=120,
    recon_key="r_rec_global",
    recon_label="recon",
):
    true_ch = rel_com_channels(results["r_true_global"], context)
    gen_ch = rel_com_channels(results["r_gen_global"], context)
    rec_ch = rel_com_channels(results[recon_key], context)

    names = [
        ("r1_norm", "||r1||"),
        ("r2_norm", "||r2||"),
        ("rel_parallel", "r_rel parallel"),
        ("rel_perp_norm", "||r_rel perp||"),
        ("com_parallel", "r_com parallel"),
        ("com_perp_norm", "||r_com perp||"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for key, title, ax in zip([n[0] for n in names], [n[1] for n in names], axes.flat):
        vals = [true_ch[key], gen_ch[key], rec_ch[key]]
        lo = min(v.min() for v in vals)
        hi = max(v.max() for v in vals)
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(true_ch[key], bins=edges, density=True, histtype="step", lw=2, label="true")
        ax.hist(gen_ch[key], bins=edges, density=True, histtype="step", lw=1.8, label="generated")
        ax.hist(rec_ch[key], bins=edges, density=True, histtype="step", lw=1.3, label=recon_label)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    return fig, axes


def plot_decoder_mean_marginals(results, bins=120, xlim=None):
    """
    Plot deterministic decoder means separately from the sampled variables.
    These curves are expected to be narrower than the full stochastic samples.
    """
    true = results["r_true_global"]
    rec_mu = results["r_rec_mu_global"]
    gen_mu = results["gen_mu_global"]
    z0_mu = results["z0_mu_global"]
    labels = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        vals = [true[:, j], rec_mu[:, j], gen_mu[:, j], z0_mu[:, j]]
        lo = min(v.min() for v in vals)
        hi = max(v.max() for v in vals)
        if xlim is not None:
            lo, hi = xlim
        edges = np.linspace(lo, hi, bins + 1)
        #ax.hist(true[:, j], bins=edges, density=True, histtype="step", lw=2, label="true")
        ax.hist(rec_mu[:, j], bins=edges, density=True, histtype="step", lw=1.5, label="posterior mean")
        ax.hist(gen_mu[:, j], bins=edges, density=True, histtype="step", lw=1.5, label="prior-sample mean")
        #ax.hist(z0_mu[:, j], bins=edges, density=True, histtype="step", lw=1.2, label="z=0 mean")
        ax.set_title(labels[j])
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    return fig, axes


def plot_decoder_sigma_marginals(results, bins=120, xlim=None):
    """
    Plot decoder standard deviations in normalized model space.
    """
    rec_sig = np.exp(results["rec_log_sig"])
    gen_sig = np.exp(results["gen_log_sig"])
    z0_sig = np.exp(results["z0_log_sig"])
    labels = ("dim 0", "dim 1", "dim 2", "dim 3", "dim 4", "dim 5")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        vals = [rec_sig[:, j], gen_sig[:, j], z0_sig[:, j]]
        lo = min(v.min() for v in vals)
        hi = max(v.max() for v in vals)
        if xlim is not None:
            lo, hi = xlim
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(rec_sig[:, j], bins=edges, density=True, histtype="step", lw=1.5, label="posterior sigma")
        ax.hist(gen_sig[:, j], bins=edges, density=True, histtype="step", lw=1.5, label="prior-sample sigma")
        #ax.hist(z0_sig[:, j], bins=edges, density=True, histtype="step", lw=1.2, label="z=0 sigma")
        ax.set_title(labels[j])
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    return fig, axes


def binned_stats(x, y, nbins=40, xlim=None, min_count=200):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    lo, hi = (x.min(), x.max()) if xlim is None else xlim
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(nbins, np.nan)
    var = np.full(nbins, np.nan)
    q05 = np.full(nbins, np.nan)
    q95 = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        if i == nbins - 1:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            m = (x >= edges[i]) & (x < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] < min_count:
            continue
        yy = y[m]
        mean[i] = yy.mean()
        var[i] = yy.var()
        q05[i], q95[i] = np.quantile(yy, [0.05, 0.95])
    return {
        "centers": centers,
        "mean": mean,
        "var": var,
        "q05": q05,
        "q95": q95,
        "counts": counts,
    }


def binned_tail_prob(x, y, threshold, nbins=40, xlim=None, min_count=200):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    lo, hi = (x.min(), x.max()) if xlim is None else xlim
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        if i == nbins - 1:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            m = (x >= edges[i]) & (x < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] < min_count:
            continue
        prob[i] = np.mean(np.abs(y[m]) > threshold)
    return {"centers": centers, "prob": prob, "counts": counts}


def plot_conditional_dx_component(
    results,
    context,
    component="r_rel_parallel",
    nbins=40,
    xlim=(0.2, 1.9),
    tail_threshold=None,
    min_count=200,
):
    component = canonical_component(component)
    dx = to_numpy(context["dx"])
    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)
    rec_y = component_values(results["r_rec_global"], context, component)

    stats = {
        "true": binned_stats(dx, true_y, nbins, xlim, min_count),
        "generated": binned_stats(dx, gen_y, nbins, xlim, min_count),
        "recon": binned_stats(dx, rec_y, nbins, xlim, min_count),
    }
    if tail_threshold is None:
        tail_threshold = np.quantile(np.abs(true_y), 0.9)
    tails = {
        "true": binned_tail_prob(dx, true_y, tail_threshold, nbins, xlim, min_count),
        "generated": binned_tail_prob(dx, gen_y, tail_threshold, nbins, xlim, min_count),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for label, st in stats.items():
        axes[0, 0].plot(st["centers"], st["mean"], label=label)
        axes[0, 1].plot(st["centers"], st["var"], label=label)
        axes[1, 0].plot(st["centers"], st["q05"], alpha=0.8, label=f"{label} q05")
        axes[1, 0].plot(st["centers"], st["q95"], alpha=0.8, linestyle="--", label=f"{label} q95")
    for label, st in tails.items():
        axes[1, 1].plot(st["centers"], st["prob"], label=label)

    axes[0, 0].set_ylabel(f"E[{component} | dx]")
    axes[0, 1].set_ylabel(f"Var[{component} | dx]")
    axes[1, 0].set_ylabel("5/95% quantiles")
    axes[1, 1].set_ylabel(f"P(|{component}| > {tail_threshold:.3g} | dx)")
    for ax in axes.flat:
        ax.set_xlabel("dx")
        ax.grid(alpha=0.25)
        ax.legend()
    return fig, axes, stats, tails


def plot_conditional_rel_parallel(*args, **kwargs):
    kwargs.setdefault("component", "r_rel_parallel")
    return plot_conditional_dx_component(*args, **kwargs)


def plot_conditional_dx_extended(
    results,
    context,
    component="||r_rel_perp||",
    nbins=40,
    xlim=(0.2, 1.9),
    min_count=200,
):
    """
    Additional Delta x diagnostics for a chosen component and bin counts.
    """
    component = canonical_component(component)
    dx = to_numpy(context["dx"])
    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)
    rec_y = component_values(results["r_rec_global"], context, component)

    stats = {
        "true": binned_stats(dx, true_y, nbins, xlim, min_count),
        "generated": binned_stats(dx, gen_y, nbins, xlim, min_count),
        "recon": binned_stats(dx, rec_y, nbins, xlim, min_count),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for label, st in stats.items():
        axes[0].plot(st["centers"], st["mean"], label=label)
        axes[1].plot(st["centers"], st["var"], label=label)
    axes[2].bar(stats["true"]["centers"], stats["true"]["counts"], width=(xlim[1] - xlim[0]) / nbins)

    axes[0].set_ylabel(f"E[{component} | dx]")
    axes[1].set_ylabel(f"Var[{component} | dx]")
    axes[2].set_ylabel("benchmark bin count")
    for ax in axes:
        ax.set_xlabel("dx")
        ax.grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    return fig, axes, stats


def plot_conditional_scalar(
    x,
    results,
    context,
    xlabel,
    component="rel_parallel",
    nbins=40,
    xlim=None,
    min_count=200,
):
    """
    Compare conditional mean/variance of a dimer channel versus an arbitrary scalar.
    """
    component = canonical_component(component)
    x = to_numpy(x)
    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)
    rec_y = component_values(results["r_rec_global"], context, component)

    stats = {
        "true": binned_stats(x, true_y, nbins, xlim, min_count),
        "generated": binned_stats(x, gen_y, nbins, xlim, min_count),
        "recon": binned_stats(x, rec_y, nbins, xlim, min_count),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, st in stats.items():
        axes[0].plot(st["centers"], st["mean"], label=label)
        axes[1].plot(st["centers"], st["var"], label=label)
    axes[0].set_ylabel(f"E[{component} | {xlabel}]")
    axes[1].set_ylabel(f"Var[{component} | {xlabel}]")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        ax.legend()
    return fig, axes, stats


def heatmap2d_stats(x, y, z, nbins=(35, 35), xlim=None, ylim=None, min_count=100):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(z).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    xlo, xhi = (x.min(), x.max()) if xlim is None else xlim
    ylo, yhi = (y.min(), y.max()) if ylim is None else ylim
    xedges = np.linspace(xlo, xhi, nbins[0] + 1)
    yedges = np.linspace(ylo, yhi, nbins[1] + 1)
    mean = np.full(nbins, np.nan)
    var = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    xi = np.digitize(x, xedges) - 1
    yi = np.digitize(y, yedges) - 1
    valid = (xi >= 0) & (xi < nbins[0]) & (yi >= 0) & (yi < nbins[1])
    xi, yi, z = xi[valid], yi[valid], z[valid]
    for i in range(nbins[0]):
        for j in range(nbins[1]):
            m = (xi == i) & (yi == j)
            counts[i, j] = int(m.sum())
            if counts[i, j] < min_count:
                continue
            zz = z[m]
            mean[i, j] = zz.mean()
            var[i, j] = zz.var()
    return {"xedges": xedges, "yedges": yedges, "mean": mean, "var": var, "counts": counts}


def plot_2d_conditionals(
    results,
    context,
    component="rel_parallel",
    nbins=(35, 35),
    xlim=(0.2, 1.9),
    ylim=(-0.8, 0.8),
    min_count=80,
):
    """
    Heatmaps of conditional mean/variance versus (Delta x, Delta v_x).
    """
    component = canonical_component(component)
    dx = to_numpy(context["dx"])
    dvx = to_numpy(context["dvx"])
    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)

    true_stats = heatmap2d_stats(dx, dvx, true_y, nbins, xlim, ylim, min_count)
    gen_stats = heatmap2d_stats(dx, dvx, gen_y, nbins, xlim, ylim, min_count)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [
        (f"true mean: {component}", true_stats["mean"], axes[0, 0]),
        (f"generated mean: {component}", gen_stats["mean"], axes[0, 1]),
        (f"true variance: {component}", true_stats["var"], axes[1, 0]),
        (f"generated variance: {component}", gen_stats["var"], axes[1, 1]),
    ]
    for title, values, ax in panels:
        im = ax.pcolormesh(true_stats["xedges"], true_stats["yedges"], values.T, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Delta x")
        ax.set_ylabel("Delta v_x")
        fig.colorbar(im, ax=ax)
    return fig, axes, {"true": true_stats, "generated": gen_stats}


def plot_decoder_calibration(results, bins=100):
    residual = (
        (results["r_true_norm"] - results["z0_mu_norm"])
        / (np.exp(results["z0_log_sig"]) + 1e-12)
    ).reshape(-1)
    sigma = np.exp(results["z0_log_sig"]).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].hist(residual, bins=bins, density=True, histtype="step", lw=2)
    axes[0].set_title("Standardized residuals at z=0")
    axes[0].set_xlabel("(r_true - mu) / sigma")
    axes[0].set_xlim(-8, 8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(sigma, bins=bins, density=True, histtype="step", lw=2)
    axes[1].set_title("Decoder sigma at z=0")
    axes[1].set_xlabel("sigma, normalized units")
    axes[1].grid(alpha=0.25)
    return fig, axes


def augmentation_region_summary(
    results,
    context,
    component="r_rel_parallel",
    dx_max=0.75,
    r_norm_quantile=0.9,
):
    """
    Diagnostics inside/outside a simple augmentation-like region:
    small bond length and large current auxiliary norm.
    """
    if "r_state" not in context:
        raise ValueError("context must include r_state. Call infer_dimer_context(..., r_pair=r_pair).")

    component = canonical_component(component)
    dx = to_numpy(context["dx"])
    r_state = to_numpy(context["r_state"])
    r_norm = np.linalg.norm(r_state, axis=-1)
    r_threshold = np.quantile(r_norm, r_norm_quantile)
    mask = (dx < dx_max) & (r_norm > r_threshold)

    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)
    sigma = np.exp(results["gen_log_sig"]).mean(axis=1)
    resid = (
        (results["r_true_norm"] - results["z0_mu_norm"])
        / (np.exp(results["z0_log_sig"]) + 1e-12)
    )
    resid_l2 = np.linalg.norm(resid, axis=1)

    rows = []
    for name, m in (("inside", mask), ("outside", ~mask)):
        if m.sum() == 0:
            continue
        rows.append({
            "region": name,
            "count": int(m.sum()),
            "fraction": float(m.mean()),
            "component": component,
            "true_mean": float(true_y[m].mean()),
            "gen_mean": float(gen_y[m].mean()),
            "true_var": float(true_y[m].var()),
            "gen_var": float(gen_y[m].var()),
            "sigma_mean": float(sigma[m].mean()),
            "nll_mean": float(results["nll_per_sample"][m].mean()),
            "std_resid_l2_mean": float(resid_l2[m].mean()),
        })
    return rows, mask, r_threshold


def plot_augmentation_region(results, context, mask, component="r_rel_parallel", bins=100):
    component = canonical_component(component)
    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)
    sigma = np.exp(results["gen_log_sig"]).mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name, m in (("inside", mask), ("outside", ~mask)):
        if m.sum() == 0:
            continue
        axes[0].hist(true_y[m], bins=bins, density=True, histtype="step", label=f"true {name}")
        axes[0].hist(gen_y[m], bins=bins, density=True, histtype="step", linestyle="--", label=f"gen {name}")
        axes[1].hist(sigma[m], bins=bins, density=True, histtype="step", label=name)
        axes[2].hist(results["nll_per_sample"][m], bins=bins, density=True, histtype="step", label=name)
    axes[0].set_title(f"{component} by region")
    axes[1].set_title("mean decoder sigma by region")
    axes[2].set_title("NLL by region")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    return fig, axes


@torch.no_grad()
def multi_sample_calibration(
    model,
    r_ph,
    c_ph,
    n_conditions=2000,
    n_draws=100,
    device="cpu",
    Tr=1.0,
    Tz=1.0,
    seed=None,
):
    """
    Draw multiple prior samples per real condition and compute compact
    calibration metrics in normalized model space.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    r_np = to_numpy(r_ph).astype(np.float32)
    c_np = to_numpy(c_ph).astype(np.float32)
    n = min(n_conditions, len(r_np))
    idx = np.random.choice(len(r_np), n, replace=False)

    r_norm = torch.as_tensor(model.scaler_r.transform(r_np[idx]), dtype=torch.float32, device=device)
    c_norm = torch.as_tensor(model.scaler_c.transform(c_np[idx]), dtype=torch.float32, device=device)

    draws = []
    for _ in range(n_draws):
        z = torch.randn(n, model.zdim, device=device, dtype=c_norm.dtype) * Tz
        mu, log_sig = model.decode(z, c_norm)
        sample = mu + torch.exp(log_sig) * torch.randn_like(mu) * Tr
        draws.append(sample.unsqueeze(0))
    draws = torch.cat(draws, dim=0).cpu().numpy()
    r_eval = r_norm.cpu().numpy()

    sample_mean = draws.mean(axis=0)
    sample_var = draws.var(axis=0)
    dist_to_cloud_mean = np.linalg.norm(r_eval - sample_mean, axis=1)

    coverages = []
    for level in (0.50, 0.90, 0.95):
        lo_q = 0.5 * (1.0 - level)
        hi_q = 1.0 - lo_q
        lo = np.quantile(draws, lo_q, axis=0)
        hi = np.quantile(draws, hi_q, axis=0)
        coverages.append({
            "level": level,
            "coordinate_coverage_mean": float(((r_eval >= lo) & (r_eval <= hi)).mean()),
            "all_dim_coverage": float(np.all((r_eval >= lo) & (r_eval <= hi), axis=1).mean()),
        })

    ranks = np.mean(draws < r_eval[None, :, :], axis=0).reshape(-1)
    return {
        "idx": idx,
        "draws": draws,
        "sample_mean": sample_mean,
        "sample_var": sample_var,
        "dist_to_cloud_mean": dist_to_cloud_mean,
        "coverages": coverages,
        "ranks": ranks,
    }


def plot_multi_sample_calibration(cal, bins=80):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].hist(cal["dist_to_cloud_mean"], bins=bins, density=True, histtype="step", lw=2)
    axes[0].set_title("||r_true - E[sample|c]||")
    axes[1].hist(cal["sample_var"].reshape(-1), bins=bins, density=True, histtype="step", lw=2)
    axes[1].set_title("Empirical sample variance")
    axes[2].hist(cal["ranks"], bins=np.linspace(0, 1, 31), density=True, histtype="step", lw=2)
    axes[2].set_title("PIT/rank histogram")
    for ax in axes:
        ax.grid(alpha=0.25)
    return fig, axes


def _context_at_suffix(context, suffix):
    if not suffix:
        return context
    mapped = dict(context)
    for base in ("q_state", "v_state", "r_state", "q1", "q2", "v1", "v2", "R", "dx", "dvx"):
        suffixed = f"{base}_{suffix}"
        if suffixed in context:
            mapped[base] = context[suffixed]
    return mapped


def _safe_corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def one_step_correlation_table(
    results,
    context,
    y_components=("r_rel_parallel",),
    x_components=None,
):
    """
    Cheap one-step correlations between state variables and next auxiliary
    components. Each row puts true/generated correlations for the same x/y pair
    next to each other.
    """
    if isinstance(y_components, str):
        y_components = (y_components,)
    y_components = tuple(canonical_component(c) for c in y_components)
    if x_components is None:
        x_components = y_components
    elif isinstance(x_components, str):
        x_components = (x_components,)
    x_components = tuple(canonical_component(c) for c in x_components)

    true_components = rel_com_channels(results["r_true_global"], context)
    gen_components = rel_com_channels(results["r_gen_global"], context)
    rows = []

    scalar_candidates = {
        "Delta x": to_numpy(context["dx"]),
        "Delta v_x": to_numpy(context["dvx"]),
    }
    vector_sources = []
    if "r_state" in context:
        vector_sources.append(("r_n", context["r_state"], context))
    if "v_state" in context:
        vector_sources.append(("v_n", context["v_state"], context))
    if "r_state_prev" in context:
        vector_sources.append(("r_n-1", context["r_state_prev"], _context_at_suffix(context, "prev")))
    if "v_state_prev" in context:
        vector_sources.append(("v_n-1", context["v_state_prev"], _context_at_suffix(context, "prev")))

    for y_component in y_components:
        y_true = true_components[y_component]
        y_gen = gen_components[y_component]
        for xname, x in scalar_candidates.items():
            rows.append({
                "x": xname,
                "y": y_component,
                "corr true": _safe_corr(x, y_true),
                "corr gen": _safe_corr(x, y_gen),
            })
        for prefix, values, source_context in vector_sources:
            for x_component in x_components:
                x = component_values(values, source_context, x_component)
                rows.append({
                    "x": f"{prefix} {x_component}",
                    "y": y_component,
                    "corr true": _safe_corr(x, y_true),
                    "corr gen": _safe_corr(x, y_gen),
                })
    return rows


def r_component_covariance(results):
    labels = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
    true = results["r_true_global"]
    gen = results["r_gen_global"]
    true_cov = np.cov(true[:, :6], rowvar=False)
    gen_cov = np.cov(gen[:, :6], rowvar=False)
    rows = []
    for i, row_label in enumerate(labels):
        row = {"component": row_label}
        for j, col_label in enumerate(labels):
            row[f"true {col_label}"] = true_cov[i, j]
            row[f"gen {col_label}"] = gen_cov[i, j]
            row[f"diff {col_label}"] = gen_cov[i, j] - true_cov[i, j]
        rows.append(row)
    return rows, true_cov, gen_cov, labels


def plot_r_component_covariance(results):
    rows, true_cov, gen_cov, labels = r_component_covariance(results)
    vmax = np.nanmax(np.abs([true_cov, gen_cov, gen_cov - true_cov]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = (
        ("true covariance", true_cov),
        ("generated covariance", gen_cov),
        ("generated - true", gen_cov - true_cov),
    )
    for ax, (title, values) in zip(axes, panels):
        im = ax.imshow(values, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        fig.colorbar(im, ax=ax)
    return fig, axes, rows


def plot_one_step_hexbin(results, context, component="r_rel_parallel", gridsize=80):
    component = canonical_component(component)
    true_y = component_values(results["r_true_global"], context, component)
    gen_y = component_values(results["r_gen_global"], context, component)
    dx = to_numpy(context["dx"])
    dvx = to_numpy(context["dvx"])

    columns = [
        (f"dx vs {component}", dx, true_y, gen_y),
        (f"dvx vs {component}", dvx, true_y, gen_y),
    ]
    if "r_state" in context:
        r_state_y = component_values(to_numpy(context["r_state"]), context, component)
        columns.append((
            f"r_n {component} vs next",
            r_state_y,
            true_y,
            gen_y,
        ))

    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False, constrained_layout=True)
    for col, (title, x, y_true, y_gen) in enumerate(columns):
        for row, (row_label, y) in enumerate((("true", y_true), ("generated", y_gen))):
            ax = axes[row, col]
            hb = ax.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=1)
            ax.set_title(f"{row_label}: {title}")
            ax.grid(alpha=0.2)
            fig.colorbar(hb, ax=ax)
    return fig, axes


@torch.no_grad()
def decoder_sensitivity_to_z(model, c_ph, n_conditions=2000, n_draws=100, device="cpu", Tz=1.0):
    c_np = to_numpy(c_ph).astype(np.float32)
    n = min(n_conditions, len(c_np))
    idx = np.random.choice(len(c_np), n, replace=False)
    c_norm = torch.as_tensor(model.scaler_c.transform(c_np[idx]), dtype=torch.float32, device=device)

    mus = []
    for _ in range(n_draws):
        z = torch.randn(n, model.zdim, device=device, dtype=c_norm.dtype) * Tz
        mu, _ = model.decode(z, c_norm)
        mus.append(mu.unsqueeze(0))
    mus = torch.cat(mus, dim=0)
    mu_std_over_z = mus.std(dim=0)
    z0 = torch.zeros(n, model.zdim, device=device, dtype=c_norm.dtype)
    mu_z0, _ = model.decode(z0, c_norm)
    return {
        "idx": idx,
        "mean_mu_std_over_z": float(mu_std_over_z.mean().item()),
        "max_mu_std_over_z": float(mu_std_over_z.max().item()),
        "per_dim_mean_std_over_z": mu_std_over_z.mean(dim=0).cpu().numpy(),
        "per_dim_std_across_c_z0": mu_z0.std(dim=0).cpu().numpy(),
        "ratio_z_to_c": float(mu_std_over_z.mean().item() / (mu_z0.std(dim=0).mean().item() + 1e-12)),
    }


def _repeat_context_index(context, index, n_repeat):
    sub = {}
    for key, value in context.items():
        if key in ("T_eff", "time_slice"):
            sub[key] = value
            continue
        if torch.is_tensor(value):
            sub[key] = value[index:index + 1].repeat(
                n_repeat,
                *([1] * (value.ndim - 1)),
            )
        else:
            arr = np.asarray(value)
            sub[key] = torch.as_tensor(arr[index:index + 1]).repeat(
                n_repeat,
                *([1] * (arr.ndim - 1)),
            )
    return sub


@torch.no_grad()
def latent_traversal(
    model,
    c_ph,
    context,
    cond_type,
    indices=None,
    dx_quantiles=(0.2, 0.5, 0.8),
    z_min=-3.0,
    z_max=3.0,
    n_points=101,
    device="cpu",
):
    """
    Sweep one latent dimension at a time while holding c fixed and all other
    latent coordinates at zero. The decoded mean is converted to global
    physical dimer channels.
    """
    c_np = to_numpy(c_ph).astype(np.float32)
    dx = to_numpy(context["dx"])
    dvx = to_numpy(context["dvx"])

    if indices is None:
        indices = []
        for q in dx_quantiles:
            target = np.quantile(dx, q)
            indices.append(int(np.argmin(np.abs(dx - target))))
    indices = list(indices)

    z_values = np.linspace(z_min, z_max, n_points, dtype=np.float32)
    curves = []

    for source_index in indices:
        c_one = np.repeat(c_np[source_index:source_index + 1], n_points, axis=0)
        c_norm = torch.as_tensor(
            model.scaler_c.transform(c_one),
            dtype=torch.float32,
            device=device,
        )
        sub_context = _repeat_context_index(context, source_index, n_points)

        for latent_dim in range(model.zdim):
            z = torch.zeros(n_points, model.zdim, dtype=c_norm.dtype, device=device)
            z[:, latent_dim] = torch.as_tensor(z_values, dtype=c_norm.dtype, device=device)
            mu_norm, log_sig = model.decode(z, c_norm)
            mu_model = model.scaler_r.inverse_transform(to_numpy(mu_norm))
            mu_global = model_space_to_global(mu_model, sub_context, cond_type)
            channels = rel_com_channels(mu_global, sub_context)

            curves.append({
                "source_index": source_index,
                "latent_dim": latent_dim,
                "z_values": z_values.copy(),
                "dx": float(dx[source_index]),
                "dvx": float(dvx[source_index]),
                "mu_global": mu_global,
                "sigma_norm": np.exp(to_numpy(log_sig)),
                "channels": channels,
            })

    return {
        "indices": indices,
        "z_values": z_values,
        "curves": curves,
        "zdim": model.zdim,
    }


def plot_latent_traversal_channels(
    traversal,
    channels=("rel_parallel", "rel_perp_norm", "com_parallel", "com_perp_norm"),
):
    zdim = traversal["zdim"]
    fig, axes = plt.subplots(
        zdim,
        len(channels),
        figsize=(4.2 * len(channels), 3.2 * zdim),
        squeeze=False,
        constrained_layout=True,
    )
    for curve in traversal["curves"]:
        dim = curve["latent_dim"]
        label = f"dx={curve['dx']:.2f}, dvx={curve['dvx']:.2f}"
        for j, ch in enumerate(channels):
            axes[dim, j].plot(curve["z_values"], curve["channels"][ch], label=label)

    for dim in range(zdim):
        for j, ch in enumerate(channels):
            ax = axes[dim, j]
            ax.axvline(0.0, color="0.7", lw=1)
            ax.set_title(f"z{dim} -> {ch}")
            ax.set_xlabel(f"z{dim}")
            ax.grid(alpha=0.25)
            if dim == 0 and j == 0:
                ax.legend(fontsize=8)
    return fig, axes


def plot_latent_traversal_norms(
    traversal,
    channels=("r1_norm", "r2_norm", "rel_norm", "com_norm"),
):
    """
    Plot latent traversals for vector norms rather than individual coordinates.
    The default channels are ||r1||, ||r2||, ||r_rel||, and ||r_com||.
    """
    zdim = traversal["zdim"]
    fig, axes = plt.subplots(
        zdim,
        len(channels),
        figsize=(4.2 * len(channels), 3.2 * zdim),
        squeeze=False,
        constrained_layout=True,
    )
    for curve in traversal["curves"]:
        dim = curve["latent_dim"]
        label = f"dx={curve['dx']:.2f}, dvx={curve['dvx']:.2f}"
        for j, ch in enumerate(channels):
            axes[dim, j].plot(curve["z_values"], curve["channels"][ch], label=label)

    titles = {
        "r1_norm": "||r1||",
        "r2_norm": "||r2||",
        "rel_norm": "||r_rel||",
        "com_norm": "||r_com||",
    }
    for dim in range(zdim):
        for j, ch in enumerate(channels):
            ax = axes[dim, j]
            ax.axvline(0.0, color="0.7", lw=1)
            ax.set_title(f"z{dim} -> {titles.get(ch, ch)}")
            ax.set_xlabel(f"z{dim}")
            ax.grid(alpha=0.25)
            if dim == 0 and j == 0:
                ax.legend(fontsize=8)
    return fig, axes


def plot_latent_traversal_coordinates(traversal):
    labels = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
    zdim = traversal["zdim"]
    fig, axes = plt.subplots(
        zdim,
        6,
        figsize=(18, 3.0 * zdim),
        squeeze=False,
        constrained_layout=True,
    )
    for curve in traversal["curves"]:
        dim = curve["latent_dim"]
        label = f"dx={curve['dx']:.2f}"
        for j, lab in enumerate(labels):
            axes[dim, j].plot(curve["z_values"], curve["mu_global"][:, j], label=label)

    for dim in range(zdim):
        for j, lab in enumerate(labels):
            ax = axes[dim, j]
            ax.axvline(0.0, color="0.7", lw=1)
            ax.set_title(f"z{dim} -> {lab}")
            ax.set_xlabel(f"z{dim}")
            ax.grid(alpha=0.25)
            if dim == 0 and j == 0:
                ax.legend(fontsize=8)
    return fig, axes


def latent_summary(results):
    q_mu = results["q_mu"]
    q_std = np.exp(0.5 * results["q_logv"])
    kl_dim = 0.5 * (q_mu ** 2 + q_std ** 2 - 1.0 - np.log(q_std ** 2))
    return {
        "kl_dim_mean": kl_dim.mean(axis=0),
        "kl_dim_median": np.median(kl_dim, axis=0),
        "q_mu_mean": q_mu.mean(axis=0),
        "q_mu_std": q_mu.std(axis=0),
        "q_std_mean": q_std.mean(axis=0),
        "active_001": int(np.sum(kl_dim.mean(axis=0) > 0.01)),
        "active_005": int(np.sum(kl_dim.mean(axis=0) > 0.05)),
    }


def plot_latent_usage(results, bins=80):
    q_mu = results["q_mu"]
    q_std = np.exp(0.5 * results["q_logv"])
    zdim = q_mu.shape[1]
    fig, axes = plt.subplots(2, zdim, figsize=(4 * zdim, 6), constrained_layout=True)
    if zdim == 1:
        axes = np.array(axes).reshape(2, 1)
    for j in range(zdim):
        axes[0, j].hist(q_mu[:, j], bins=bins, density=True, histtype="step", lw=2)
        axes[0, j].set_title(f"q_mu[{j}]")
        axes[0, j].grid(alpha=0.25)
        axes[1, j].hist(q_std[:, j], bins=bins, density=True, histtype="step", lw=2)
        axes[1, j].set_title(f"q_std[{j}]")
        axes[1, j].grid(alpha=0.25)
    return fig, axes
