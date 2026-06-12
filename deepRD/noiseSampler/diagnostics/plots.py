import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import deepRD.noiseSampler.diagnostics.evaluate as evaluate
import torch

"""
General plotting utilities for CVAE training/evaluation visualization
"""


# ====================================
# === TRAINING EVAL PLOTS ===
# ====================================


def plot_losses(losses, conditionedOn):
    train_total_losses, train_nll_losses, train_kl_losses, val_total_losses, val_nll_losses, val_kl_losses, best_val_loss, best_epoch = losses.values()
    
    print('Final loss:', train_total_losses[-1])
    epochs_range = range(1, len(train_total_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_total_losses, label='Train Total Loss')
    plt.plot(epochs_range, train_nll_losses, label='Train NLL Loss', linestyle='--')
    plt.plot(epochs_range, train_kl_losses, label='Train KL Loss', linestyle='--')

    if len(val_total_losses) > 0:
        plt.plot(epochs_range, val_total_losses, 'x--', label='Validation Loss', linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training / Validation Loss {conditionedOn}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return (
        (train_total_losses, train_nll_losses, train_kl_losses),
        (val_total_losses, val_nll_losses, val_kl_losses),
        (best_val_loss, best_epoch)
    )

def visualize_latent_distributions(model, r_next, c_n, n_samples=50000, device='cpu'):
    """
    Visualize latent space distributions for encoder vs conditional prior.

    Args:
        model: trained CVAE
        c_n (np.ndarray): conditioning variables (r_n, v_n)
        r_next (np.ndarray): true target variables (r_{n+1})
        n_samples (int): number of samples to analyze
        device: torch device
    """
    N = len(c_n)
    idx = np.random.choice(N, size=min(n_samples, N), replace=False)
    c_s = c_n[idx]
    r_next_s = r_next[idx]

    # Normalize
    c_norm = torch.tensor(model.scaler_c.transform(c_s), dtype=torch.float32, device=device)
    r_next_norm = torch.tensor(model.scaler_r.transform(r_next_s), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        _, (q_mu, q_logv), _ = model(r_next_norm, c_norm)
        p_mu, p_logv = model.prior_params(c_norm)
        q_mu, q_logv = q_mu.cpu().numpy(), q_logv.cpu().numpy()
        p_mu, p_logv = p_mu.cpu().numpy(), p_logv.cpu().numpy()

    q_std = np.exp(0.5 * q_logv)
    p_std = np.exp(0.5 * p_logv)

    zdim = q_mu.shape[1]

    # --- Plot histograms of latent means ---
    fig, axes = plt.subplots(2, zdim, figsize=(4*zdim, 6))
    for i in range(zdim):
        if zdim == 1:
            ax = axes[0]
        else:
            ax = axes[0, i]
        ax.hist(q_mu[:, i], bins=50, alpha=0.6, label="Encoder μ_q", color="tab:blue")
        #ax.hist(p_mu[:, i], bins=50, alpha=0.6, label="Prior μ_p", color="tab:orange")
        ax.set_title(f"Latent dim {i}: means")
        ax.legend()

        if zdim == 1:
            ax = axes[1]
        else:
            ax = axes[1, i]
        ax.hist(q_std[:, i], bins=50, alpha=0.6, label="Encoder σ_q", color="tab:blue")
        #ax.hist(p_std[:, i], bins=50, alpha=0.6, label="Prior σ_p", color="tab:orange")
        ax.set_title(f"Latent dim {i}: stds")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_r_distributions(model, r_next, c_n,
                         n_samples=50000, device='cpu',
                         Tr=1.0, Tz=1.0):
    """
    Compare benchmark, reconstructed, and generated distributions of r_{n+1}.

    Handles both:
      - single particle: r_next shape [N, 3]
      - dimer (two particles): r_next shape [N, 6], interpreted as
            (r1_x, r1_y, r1_z, r2_x, r2_y, r2_z)

    Args:
        model: trained CVAE model with attached scalers
        c_n (np.ndarray): conditioning vectors, shape [N, c_dim]
        r_next (np.ndarray): true target auxiliary variable, shape [N, 3] or [N, 6]
        n_samples: number of samples to draw for visualization
        device: torch device
    """
    # --- Subsample for efficiency ---
    N = len(c_n)
    idx = np.random.choice(N, size=min(n_samples, N), replace=False)
    c_s = c_n[idx]
    r_next_s = r_next[idx]

    # dimensionality check
    D = r_next_s.shape[1]
    if D not in (3, 6):
        raise ValueError(f"Expected r_next to have dim 3 or 6, got {D}.")

    # number of particles (1 or 2), each with 3 dims
    n_particles = D // 3

    # --- Normalise ---
    c_norm = torch.tensor(model.scaler_c.transform(c_s), dtype=torch.float32)
    r_next_norm = torch.tensor(model.scaler_r.transform(r_next_s), dtype=torch.float32)

    # --- Forward pass (reconstruction) ---
    model.eval()
    with torch.no_grad():
        c_t = torch.tensor(c_norm, dtype=torch.float32, device=device)
        r_next_t = torch.tensor(r_next_norm, dtype=torch.float32, device=device)

        dec_out, q, p = model(r_next_t, c_t)
        mu_r, log_sig_r = dec_out
        r_rec_norm_t = mu_r + torch.exp(log_sig_r) * torch.randn_like(mu_r) * Tr
        
        # denormalizing to physical units
        r_rec = model.scaler_r.inverse_transform(r_rec_norm_t.cpu().numpy())

        # --- Generated samples (already returns physical units) ---
        r_gen_norm_t = model.sample_torch(c_t, Tr=Tr, Tz=Tz)  # shape [N, D]
        r_gen = model.scaler_r.inverse_transform(r_gen_norm_t.cpu().numpy())

    # --- Plot ---
    # rows = n_particles (1 or 2), cols = 3 (x,y,z)
    fig, axes = plt.subplots(n_particles, 3, figsize=(14, 4 * n_particles), sharey=False)

    # make axes always 2D for uniform indexing
    if n_particles == 1:
        axes = np.expand_dims(axes, axis=0)   # shape -> [1, 3]

    coord_labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for p in range(n_particles):
        for j in range(3):
            dim_idx = p * 3 + j  # 0..2 for particle 1, 3..5 for particle 2
            ax = axes[p, j]

            # KDEs for benchmark, reconstructed, generated
            kde_bench = gaussian_kde(r_next_s[:, dim_idx])
            kde_rec   = gaussian_kde(r_rec[:, dim_idx])
            kde_gen   = gaussian_kde(r_gen[:, dim_idx])

            xs = np.linspace(
                min(r_next_s[:, dim_idx].min(), r_rec[:, dim_idx].min(), r_gen[:, dim_idx].min()),
                max(r_next_s[:, dim_idx].max(), r_rec[:, dim_idx].max(), r_gen[:, dim_idx].max()),
                300,
            )

            ax.plot(xs, kde_bench(xs), label="Benchmark", color=colors[j], lw=2)
            ax.plot(xs, kde_rec(xs), "--", color="black", lw=1.5, label="Reconstructed")
            ax.plot(xs, kde_gen(xs), ":", color="red", lw=1.5, label="Generated")

            part_label = f"r{p+1}_{coord_labels[j]}" if n_particles == 2 else f"r_{coord_labels[j]}"
            ax.set_title(f"Distribution of {part_label}")
            ax.set_xlabel(part_label)
            ax.set_ylabel("Density")
            ax.legend()

            # optional fixed x-limits (keep if you like; otherwise comment out)
            ax.set_xlim(-0.05, 0.05)

    plt.tight_layout()
    plt.show()

    return r_next_s, r_rec, r_gen

# ====================================
# === MARGINAL DISTRIBUTIONS PLOTS ===
# ====================================

def plot_qvr_distributions(
    q, v, r, bins=100, density=True, xlims=None, figsize=(24, 12), label="Data"
):
    """
    Plot marginal distributions of q, v, r for one trajectory dataset.
    Parameters
    ----------
    q, v, r : array-like
        Arrays of shape (n_trajs, n_timesteps, 6).
    bins : int or dict
        Either an int for same number of bins or a dict for specific bins.
    density : bool
        Whether to normalize histograms as densities.
    xlims : dict or None
        Optional x-limits for controlled binning.
    figsize : tuple
        Figure size.
    label : str
        Legend label for the dataset.
    """
    q, v, r = map(np.asarray, (q, v, r))

    for name, arr in zip(["q", "v", "r"], [q, v, r]):
        if arr.ndim != 3 or arr.shape[-1] != 6:
            raise ValueError(f"{name} must have shape (n_trajs, n_timesteps, 6), got {arr.shape}")

    data_dict = {var: arr.reshape(-1, 6) for var, arr in zip(["q", "v", "r"], [q, v, r])}
    coord_labels = ["p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"]

    def get_bins(var):
        return bins if isinstance(bins, int) else bins[var]

    fig, axes = plt.subplots(3, 6, figsize=figsize, constrained_layout=True)

    for row, var in enumerate(["q", "v", "r"]):
        for col in range(6):
            ax = axes[row, col]
            bin_edges = np.linspace(*xlims[var][col], get_bins(var) + 1) if xlims and var in xlims else get_bins(var)

            ax.hist(data_dict[var][:, col], bins=bin_edges, density=density, histtype="stepfilled", alpha=0.45,
                    label=label if (row == 0 and col == 0) else None)
            ax.set_title(coord_labels[col], fontsize=11) if row == 0 else None
            ax.set_ylabel(var, fontsize=12) if col == 0 else None
            ax.grid(alpha=0.25)

    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=1, frameon=False)
    plt.show()
    return fig, axes

def plot_dimer_distributions(delta_x, delta_vx, cdx=None, cdv=None):
    """
    Plot KDEs of Δx (left) and axis-relative velocity Δv_x (right),
    with optional multiple comparison models.

    Args:
        delta_x : array-like, benchmark Δx samples (any shape; flattened internally)
        delta_vx: array-like, benchmark axisRelVel samples (any shape; flattened internally)
        cdx     : optional array-like, shape (nModels, nTrajs, nTimesteps) for Δx
        cdv     : optional array-like, shape (nModels, nTrajs, nTimesteps) for axisRelVel
    """

    def _to_flat_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):  # torch.Tensor
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        return x.reshape(-1)

    # --- benchmark, flattened ---
    dx_bench  = _to_flat_np(delta_x)
    dvx_bench = _to_flat_np(delta_vx)

    # --- models: list of 1D arrays per model ---
    cdx_models = []
    if cdx is not None:
        cdx_np = cdx.detach().cpu().numpy() if hasattr(cdx, "detach") else np.asarray(cdx)
        for i in range(cdx_np.shape[0]):
            cdx_models.append(cdx_np[i].reshape(-1))

    cdv_models = []
    if cdv is not None:
        cdv_np = cdv.detach().cpu().numpy() if hasattr(cdv, "detach") else np.asarray(cdv)
        for i in range(cdv_np.shape[0]):
            cdv_models.append(cdv_np[i].reshape(-1))

    # --- KDEs for benchmark ---
    kde_dx_bench  = gaussian_kde(dx_bench)
    kde_dvx_bench = gaussian_kde(dvx_bench)

    # --- grids ---
    # Δx grid bounds
    xmin = dx_bench.min()
    xmax = dx_bench.max()
    for arr in cdx_models:
        xmin = min(xmin, arr.min())
        xmax = max(xmax, arr.max())
    xs_dx = np.linspace(xmin, xmax, 100)

    # Δv grid bounds
    vmin = dvx_bench.min()
    vmax = dvx_bench.max()
    for arr in cdv_models:
        vmin = min(vmin, arr.min())
        vmax = max(vmax, arr.max())
    xs_dv = np.linspace(vmin, vmax, 100)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---- LEFT: Δx ----
    axes[0].plot(xs_dx, kde_dx_bench(xs_dx), lw=2, label="Δx (benchmark)")
    for i, arr in enumerate(cdx_models):
        kde_model = gaussian_kde(arr)
        axes[0].plot(xs_dx, kde_model(xs_dx), lw=1.5, linestyle="--", label=f"Δx ({datasetLabels[i]})")

    axes[0].set_title("Δx distribution")
    axes[0].set_xlabel("Δx")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    
    #axes[0].set_xlim([0, 2.0])

    # ---- RIGHT: axis-relative velocity ----
    axes[1].plot(xs_dv, kde_dvx_bench(xs_dv), lw=2, label="axisRelVel (benchmark)")
    for i, arr in enumerate(cdv_models):
        kde_model = gaussian_kde(arr)
        axes[1].plot(xs_dv, kde_model(xs_dv), lw=1.5, linestyle="--", label=f"axisRelVel ({datasetLabels[i]})")

    axes[1].set_title("Axis-relative velocity distribution")
    axes[1].set_xlabel("axisRelVel")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    
    #axes[1].set_xlim([-2.0, 2.0])

    plt.tight_layout()
    plt.show()


# ===========================
# ======== ACF PLOTS ========
# ===========================


def plot_acf_single(acf, lagtimesteps, title=None, xlim=None):
    # lag axis
    dt = 0.05
    lags = np.arange(lagtimesteps) * dt

    # ----- PLOTS -----
    plt.figure(figsize=(6, 4))

    # --- ACF Plot ---
    plt.plot(lags, acf, lw=2, label='ACF')
    
    plt.title("Autocorrelation Function")
    plt.xlabel("Lag [ns]")
    plt.ylabel("ACF")
    plt.grid(alpha=0.3)

    if title is not None:
        plt.title(title)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_acf(acf1, acf2, lagtimesteps, titles, xlims=None):

    dt = 0.05
    lags = np.arange(lagtimesteps) * dt

    plt.figure(figsize=(12, 5))

    # Plot for tensor1
    plt.subplot(1, 2, 1)
    plt.plot(lags, acf1, lw=2, label='ACF Tensor 1')
    plt.title(titles[0])
    plt.xlabel("Lag [ns]")
    plt.ylabel("ACF")
    plt.grid(alpha=0.3)
    if xlims is not None:
        plt.xlim(xlims[0])

    # Plot for tensor2
    plt.subplot(1, 2, 2)
    plt.plot(lags, acf2, lw=2, label='ACF Tensor 2')
    plt.title(titles[1])
    plt.xlabel("Lag [ns]")
    plt.ylabel("ACF")
    plt.grid(alpha=0.3)
    if xlims is not None:
        plt.xlim(xlims[1])

    plt.tight_layout()
    plt.show()


# ===========================
# ====== BINNING PLOTS ======
# ===========================

def plot_binned_mean_var(x_loc, r_scalar, title, nbins=30, rmin=None, rmax=None, min_count=100):
    c, mpar, mperp, vpar, vperp, counts = evaluate.get_binned_stats_local(
        x_loc, r_scalar, nbins=nbins, rmin=rmin, rmax=rmax, min_count=min_count
    )

    # Mean plot
    plt.figure(figsize=(7, 4))
    plt.plot(c.numpy(), mpar.numpy(), label="Mean(parallel)")
    plt.plot(c.numpy(), mperp.numpy(), label="Mean(perp pooled)")
    plt.title(title + " — Mean(component | bond length)")
    plt.xlabel("bond length r")
    plt.ylabel("conditional mean")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Variance plot
    plt.figure(figsize=(7, 4))
    plt.plot(c.numpy(), vpar.numpy(), label="Var(parallel)")
    plt.plot(c.numpy(), vperp.numpy(), label="Var(perp pooled)")
    plt.title(title + " — Var(component | bond length)")
    plt.xlabel("bond length r")
    plt.ylabel("conditional variance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()