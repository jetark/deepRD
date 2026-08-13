"""
Rollout diagnostics for reduced CVAE dimer simulations.

All plot functions return a matplotlib Figure (no plt.show() calls).
The main entry point is run_all_diagnostics(), which loads trajectories,
computes all statistics, saves plots to {sim_dir}/diagnostics/, and
writes a numerical summary to {sim_dir}/diagnostics/summary.txt.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance

import deepRD.tools.analysisTools as analysisTools
import deepRD.tools.trajectoryTools as trajectoryTools

BOXSIZE_DEFAULT = 5.0
BENCH_COLOR = "tab:blue"
MODEL_COLOR = "tab:orange"


# ── Geometry helpers ──────────────────────────────────────────────────────────

def minimal_image_rel(q1: torch.Tensor, q2: torch.Tensor,
                       boxsize: float = BOXSIZE_DEFAULT) -> torch.Tensor:
    """q1, q2: (..., 3). Returns q2-q1 under the periodic minimal-image convention."""
    rel = q2 - q1
    box = torch.full((3,), float(boxsize), dtype=rel.dtype, device=rel.device)
    return rel - box * torch.round(rel / box)


def _bond_unit_vec(q1: torch.Tensor, q2: torch.Tensor,
                   boxsize: float = BOXSIZE_DEFAULT,
                   eps: float = 1e-12):
    """Returns (e, length) where e is the unit bond vector q1→q2."""
    bond = minimal_image_rel(q1, q2, boxsize=boxsize)
    length = bond.norm(dim=-1, keepdim=True) + eps
    return bond / length, length[..., 0]


def _decompose(x1: torch.Tensor, x2: torch.Tensor,
               q1: torch.Tensor, q2: torch.Tensor,
               boxsize: float = BOXSIZE_DEFAULT,
               eps: float = 1e-12) -> dict:
    """
    Decompose vector pair (x1, x2) into rel/COM and bond-parallel/perp parts.
    x1, x2, q1, q2: (..., 3)
    """
    e, _ = _bond_unit_vec(q1, q2, boxsize=boxsize, eps=eps)
    rel = x2 - x1
    com = 0.5 * (x1 + x2)

    rel_par_scalar = (rel * e).sum(dim=-1)
    rel_par = rel_par_scalar.unsqueeze(-1) * e
    rel_perp = rel - rel_par

    com_par_scalar = (com * e).sum(dim=-1)
    com_par = com_par_scalar.unsqueeze(-1) * e
    com_perp = com - com_par

    return {
        "e": e,
        "rel": rel, "com": com,
        "norm_1": x1.norm(dim=-1), "norm_2": x2.norm(dim=-1),
        "rel_norm": rel.norm(dim=-1), "com_norm": com.norm(dim=-1),
        "rel_par_scalar": rel_par_scalar,
        "rel_par": rel_par,
        "rel_perp": rel_perp,
        "rel_perp_norm": rel_perp.norm(dim=-1),
        "com_par_scalar": com_par_scalar,
        "com_par": com_par,
        "com_perp": com_perp,
        "com_perp_norm": com_perp.norm(dim=-1),
    }


def compute_dx_dvx(qT: torch.Tensor, vT: torch.Tensor,
                   boxsize: float = BOXSIZE_DEFAULT):
    """
    qT, vT: (nTrajs, T, 6).
    Returns (dx, dvx) each (nTrajs, T):
        dx  = bond length |q2 - q1|
        dvx = axis-relative velocity (v2-v1)·e
    """
    q1, q2 = qT[..., :3], qT[..., 3:]
    v1, v2 = vT[..., :3], vT[..., 3:]
    e, _ = _bond_unit_vec(q1, q2, boxsize=boxsize)
    dx = minimal_image_rel(q1, q2, boxsize=boxsize).norm(dim=-1)
    dvx = ((v2 - v1) * e).sum(dim=-1)
    return dx, dvx


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_raw(ds: np.ndarray):
    """
    ds: (2*T, 10 or 11) array from loadTrajectory.
    Returns q, v, r each (T, 6) torch float32 tensors.
    """
    ds = torch.tensor(np.asarray(ds), dtype=torch.float32)
    if ds.shape[1] == 11:
        ds = torch.cat((ds[:, :7], ds[:, -3:]), dim=1)
    p1, p2 = ds[::2], ds[1::2]
    T = min(p1.shape[0], p2.shape[0])
    p1, p2 = p1[:T], p2[:T]
    q = torch.cat((p1[:, 1:4], p2[:, 1:4]), dim=-1)
    v = torch.cat((p1[:, 4:7], p2[:, 4:7]), dim=-1)
    r = torch.cat((p1[:, 7:10], p2[:, 7:10]), dim=-1)
    return q, v, r


def load_trajectories(base_path: str | Path, n_trajs: int, max_scan: int = 2500,
                      verbose: bool = True):
    """
    Load up to n_trajs trajectory files from base_path.
    Files are numbered from 0 up to max_scan.
    Returns q, v, r each (loaded_n, T, 6) float32 tensors.
    """
    base_path = str(base_path)
    q_list, v_list, r_list = [], [], []
    loaded = 0
    for fnum in range(max_scan):
        if loaded >= n_trajs:
            break
        try:
            ds = trajectoryTools.loadTrajectory(base_path, fnum)
        except (FileNotFoundError, OSError, KeyError):
            continue
        q, v, r = _parse_raw(ds)
        q_list.append(q)
        v_list.append(v)
        r_list.append(r)
        loaded += 1
        if verbose and loaded % 20 == 0:
            print(f"  {loaded}/{n_trajs} loaded", end="\r")
    if verbose:
        print(f"  Loaded {loaded} trajectories from {base_path}     ")
    if not q_list:
        raise FileNotFoundError(f"No trajectory files found at {base_path}")
    return torch.stack(q_list), torch.stack(v_list), torch.stack(r_list)


# ── ACF ───────────────────────────────────────────────────────────────────────

def _acf_1d(series: np.ndarray, trunc: int) -> np.ndarray:
    """FFT-based unnormalized ACF of a 1D series, length trunc."""
    n = len(series)
    s = series - series.mean()
    s_pad = np.concatenate([s, np.zeros(n)])
    f = np.fft.fft(s_pad)
    corr = np.fft.ifft(f * np.conj(f)).real[:trunc]
    corr /= np.linspace(n, n - trunc + 1, trunc)
    return corr


def acf_tensor(tensor, lagtimesteps: int, mTrajs: int = 50) -> np.ndarray:
    """
    Compute normalized ACF averaged over mTrajs randomly chosen trajectories.

    tensor: (nTrajs, T) or (nTrajs, T, d) — numpy array or torch tensor.
    Returns normalized ACF (lagtimesteps,).
    """
    arr = tensor.numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    nTrajs = arr.shape[0]
    m = min(mTrajs, nTrajs)
    indices = np.random.choice(nTrajs, m, replace=False)
    acf = np.zeros(lagtimesteps)
    for i in indices:
        traj = arr[i]
        if traj.ndim == 1:
            acf += _acf_1d(traj, lagtimesteps)
        else:
            for d in range(traj.shape[-1]):
                acf += _acf_1d(traj[:, d], lagtimesteps)
    norm = acf[0] if acf[0] != 0 else 1.0
    return acf / norm


# ── Numerical statistics ──────────────────────────────────────────────────────

def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def compute_summary_stats(b_q, b_v, b_r, b_dx, b_dvx,
                          q, v, r, dx, dvx, label: str = "model") -> dict:
    """
    Returns a flat dict of scalar statistics:
      mean/std for each component of v and r (6 dims each),
      mean/std for dx and dvx,
      per-step per-dim MSD for q.
    """
    stats: dict = {}
    for name, bt, mt in [("v", b_v, v), ("r", b_r, r)]:
        bn, mn = _np(bt).reshape(-1, 6), _np(mt).reshape(-1, 6)
        for d in range(6):
            stats[f"bench_{name}{d}_mean"] = float(bn[:, d].mean())
            stats[f"bench_{name}{d}_std"] = float(bn[:, d].std())
            stats[f"{label}_{name}{d}_mean"] = float(mn[:, d].mean())
            stats[f"{label}_{name}{d}_std"] = float(mn[:, d].std())

    for name, bt, mt in [("dx", b_dx, dx), ("dvx", b_dvx, dvx)]:
        bn, mn = _np(bt).reshape(-1), _np(mt).reshape(-1)
        stats[f"bench_{name}_mean"] = float(bn.mean())
        stats[f"bench_{name}_std"] = float(bn.std())
        stats[f"{label}_{name}_mean"] = float(mn.mean())
        stats[f"{label}_{name}_std"] = float(mn.std())

    # Per-step per-dim MSD: mean(|q_{t+1} - q_t|^2) for each of 6 dims
    bq_step = np.diff(_np(b_q), axis=1)   # (nTrajs, T-1, 6)
    mq_step = np.diff(_np(q), axis=1)
    for d in range(6):
        stats[f"bench_msd{d}"] = float((bq_step[..., d] ** 2).mean())
        stats[f"{label}_msd{d}"] = float((mq_step[..., d] ** 2).mean())

    return stats


def compute_wasserstein(b_q, b_v, b_r, b_dx, b_dvx,
                        q, v, r, dx, dvx) -> dict:
    """1D Wasserstein-1 distances for each variable / component."""
    wdists: dict = {}
    for name, bt, mt in [("q", b_q, q), ("v", b_v, v), ("r", b_r, r)]:
        bn = _np(bt).reshape(-1, 6)
        mn = _np(mt).reshape(-1, 6)
        for d in range(6):
            wdists[f"{name}{d}"] = wasserstein_distance(bn[:, d], mn[:, d])
    for name, bt, mt in [("dx", b_dx, dx), ("dvx", b_dvx, dvx)]:
        wdists[name] = wasserstein_distance(_np(bt).reshape(-1), _np(mt).reshape(-1))
    return wdists


# ── Plot functions ─────────────────────────────────────────────────────────────

def fig_qvr_marginals(b_q, b_v, b_r, q, v, r, label: str = "model",
                      bins: int = 100,
                      xlims: dict | None = None) -> plt.Figure:
    """3×6 grid of marginal distributions of q, v, r."""
    if xlims is None:
        xlims = {
            "q": [(-3, 3)] * 6,
            "v": [(-1, 1)] * 6,
            "r": [(-0.06, 0.06)] * 6,
        }
    data_b = {k: _np(arr).reshape(-1, 6)
              for k, arr in [("q", b_q), ("v", b_v), ("r", b_r)]}
    data_m = {k: _np(arr).reshape(-1, 6)
              for k, arr in [("q", q), ("v", v), ("r", r)]}
    coord_labels = ["p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"]

    fig, axes = plt.subplots(3, 6, figsize=(24, 9))
    for row, var in enumerate(["q", "v", "r"]):
        for col in range(6):
            ax = axes[row, col]
            lim = xlims[var][col]
            edges = np.linspace(*lim, bins + 1)
            ax.hist(data_b[var][:, col], bins=edges, density=True,
                    histtype="step", color=BENCH_COLOR,
                    label="benchmark" if (row == 0 and col == 0) else None)
            ax.hist(data_m[var][:, col], bins=edges, density=True,
                    histtype="step", color=MODEL_COLOR,
                    label=label if (row == 0 and col == 0) else None)
            if row == 0:
                ax.set_title(coord_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(var, fontsize=11)
            ax.grid(alpha=0.2)
    fig.legend(loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("q / v / r marginal distributions", fontsize=13)
    fig.tight_layout()
    return fig


def fig_dx_dvx(b_dx, b_dvx, dx, dvx, label: str = "model",
               bins: int = 150) -> plt.Figure:
    """P(Δx) and P(Δv_x) distributions."""
    b_dx_np  = _np(b_dx).reshape(-1)
    b_dvx_np = _np(b_dvx).reshape(-1)
    m_dx_np  = _np(dx).reshape(-1)
    m_dvx_np = _np(dvx).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(b_dx_np, bins=bins, density=True, histtype="step",
                 color=BENCH_COLOR, label="benchmark")
    axes[0].hist(m_dx_np, bins=bins, density=True, histtype="step",
                 color=MODEL_COLOR, label=label)
    axes[0].set_xlabel(r"$\Delta x$ (bond length)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$P(\Delta x)$")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    axes[1].hist(b_dvx_np, bins=bins, density=True, histtype="step",
                 color=BENCH_COLOR, label="benchmark")
    axes[1].hist(m_dvx_np, bins=bins, density=True, histtype="step",
                 color=MODEL_COLOR, label=label)
    axes[1].set_xlabel(r"$\Delta v_x$ (axis-relative velocity)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(r"$P(\Delta v_x)$")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def fig_acf_pair(b_acf_1, m_acf_1, b_acf_2, m_acf_2,
                 lagtimesteps: int, dt: float,
                 titles: tuple = ("Particle 1", "Particle 2"),
                 label: str = "model",
                 xlim=None) -> plt.Figure:
    """
    2-panel ACF figure. Each panel shows benchmark vs model for one quantity.
    All four acf arrays: (lagtimesteps,) numpy.

    xlim: None, a single (lo, hi) tuple applied to both panels, or a list/tuple
          of two (lo, hi) tuples for per-panel limits.
    """
    lags = np.arange(lagtimesteps) * dt
    # normalise xlim into a per-panel list
    if xlim is None:
        xlims = [None, None]
    elif isinstance(xlim[0], (int, float)):
        xlims = [xlim, xlim]
    else:
        xlims = list(xlim)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ab, am, title, xl in zip(axes,
                                      (b_acf_1, b_acf_2),
                                      (m_acf_1, m_acf_2),
                                      titles, xlims):
        ax.plot(lags, ab, lw=2, color=BENCH_COLOR, label="benchmark")
        ax.plot(lags, am, "--", lw=2, color=MODEL_COLOR, label=label)
        ax.set_title(title)
        ax.set_xlabel("Lag [ns]")
        ax.set_ylabel("ACF")
        ax.grid(alpha=0.3)
        if xl is not None:
            ax.set_xlim(xl)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_r_norms(b_r1, b_r2, r1, r2, label: str = "model",
                bins: int = 200) -> plt.Figure:
    """Distributions of ||r1|| and ||r2|| for both particles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, b_ri, m_ri, title in zip(
        axes,
        (b_r1, b_r2),
        (r1, r2),
        (r"$\|r_1\|$", r"$\|r_2\|$"),
    ):
        bn = np.linalg.norm(_np(b_ri).reshape(-1, 3), axis=-1)
        mn = np.linalg.norm(_np(m_ri).reshape(-1, 3), axis=-1)
        ax.hist(bn, bins=bins, density=True, histtype="step",
                color=BENCH_COLOR, label="benchmark")
        ax.hist(mn, bins=bins, density=True, histtype="step",
                color=MODEL_COLOR, label=label)
        ax.set_title(title)
        ax.set_xlabel("Norm")
        ax.set_ylabel("Density")
        ax.legend(); ax.grid(alpha=0.25)
    fig.suptitle("r norm distributions")
    fig.tight_layout()
    return fig


def fig_r_per_dim(b_r1, b_r2, r1, r2, label: str = "model",
                  bins: int = 200) -> plt.Figure:
    """Per-dimension distributions of r1 and r2 (2×3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    coord = ["x", "y", "z"]
    for row, (b_ri, m_ri, pname) in enumerate(
        [(b_r1, r1, "r1"), (b_r2, r2, "r2")]
    ):
        bn = _np(b_ri).reshape(-1, 3)
        mn = _np(m_ri).reshape(-1, 3)
        for col in range(3):
            ax = axes[row, col]
            ax.hist(bn[:, col], bins=bins, density=True, histtype="step",
                    color=BENCH_COLOR, label="benchmark")
            ax.hist(mn[:, col], bins=bins, density=True, histtype="step",
                    color=MODEL_COLOR, label=label)
            ax.set_title(f"{pname}_{coord[col]}")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle("r per-dimension distributions")
    fig.tight_layout()
    return fig


def fig_r_relcom_channels(b_r1, b_r2, b_q1, b_q2, r1, r2, q1, q2,
                           label: str = "model", bins: int = 100,
                           boxsize: float = BOXSIZE_DEFAULT) -> plt.Figure:
    """
    4-panel: rel-parallel, |rel-perp|, com-parallel, |com-perp|
    for the r noise variable decomposed in the bond frame.
    """
    def _flat(t):
        return torch.tensor(_np(t).reshape(-1, 3), dtype=torch.float32)

    b_dec = _decompose(_flat(b_r1), _flat(b_r2), _flat(b_q1), _flat(b_q2),
                       boxsize=boxsize)
    m_dec = _decompose(_flat(r1), _flat(r2), _flat(q1), _flat(q2),
                       boxsize=boxsize)

    panels = [
        ("rel_par_scalar", r"$r_{\mathrm{rel},\parallel}$",   None),
        ("rel_perp_norm",  r"$\|r_{\mathrm{rel},\perp}\|$", (-0.005, 0.15)),
        ("com_par_scalar", r"$r_{\mathrm{com},\parallel}$",   None),
        ("com_perp_norm",  r"$\|r_{\mathrm{com},\perp}\|$", (-0.005, 0.15)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, title, xlim) in zip(axes.ravel(), panels):
        bv = b_dec[key].numpy()
        mv = m_dec[key].numpy()
        ax.hist(bv, bins=bins, density=True, histtype="step",
                color=BENCH_COLOR, label="benchmark")
        ax.hist(mv, bins=bins, density=True, histtype="step",
                color=MODEL_COLOR, label=label)
        ax.set_title(title)
        ax.set_ylabel("Density")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(); ax.grid(alpha=0.25)
    fig.suptitle("r rel/COM channel distributions")
    fig.tight_layout()
    return fig


def fig_conditional_vs_dx(b_qT, b_vT, b_rT, b_dx,
                           qT, vT, rT, dx,
                           variable: str = "r",
                           component: str = "rel_par_scalar",
                           label: str = "model",
                           nbins: int = 40,
                           xlim: tuple = (0.1, 1.9),
                           boxsize: float = BOXSIZE_DEFAULT) -> plt.Figure:
    """
    E[component(variable_{n+1}) | Δx_n] — mean and variance panels.
    variable: 'r' or 'v'.
    component: any key in the _decompose() output dict.
    """
    def _get_var(variable_):
        if variable_ == "r":
            return (b_rT[..., :3], b_rT[..., 3:],
                    rT[..., :3],   rT[..., 3:])
        if variable_ == "v":
            return (b_vT[..., :3], b_vT[..., 3:],
                    vT[..., :3],   vT[..., 3:])
        raise ValueError(f"variable must be 'r' or 'v', got {variable_!r}")

    b_x1, b_x2, m_x1, m_x2 = _get_var(variable)
    b_q1, b_q2 = b_qT[..., :3], b_qT[..., 3:]
    m_q1, m_q2 = qT[..., :3], qT[..., 3:]

    def _pairs_flat(t):
        return t[:, :-1].reshape(-1, 3), t[:, 1:].reshape(-1, 3)

    def _scalar_flat(t):
        return t[:, :-1].reshape(-1)

    b_x1n, b_x1np1 = _pairs_flat(b_x1)
    b_x2n, b_x2np1 = _pairs_flat(b_x2)
    b_q1n, _ = _pairs_flat(b_q1)
    b_q2n, _ = _pairs_flat(b_q2)
    b_dx_n   = _scalar_flat(b_dx)

    m_x1n, m_x1np1 = _pairs_flat(m_x1)
    m_x2n, m_x2np1 = _pairs_flat(m_x2)
    m_q1n, _ = _pairs_flat(m_q1)
    m_q2n, _ = _pairs_flat(m_q2)
    m_dx_n   = _scalar_flat(dx)

    def _tf(arr):
        return torch.tensor(_np(arr), dtype=torch.float32)

    b_dec = _decompose(_tf(b_x1np1), _tf(b_x2np1),
                       _tf(b_q1n),   _tf(b_q2n), boxsize=boxsize)
    m_dec = _decompose(_tf(m_x1np1), _tf(m_x2np1),
                       _tf(m_q1n),   _tf(m_q2n), boxsize=boxsize)

    def _bin_stats(x, y, nbins_, xlim_):
        x, y = _np(x), _np(y)
        edges = np.linspace(*xlim_, nbins_ + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean = np.full(nbins_, np.nan)
        var  = np.full(nbins_, np.nan)
        for i in range(nbins_):
            mask = (x >= edges[i]) & (x < edges[i + 1])
            if mask.sum() > 20:
                yy = y[mask]
                mean[i] = yy.mean()
                var[i]  = yy.var()
        return centers, mean, var

    cb, mb, vb = _bin_stats(_np(b_dx_n), b_dec[component].numpy(), nbins, xlim)
    cm, mm, vm = _bin_stats(_np(m_dx_n), m_dec[component].numpy(), nbins, xlim)

    var_label = r"r" if variable == "r" else r"v"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(cb, mb, lw=2, color=BENCH_COLOR, label="benchmark")
    axes[0].plot(cm, mm, "--", lw=2, color=MODEL_COLOR, label=label)
    axes[0].set_xlabel(r"$\Delta x_n$")
    axes[0].set_ylabel(rf"$E[{var_label}_{{\mathrm{{rel}},\parallel}}^{{n+1}} \mid \Delta x_n]$")
    axes[0].set_title(f"Mean: {variable} rel-parallel | Δx")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(cb, vb, lw=2, color=BENCH_COLOR, label="benchmark")
    axes[1].plot(cm, vm, "--", lw=2, color=MODEL_COLOR, label=label)
    axes[1].set_xlabel(r"$\Delta x_n$")
    axes[1].set_ylabel(rf"$\mathrm{{Var}}[{var_label}_{{\mathrm{{rel}},\parallel}}^{{n+1}} \mid \Delta x_n]$")
    axes[1].set_title(f"Var: {variable} rel-parallel | Δx")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_wasserstein_bar(wdists: dict, label: str = "model") -> plt.Figure:
    """Bar chart of 1D Wasserstein distances for all variables/components."""
    keys = list(wdists.keys())
    vals = np.array([wdists[k] for k in keys])
    median = float(np.median(vals))

    colors = [MODEL_COLOR if v <= median * 3 else "tab:red" for v in vals]
    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.6), 4))
    ax.bar(range(len(keys)), vals, color=colors, alpha=0.75)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Wasserstein-1 distance")
    ax.set_title(f"Wasserstein-1 distances: benchmark vs {label}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_r_norm_vs_conditioning(
    b_rT, rT,
    b_dx,  dx,
    b_dvx, dvx,
    label: str = "model",
    nbins: int = 30,
    dx_xlim: tuple = (0.1, 1.9),
    dvx_xlim: tuple | None = None,
    rn_xlim: tuple | None = None,
) -> plt.Figure:
    """
    Conditional mean and variance of ||r_{n+1}||_6D as a function of
    three conditioning variables: Δx_n, Δv_{x,n}, ||r_n||.

    Layout: 3 rows × 2 cols (mean | variance) for each conditioning variable.
    ||r|| is the full 6D norm sqrt(||r1||² + ||r2||²).
    """
    def _bin_stats(x, y, nbins_, xlim_):
        edges = np.linspace(*xlim_, nbins_ + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean = np.full(nbins_, np.nan)
        var  = np.full(nbins_, np.nan)
        for i in range(nbins_):
            mask = (x >= edges[i]) & (x < edges[i + 1])
            if mask.sum() > 20:
                yy = y[mask]
                mean[i] = yy.mean()
                var[i]  = yy.var()
        return centers, mean, var

    # consecutive r pairs: r_n and r_{n+1}
    b_rn_flat   = _np(b_rT[:, :-1]).reshape(-1, 6)
    b_rnp1_flat = _np(b_rT[:, 1:]).reshape(-1, 6)
    m_rn_flat   = _np(rT[:, :-1]).reshape(-1, 6)
    m_rnp1_flat = _np(rT[:, 1:]).reshape(-1, 6)

    b_rn_norm   = np.linalg.norm(b_rn_flat,   axis=-1)
    b_rnp1_norm = np.linalg.norm(b_rnp1_flat, axis=-1)
    m_rn_norm   = np.linalg.norm(m_rn_flat,   axis=-1)
    m_rnp1_norm = np.linalg.norm(m_rnp1_flat, axis=-1)

    b_dx_n  = _np(b_dx[:,  :-1]).reshape(-1)
    m_dx_n  = _np(dx[:,    :-1]).reshape(-1)
    b_dvx_n = _np(b_dvx[:, :-1]).reshape(-1)
    m_dvx_n = _np(dvx[:,   :-1]).reshape(-1)

    # auto xlim defaults
    if dvx_xlim is None:
        lo = float(np.percentile(b_dvx_n, 1))
        hi = float(np.percentile(b_dvx_n, 99))
        dvx_xlim = (lo, hi)
    if rn_xlim is None:
        rn_xlim = (0.0, float(np.percentile(b_rn_norm, 99.5)))

    rows = [
        (r"$\Delta x_n$",         b_dx_n,  m_dx_n,  dx_xlim),
        (r"$\Delta v_{x,n}$",     b_dvx_n, m_dvx_n, dvx_xlim),
        (r"$\|r_n\|$",            b_rn_norm, m_rn_norm, rn_xlim),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for row_i, (xlabel, bx, mx, xlim_) in enumerate(rows):
        cb, mb, vb = _bin_stats(bx, b_rnp1_norm, nbins, xlim_)
        cm, mm, vm = _bin_stats(mx, m_rnp1_norm, nbins, xlim_)

        axes[row_i, 0].plot(cb, mb, lw=2, color=BENCH_COLOR, label="benchmark")
        axes[row_i, 0].plot(cm, mm, "--", lw=2, color=MODEL_COLOR, label=label)
        axes[row_i, 0].set_xlabel(xlabel)
        axes[row_i, 0].set_ylabel(r"$E[\|r_{n+1}\|]$")
        axes[row_i, 0].legend(); axes[row_i, 0].grid(alpha=0.3)

        axes[row_i, 1].plot(cb, vb, lw=2, color=BENCH_COLOR, label="benchmark")
        axes[row_i, 1].plot(cm, vm, "--", lw=2, color=MODEL_COLOR, label=label)
        axes[row_i, 1].set_xlabel(xlabel)
        axes[row_i, 1].set_ylabel(r"$\mathrm{Var}[\|r_{n+1}\|]$")
        axes[row_i, 1].legend(); axes[row_i, 1].grid(alpha=0.3)

    fig.suptitle(r"$\|r_{n+1}\|$ conditional statistics", fontsize=13)
    fig.tight_layout()
    return fig


# ── Summary text ──────────────────────────────────────────────────────────────

def _format_summary(stats: dict, wdists: dict, label: str) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"REDUCED DIMER DIAGNOSTICS  —  model label: {label}")
    lines.append("=" * 65)

    lines.append("\n── v / r mean and std ──────────────────────────────────────")
    coord = ["p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"]
    for var in ("v", "r"):
        lines.append(f"\n  {var}  (benchmark / {label})")
        for d, c in enumerate(coord):
            bm = stats[f"bench_{var}{d}_mean"]
            bs = stats[f"bench_{var}{d}_std"]
            mm = stats[f"{label}_{var}{d}_mean"]
            ms = stats[f"{label}_{var}{d}_std"]
            lines.append(f"    {c:6s}  bench: {bm:+.5f} ± {bs:.5f}   "
                         f"{label}: {mm:+.5f} ± {ms:.5f}")

    lines.append("\n── Δx / Δv_x ───────────────────────────────────────────────")
    for name in ("dx", "dvx"):
        bm = stats[f"bench_{name}_mean"]
        bs = stats[f"bench_{name}_std"]
        mm = stats[f"{label}_{name}_mean"]
        ms = stats[f"{label}_{name}_std"]
        lines.append(f"  {name:5s}  bench: {bm:+.5f} ± {bs:.5f}   "
                     f"{label}: {mm:+.5f} ± {ms:.5f}")

    lines.append("\n── Per-step MSD of q (× 10⁻⁵) ─────────────────────────────")
    for d, c in enumerate(coord):
        bm = stats[f"bench_msd{d}"]
        mm = stats[f"{label}_msd{d}"]
        lines.append(f"  {c:6s}  bench: {bm*1e5:.4f}   {label}: {mm*1e5:.4f}")

    lines.append("\n── Wasserstein-1 distances ─────────────────────────────────")
    for k, v in wdists.items():
        lines.append(f"  {k:8s}  {v:.6f}")

    lines.append("\n── Wasserstein summary per variable ───────────────────────")
    for var in ("q", "v", "r"):
        ws = [wdists[f"{var}{d}"] for d in range(6)]
        lines.append(f"  {var}  mean={np.mean(ws):.6f}  max={np.max(ws):.6f}")
    lines.append(f"  dx       {wdists['dx']:.6f}")
    lines.append(f"  dvx      {wdists['dvx']:.6f}")
    lines.append("")
    return "\n".join(lines)


# ── FPT (first passage time) diagnostics ────────────────────────────────────────

def load_fpt(path: str | Path) -> np.ndarray:
    """Load first-passage times (one value per line) from a .xyz file."""
    return np.loadtxt(str(path)).reshape(-1)


# ── Core-set (entry-to-arrival) first-passage estimator ───────────────────────
# Project-standard FPT method: apply the core-set / committor crossing analysis
# directly to a long *equilibrium* rollout, instead of the cold-start
# ``propagateFPT`` reset-and-run estimator (which suffers a dead-time /
# inspection-paradox bias — see tests/benchmarkFPTs/RESULTS.md, "Update
# 2026-07-09"). ``entry_to_arrival_fpt`` below is byte-for-byte the reference
# implementation from tests/benchmarkFPTs/compute_true_fpt.py, promoted here so
# the canonical scoreboard and the benchmark reference share one estimator.
FPT_X0 = 0.5        # closed-state (first minima) separation
FPT_RAD = 0.5       # half distance between minima; open state at X0 + 2*RAD = 1.5
FPT_MARGIN = 0.3    # core-set margin: closed core dx < X0+margin, open core dx > X0+2*RAD-margin


def entry_to_arrival_fpt(dx: np.ndarray, lo: float, hi: float,
                         direction: str, dt_out: float) -> np.ndarray:
    """
    FPT from the moment the trajectory commits to core A (first crossing in,
    following residence in core B) until it first arrives in core B.

    direction='CO': A = closed core (dx < lo), B = open core (dx > hi)
    direction='OC': A = open core   (dx > hi), B = closed core (dx < lo)

    ``dx`` is a (n_trajs, T) array; ``dt_out`` the per-output-step time in ns.
    """
    dx = _np(dx)
    n, T = dx.shape
    tarr = np.arange(T)[None, :]
    if direction == "CO":
        raw_in_A = dx < lo
        raw_in_B = dx > hi
    else:
        raw_in_A = dx > hi
        raw_in_B = dx < lo

    signal = np.where(raw_in_A, 0, np.where(raw_in_B, 1, -1))
    idx = np.where(signal != -1, tarr, 0)
    idx = np.maximum.accumulate(idx, axis=1)
    rows = np.arange(n)[:, None]
    ffilled_state = signal[rows, idx]
    prev_state = np.concatenate([np.full((n, 1), -1), ffilled_state[:, :-1]], axis=1)

    # genuine new commitments to A (excludes mid-stretch flicker)
    new_commit_A = raw_in_A & (prev_state != 0)

    # next index (>= t) where raw_in_B is true, else sentinel T
    rev_idx = np.where(raw_in_B, tarr, T)
    next_B = np.minimum.accumulate(rev_idx[:, ::-1], axis=1)[:, ::-1]

    t_begin = np.broadcast_to(tarr, dx.shape)[new_commit_A]
    t_end = next_B[new_commit_A]
    valid = t_end < T
    return (t_end[valid] - t_begin[valid]) * dt_out


def reflected_kde(data: np.ndarray, grid: np.ndarray, bw_method=None) -> np.ndarray:
    """
    Boundary-corrected Gaussian KDE on the positive half-line. First-passage
    times are strictly positive and pile up near t=0, where a plain KDE leaks
    below zero and under-estimates the near-origin density. Reflecting the sample
    at t=0 and doubling on t>=0 removes that leakage → a smoother, more faithful
    curve. ``bw_method`` is forwarded to gaussian_kde for finer/consistent
    bandwidth control.
    """
    data = np.asarray(data).reshape(-1)
    data = data[data > 0]
    kde = gaussian_kde(np.concatenate([data, -data]), bw_method=bw_method)
    dens = 2.0 * kde(grid)
    dens[grid < 0] = 0.0
    return dens


def other_transition_fpt_dir(fpt_sim_dir: str | Path) -> tuple[str, str, Path]:
    """
    Given an FPT sim directory for one transition type (name matching
    FPT_{OC|CO}_<nsims>x<tfinal>[_<tag>], as produced by
    benchmarkFPTreducedDimerCVAEGen.py / benchmarkFPTreducedDimerE3Gen.py),
    return (this_type, other_type, sibling_dir) where sibling_dir is the
    directory with the same name but the other transition type, in the same
    parent directory. Existence of sibling_dir is not checked here.
    """
    fpt_sim_dir = Path(fpt_sim_dir)
    m = re.match(r"^(FPT_)(OC|CO)(_.*)$", fpt_sim_dir.name)
    if not m:
        raise ValueError(
            f"Cannot determine transition type from FPT sim-dir name "
            f"{fpt_sim_dir.name!r}; expected the pattern "
            f"FPT_{{OC|CO}}_<nsims>x<tfinal>[_<tag>]."
        )
    prefix, this_type, suffix = m.groups()
    other_type = "CO" if this_type == "OC" else "OC"
    other_dir = fpt_sim_dir.parent / f"{prefix}{other_type}{suffix}"
    return this_type, other_type, other_dir


def find_fpt_file(directory: str | Path, transition_type: str | None = None) -> Path:
    """
    Locate a simMoriZwanzigFPTs_*.xyz file inside directory.
    If transition_type is given, restrict to simMoriZwanzigFPTs_<type>_*.xyz.
    If several matches remain, the one with the largest nsims is picked.
    """
    directory = Path(directory)
    pattern = (f"simMoriZwanzigFPTs_{transition_type}_*.xyz"
               if transition_type else "simMoriZwanzigFPTs_*.xyz")
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No FPT file matching '{pattern}' found in {directory}")
    if len(matches) > 1:
        def _nsims(p: Path) -> int:
            m = re.search(r"nsims(\d+)", p.stem)
            return int(m.group(1)) if m else 0
        matches.sort(key=_nsims, reverse=True)
    return matches[0]


def _fpt_unavailable_note(ax, ttype: str, label: str) -> None:
    ax.text(0.5, 0.92, f"{label} FPT data not available for {ttype}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="tab:red",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="tab:red"))


def fig_fpt_distributions_grid(rows: list[dict], label: str = "model",
                               bins: int = 80, bw_method=None) -> plt.Figure:
    """
    Histogram and KDE comparison of FPT distributions, one row per transition type.

    rows: list of dicts, each with keys 'transition_type', 'b_fpt', 'm_fpt'
          (None if unavailable), 'available' (bool).

    The left (histogram) panel uses a finer binning (``bins`` default 80) for a
    more precise view of the distribution shape; the right panel uses a
    boundary-corrected reflected KDE (see :func:`reflected_kde`) so the estimate
    is consistent near t=0 instead of leaking below zero.
    """
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        ttype, b_fpt, m_fpt, available = (
            row["transition_type"], row["b_fpt"], row["m_fpt"], row["available"]
        )
        ax_hist, ax_kde = axes[i]

        tmax = float(max(b_fpt.max(), m_fpt.max()) if available else b_fpt.max())
        edges = np.linspace(0.0, tmax, bins + 1)
        ax_hist.hist(b_fpt, bins=edges, density=True, histtype="step",
                     color=BENCH_COLOR, label="benchmark")
        if available:
            ax_hist.hist(m_fpt, bins=edges, density=True, histtype="step",
                         color=MODEL_COLOR, label=label)
        ax_hist.set_xlim(0, tmax)
        ax_hist.set_xlabel("FPT [ns]")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"{ttype}: FPT histogram")
        ax_hist.legend(); ax_hist.grid(alpha=0.25)

        tgrid = np.linspace(0.0, tmax, 512)
        ax_kde.plot(tgrid, reflected_kde(b_fpt, tgrid, bw_method),
                    lw=2, color=BENCH_COLOR, label="benchmark")
        if available:
            ax_kde.plot(tgrid, reflected_kde(m_fpt, tgrid, bw_method),
                        "--", lw=2, color=MODEL_COLOR, label=label)
        ax_kde.set_xlim(0, tmax)
        ax_kde.set_xlabel("FPT [ns]")
        ax_kde.set_ylabel("Density")
        ax_kde.set_title(f"{ttype}: FPT KDE (reflected)")
        ax_kde.legend(); ax_kde.grid(alpha=0.25)

        if not available:
            _fpt_unavailable_note(ax_hist, ttype, label)
            _fpt_unavailable_note(ax_kde, ttype, label)

    fig.suptitle("First passage time distributions (OC / CO)", fontsize=13)
    fig.tight_layout()
    return fig


def fig_fpt_survival_grid(rows: list[dict], label: str = "model") -> plt.Figure:
    """Survival probability P(T > t), semilog-y, one row per transition type."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.5, 4.5 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        ttype, b_fpt, m_fpt, available = (
            row["transition_type"], row["b_fpt"], row["m_fpt"], row["available"]
        )
        ax = axes[i, 0]
        curves = [(b_fpt, BENCH_COLOR, "benchmark")]
        if available:
            curves.append((m_fpt, MODEL_COLOR, label))
        for fpt, color, lbl in curves:
            t_sorted = np.sort(fpt)
            survival = 1.0 - np.arange(1, len(t_sorted) + 1) / len(t_sorted)
            ax.step(t_sorted, survival, where="post", lw=2, color=color, label=lbl)
        ax.set_yscale("log")
        ax.set_xlabel("FPT [ns]")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"{ttype}: FPT survival curve")
        ax.legend(); ax.grid(alpha=0.3, which="both")

        if not available:
            _fpt_unavailable_note(ax, ttype, label)

    fig.suptitle("First passage time survival curves (OC / CO)", fontsize=13)
    fig.tight_layout()
    return fig


def compute_fpt_stats(b_fpt, m_fpt, label: str = "model") -> dict:
    """Scalar FPT summary stats: mean/std/median, Wasserstein-1, KS test."""
    ks_stat, ks_pvalue = ks_2samp(b_fpt, m_fpt)
    return {
        "bench_n": int(len(b_fpt)),
        f"{label}_n": int(len(m_fpt)),
        "bench_mean": float(np.mean(b_fpt)),
        f"{label}_mean": float(np.mean(m_fpt)),
        "bench_std": float(np.std(b_fpt)),
        f"{label}_std": float(np.std(m_fpt)),
        "bench_median": float(np.median(b_fpt)),
        f"{label}_median": float(np.median(m_fpt)),
        "wasserstein": float(wasserstein_distance(b_fpt, m_fpt)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def _format_fpt_summary_combined(rows: list[dict], label: str) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"FIRST PASSAGE TIME DIAGNOSTICS (OC & CO)  —  model label: {label}")
    lines.append("=" * 65)
    for row in rows:
        ttype = row["transition_type"]
        lines.append(f"\n── {ttype} " + "─" * (58 - len(ttype)))
        if not row["available"]:
            b_fpt = row["b_fpt"]
            lines.append(f"  {label} FPT data NOT AVAILABLE for {ttype} "
                         f"(expected: {row['m_dir_expected']})")
            lines.append(f"  N samples    bench: {len(b_fpt):6d}")
            lines.append(f"  Mean  [ns]   bench: {np.mean(b_fpt):9.4f}")
            lines.append(f"  Std   [ns]   bench: {np.std(b_fpt):9.4f}")
            lines.append(f"  Median[ns]   bench: {np.median(b_fpt):9.4f}")
            continue
        stats = row["stats"]
        lines.append(f"  N samples    bench: {stats['bench_n']:6d}   {label}: {stats[f'{label}_n']:6d}")
        lines.append(f"  Mean  [ns]   bench: {stats['bench_mean']:9.4f}   {label}: {stats[f'{label}_mean']:9.4f}")
        lines.append(f"  Std   [ns]   bench: {stats['bench_std']:9.4f}   {label}: {stats[f'{label}_std']:9.4f}")
        lines.append(f"  Median[ns]   bench: {stats['bench_median']:9.4f}   {label}: {stats[f'{label}_median']:9.4f}")
        lines.append(f"  Wasserstein-1 distance: {stats['wasserstein']:.4f}")
        lines.append(f"  KS statistic: {stats['ks_stat']:.4f}  (p-value: {stats['ks_pvalue']:.4g})")
    lines.append("")
    return "\n".join(lines)


def run_fpt_diagnostics(
    fpt_sim_dir: str | Path,
    fpt_bench_dir: str | Path,
    label: str | None = None,
    diag_dir: str | Path | None = None,
) -> None:
    """
    Compare model vs benchmark first-passage-time distributions for BOTH
    transition types (OC and CO) in one combined set of plots + summary.

    fpt_sim_dir   : simulation-output directory for ONE transition type
                     (name matching FPT_{OC|CO}_<nsims>x<tfinal>[_<tag>], as
                     produced by benchmarkFPTreducedDimerCVAEGen.py /
                     benchmarkFPTreducedDimerE3Gen.py). The sibling directory
                     for the other transition type -- same parent, same name
                     with OC/CO swapped -- is auto-discovered alongside it.
                     If that sibling (or its FPT file) doesn't exist, the
                     corresponding row falls back to benchmark-only, noted
                     in the plot and the summary.
    fpt_bench_dir : directory with reference simMoriZwanzigFPTs_{OC,CO}_*.xyz files
    diag_dir      : explicit output directory; defaults to {fpt_sim_dir}/diagnostics/
    """
    fpt_sim_dir = Path(fpt_sim_dir)
    fpt_bench_dir = Path(fpt_bench_dir)
    if label is None:
        label = fpt_sim_dir.name

    diag_dir = Path(diag_dir) if diag_dir is not None else fpt_sim_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    this_type, other_type, other_dir = other_transition_fpt_dir(fpt_sim_dir)
    sim_dir_by_type = {this_type: fpt_sim_dir, other_type: other_dir}

    rows = []
    for ttype in ("OC", "CO"):
        sdir = sim_dir_by_type[ttype]
        b_file = find_fpt_file(fpt_bench_dir, ttype)
        b_fpt = load_fpt(b_file)
        print(f"[{ttype}] Loading benchmark FPTs from: {b_file}  ({len(b_fpt)} samples)")

        m_fpt, available = None, False
        if sdir.exists():
            try:
                m_file = find_fpt_file(sdir, ttype)
                m_fpt = load_fpt(m_file)
                available = True
                print(f"[{ttype}] Loading model FPTs from:     {m_file}  ({len(m_fpt)} samples)")
            except FileNotFoundError:
                print(f"[{ttype}] Warning: no FPT file found in {sdir}; using benchmark only")
        else:
            print(f"[{ttype}] Note: sibling directory not found ({sdir}); using benchmark only")

        row = {
            "transition_type": ttype,
            "b_fpt": b_fpt,
            "m_fpt": m_fpt,
            "available": available,
            "m_dir_expected": str(sdir),
        }
        if available:
            row["stats"] = compute_fpt_stats(b_fpt, m_fpt, label=label)
        rows.append(row)

    def _save(fig, name: str):
        path = diag_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("Generating FPT plots...")
    # Plot 14 uses the shared publication-styled builder (same as the panels in
    # the combined plot 16); deferred import avoids a circular dependency.
    from deepRD.noiseSampler.diagnostics import plots_dimer
    _order = {"CO": 0, "OC": 1}
    rows_pub = sorted(rows, key=lambda r: _order.get(r["transition_type"], 9))
    _save(plots_dimer.fig_fpt(rows_pub, label=label), "14_fpt_distributions")
    _save(fig_fpt_survival_grid(rows, label=label), "15_fpt_survival")

    summary_text = _format_fpt_summary_combined(rows, label=label)
    (diag_dir / "fpt_summary.txt").write_text(summary_text)

    print(f"\nFPT diagnostics written to: {diag_dir}")
    print(summary_text)

    return rows


def run_fpt_diagnostics_coreset(
    model_dx,
    dt: float,
    fpt_bench_dir: str | Path,
    label: str | None = None,
    diag_dir: str | Path | None = None,
    margin: float = FPT_MARGIN,
) -> list[dict]:
    """
    Core-set (entry-to-arrival) FPT diagnostics computed DIRECTLY from a model's
    equilibrium rollout Δx, compared against the correctly-labeled benchmark
    reference. This is the project-standard, cold-start-free method (see
    :func:`entry_to_arrival_fpt`); it needs no separate ``propagateFPT`` run.

    model_dx      : (n_trajs, T) rollout Δx array (np or torch); e.g. the ``"dx"``
                    entry of the dict returned by :func:`run_all_diagnostics`.
    dt            : per-output-step time in ns (the ``"dt"`` entry of that dict).
    fpt_bench_dir : directory with reference simMoriZwanzigFPTs_{OC,CO}_*.xyz
                    (benchmarkFPTreference/, itself core-set — see
                    tests/benchmarkFPTs/compute_true_fpt.py).
    diag_dir      : output directory for 14/15 + fpt_summary.txt.
    margin        : core-set margin (default 0.3).

    Produces the SAME plots (14_fpt_distributions, 15_fpt_survival) and
    fpt_summary.txt as :func:`run_fpt_diagnostics`, and returns the same ``rows``
    structure (usable by :func:`save_combined_key_diagnostics` for plot 16).
    """
    model_dx = _np(model_dx)
    fpt_bench_dir = Path(fpt_bench_dir)
    if label is None:
        label = "model"
    if diag_dir is not None:
        diag_dir = Path(diag_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)

    lo = FPT_X0 + margin
    hi = FPT_X0 + 2 * FPT_RAD - margin

    rows = []
    for ttype in ("OC", "CO"):
        b_file = find_fpt_file(fpt_bench_dir, ttype)
        b_fpt = load_fpt(b_file)
        m_fpt = entry_to_arrival_fpt(model_dx, lo, hi, ttype, dt)
        print(f"[{ttype}] benchmark FPTs: {b_file}  ({len(b_fpt)} samples)  |  "
              f"{label} core-set FPTs from rollout Δx  ({len(m_fpt)} samples)")
        rows.append({
            "transition_type": ttype,
            "b_fpt": b_fpt,
            "m_fpt": m_fpt,
            "available": True,
            "m_dir_expected": "core-set(rollout Δx)",
            "stats": compute_fpt_stats(b_fpt, m_fpt, label=label),
        })

    if diag_dir is not None:
        def _save(fig, name: str):
            fig.savefig(diag_dir / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        from deepRD.noiseSampler.diagnostics import plots_dimer
        _order = {"CO": 0, "OC": 1}
        rows_pub = sorted(rows, key=lambda r: _order.get(r["transition_type"], 9))
        _save(plots_dimer.fig_fpt(rows_pub, label=label), "14_fpt_distributions")
        _save(fig_fpt_survival_grid(rows, label=label), "15_fpt_survival")

        summary_text = _format_fpt_summary_combined(rows, label=label)
        (diag_dir / "fpt_summary.txt").write_text(summary_text)
        print(f"\nCore-set FPT diagnostics written to: {diag_dir}")
        print(summary_text)

    return rows


def save_combined_key_diagnostics(roll: dict, fpt_rows: list[dict],
                                  diag_dir: str | Path,
                                  label: str | None = None) -> None:
    """
    Build and save the combined "key diagnostics" figure (plot 16): the Δx/Δv_x
    distributions (02), Δx/Δv_x ACFs (03) and FPT distributions (14) on one
    canvas. ``roll`` is the dict returned by :func:`run_all_diagnostics`;
    ``fpt_rows`` the list returned by :func:`run_fpt_diagnostics`.
    """
    # Deferred import avoids a circular dependency (plots_dimer imports this module).
    from deepRD.noiseSampler.diagnostics import plots_dimer

    diag_dir = Path(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    label = label or roll.get("label", "model")

    # order FPT rows CO, OC for a stable layout
    order = {"CO": 0, "OC": 1}
    fpt_rows = sorted(fpt_rows, key=lambda r: order.get(r["transition_type"], 9))

    fig = plots_dimer.fig_key_diagnostics(roll, fpt_rows, label=label)
    path = diag_dir / "16_key_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined key-diagnostics figure written to: {path}")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_all_diagnostics(
    sim_dir: str | Path,
    bench_dir: str | Path,
    n_trajs: int = 100,
    n_bench_trajs: int = 100,
    mTrajs_acf: int = 50,
    label: str | None = None,
    boxsize: float = BOXSIZE_DEFAULT,
    dt: float | None = None,
    diag_dir: str | Path | None = None,
) -> None:
    """
    Run the full suite of rollout diagnostics, saving plots and a text summary
    to diag_dir (default: {sim_dir}/diagnostics/).

    Parameters
    ----------
    sim_dir      : directory containing simMoriZwanzigReduced_*.h5 and parameters
    bench_dir    : directory containing simMoriZwanzig_*.h5 benchmark files
    n_trajs      : number of reduced trajectories to load
    n_bench_trajs: number of benchmark trajectories to load
    mTrajs_acf   : number of trajectories used in ACF estimation
    label        : short model label for plot legends (default: sim_dir.name)
    boxsize      : periodic box side length
    dt           : output time step in ns; if None, read from parameters file
    diag_dir     : explicit output directory; defaults to {sim_dir}/diagnostics/
    """
    sim_dir   = Path(sim_dir)
    bench_dir = Path(bench_dir)
    if label is None:
        label = sim_dir.name

    diag_dir = Path(diag_dir) if diag_dir is not None else sim_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ── read dt from parameters ──────────────────────────────────────────────
    if dt is None:
        try:
            params = analysisTools.readParameters(str(sim_dir / "parameters"))
            dt = float(params["dt"]) * int(float(params.get("stride", 1)))
        except Exception:
            dt = 0.05
            print(f"  Warning: could not read dt from parameters, using dt={dt}")

    # ── load trajectories ─────────────────────────────────────────────────────
    print("Loading benchmark trajectories...")
    b_q, b_v, b_r = load_trajectories(
        bench_dir / "simMoriZwanzig_", n_bench_trajs
    )
    print("Loading reduced model trajectories...")
    q, v, r = load_trajectories(
        sim_dir / "simMoriZwanzigReduced_", n_trajs
    )

    T_bench = b_q.shape[1]
    T_model = q.shape[1]
    print(f"  Benchmark: {b_q.shape[0]} trajs × {T_bench} steps   "
          f"| Model: {q.shape[0]} trajs × {T_model} steps")
    print(f"  dt_output = {dt} ns")

    # ── derived quantities ────────────────────────────────────────────────────
    b_dx, b_dvx = compute_dx_dvx(b_q, b_v, boxsize=boxsize)
    dx,   dvx   = compute_dx_dvx(q,   v,   boxsize=boxsize)

    b_r1, b_r2 = b_r[..., :3], b_r[..., 3:]
    b_q1, b_q2 = b_q[..., :3], b_q[..., 3:]
    b_v1, b_v2 = b_v[..., :3], b_v[..., 3:]
    r1,   r2   = r[..., :3],   r[..., 3:]
    q1,   q2   = q[..., :3],   q[..., 3:]
    v1,   v2   = v[..., :3],   v[..., 3:]

    # ── ACF lag budgets ───────────────────────────────────────────────────────
    T_min = min(T_bench, T_model)
    lag_dx  = min(2000, T_min // 5)
    lag_r   = min(100,  T_min // 20)
    lag_v   = min(500,  T_min // 10)
    lag_q   = min(2000, T_min // 5)

    print("Computing ACFs...")
    b_acf_dx  = acf_tensor(b_dx,  lag_dx,  mTrajs_acf)
    m_acf_dx  = acf_tensor(dx,    lag_dx,  mTrajs_acf)
    b_acf_dvx = acf_tensor(b_dvx, lag_dx,  mTrajs_acf)
    m_acf_dvx = acf_tensor(dvx,   lag_dx,  mTrajs_acf)

    b_acf_r1  = acf_tensor(b_r1,  lag_r,   mTrajs_acf)
    m_acf_r1  = acf_tensor(r1,    lag_r,   mTrajs_acf)
    b_acf_r2  = acf_tensor(b_r2,  lag_r,   mTrajs_acf)
    m_acf_r2  = acf_tensor(r2,    lag_r,   mTrajs_acf)

    b_acf_v1  = acf_tensor(b_v1,  lag_v,   mTrajs_acf)
    m_acf_v1  = acf_tensor(v1,    lag_v,   mTrajs_acf)
    b_acf_v2  = acf_tensor(b_v2,  lag_v,   mTrajs_acf)
    m_acf_v2  = acf_tensor(v2,    lag_v,   mTrajs_acf)

    b_acf_q1  = acf_tensor(b_q1,  lag_q,   mTrajs_acf)
    m_acf_q1  = acf_tensor(q1,    lag_q,   mTrajs_acf)
    b_acf_q2  = acf_tensor(b_q2,  lag_q,   mTrajs_acf)
    m_acf_q2  = acf_tensor(q2,    lag_q,   mTrajs_acf)

    print("Computing statistics and Wasserstein distances...")
    stats  = compute_summary_stats(b_q, b_v, b_r, b_dx, b_dvx,
                                   q, v, r, dx, dvx, label=label)
    wdists = compute_wasserstein(b_q, b_v, b_r, b_dx, b_dvx,
                                 q, v, r, dx, dvx)

    # ── save plots ────────────────────────────────────────────────────────────
    def _save(fig, name: str):
        path = diag_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("Generating plots...")

    # Plots 02 and 03 use the shared publication-styled builders (identical to
    # the panels in the combined plot 16); deferred import avoids a circular
    # dependency (plots_dimer imports this module).
    from deepRD.noiseSampler.diagnostics import plots_dimer

    _save(fig_qvr_marginals(b_q, b_v, b_r, q, v, r, label=label),
          "01_qvr_marginals")

    _save(plots_dimer.fig_dx_dvx(b_dx, b_dvx, dx, dvx, label=label),
          "02_dx_dvx_distributions")

    # dvx ACF: show up to 10 ns; dx ACF: unrestricted
    lags_dx = np.arange(lag_dx) * dt
    xlim_dvx = (0.0, min(10.0, lag_dx * dt))
    _save(plots_dimer.fig_acf_dx_dvx(lags_dx, b_acf_dx, m_acf_dx,
                                     b_acf_dvx, m_acf_dvx, label=label,
                                     dvx_xlim=xlim_dvx),
          "03_acf_dx_dvx")

    # r ACF: show up to 2 ns
    xlim_r = (0.0, min(2.0, lag_r * dt))
    _save(fig_acf_pair(b_acf_r1, m_acf_r1, b_acf_r2, m_acf_r2,
                       lag_r, dt, titles=("ACF: $r_1$", "ACF: $r_2$"),
                       label=label, xlim=xlim_r),
          "04_acf_r")

    _save(fig_acf_pair(b_acf_v1, m_acf_v1, b_acf_v2, m_acf_v2,
                       lag_v, dt, titles=("ACF: $v_1$", "ACF: $v_2$"),
                       label=label, xlim=(0.0, lag_v * dt * 0.8)),
          "05_acf_v")

    _save(fig_acf_pair(b_acf_q1, m_acf_q1, b_acf_q2, m_acf_q2,
                       lag_q, dt, titles=("ACF: $q_1$", "ACF: $q_2$"),
                       label=label, xlim=(0.0, lag_q * dt * 0.8)),
          "06_acf_q")

    _save(fig_r_norms(b_r1, b_r2, r1, r2, label=label),
          "07_r_norms")

    _save(fig_r_per_dim(b_r1, b_r2, r1, r2, label=label),
          "08_r_per_dim")

    _save(fig_r_relcom_channels(b_r1, b_r2, b_q1, b_q2,
                                r1, r2, q1, q2, label=label, boxsize=boxsize),
          "09_r_relcom_channels")

    _save(fig_conditional_vs_dx(b_q, b_v, b_r, b_dx, q, v, r, dx,
                                variable="r", label=label, boxsize=boxsize),
          "10_conditional_r_vs_dx")

    _save(fig_conditional_vs_dx(b_q, b_v, b_r, b_dx, q, v, r, dx,
                                variable="v", label=label, boxsize=boxsize),
          "11_conditional_v_vs_dx")

    _save(fig_wasserstein_bar(wdists, label=label),
          "12_wasserstein_distances")

    _save(fig_r_norm_vs_conditioning(b_r, r, b_dx, dx, b_dvx, dvx, label=label),
          "13_r_norm_vs_conditioning")

    # ── save text summary ─────────────────────────────────────────────────────
    summary_text = _format_summary(stats, wdists, label=label)
    summary_path = diag_dir / "summary.txt"
    summary_path.write_text(summary_text)

    print(f"\nDiagnostics written to: {diag_dir}")
    print(summary_text)

    # Arrays needed to build the combined "key diagnostics" figure (plot 16).
    return {
        "dt": dt,
        "lags": np.arange(lag_dx) * dt,
        "b_dx": b_dx, "b_dvx": b_dvx, "dx": dx, "dvx": dvx,
        "b_acf_dx": b_acf_dx, "m_acf_dx": m_acf_dx,
        "b_acf_dvx": b_acf_dvx, "m_acf_dvx": m_acf_dvx,
        "label": label,
    }
