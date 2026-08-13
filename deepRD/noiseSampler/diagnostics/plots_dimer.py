"""
Publication-quality plotting for the reduced-dimer noise-sampler diagnostics.

This module hosts the *presentation* layer for the three diagnostics that matter
most when comparing a reduced CVAE model against the benchmark:

    * plot 02 — P(Δx) and P(Δv_x) marginal distributions
    * plot 03 — Δx and Δv_x autocorrelation functions
    * plot 14 — first-passage-time (FPT) distributions (CO / OC)

Each diagnostic is exposed at two levels:

    draw_*(ax, ...)   draw onto caller-supplied Axes (composable — used to build
                      the combined figure and to place several panels together)
    fig_*(...)        convenience wrapper that creates a Figure and returns it

The heavy lifting (trajectory loading, Δx/Δv_x extraction, ACF estimation, FPT
file discovery) is reused from ``rollout_diags`` via the ``prepare_*`` helpers so
a notebook can go from a run directory to a finished figure with a couple of
fully-parameterised calls.

Styling is deliberately restrained and consistent: the benchmark is a dark
reference (subtle fill + crisp outline), the model a single accent colour. Call
:func:`use_pub_style` once (the ``fig_*`` wrappers do it for you) to apply the
shared rcParams.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance

from deepRD.noiseSampler.diagnostics.rollout_diags import (
    acf_tensor,
    compute_dx_dvx,
    find_fpt_file,
    load_fpt,
    load_trajectories,
    other_transition_fpt_dir,
)
import deepRD.tools.analysisTools as analysisTools

# ── Shared palette / style ──────────────────────────────────────────────────────
BENCH_COLOR = "#1a1a1a"   # near-black reference
BENCH_FILL  = "#1a1a1a"   # fill colour (used at low alpha)
MODEL_COLOR = "#c0392b"   # accent (crimson) — primary model
MODEL2_COLOR = "#2c7fb8"  # accent (blue)    — second model in overlays
# Default cycle for additional models passed via ``extras`` without an explicit colour.
EXTRA_COLORS = ["#2c7fb8", "#2ca25f", "#e6a817", "#8856a7"]
GRID_COLOR  = "#b8b8b8"

_PUB_RC = {
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.titleweight":  "semibold",
    "axes.labelsize":    12.5,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.18,
    "grid.linewidth":    0.7,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "legend.frameon":    False,
    "lines.linewidth":   2.0,
    "lines.solid_capstyle": "round",
    "mathtext.default":  "regular",
}


def use_pub_style() -> None:
    """Apply the shared publication rcParams (idempotent)."""
    plt.rcParams.update(_PUB_RC)


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


# ── FPT boundary-corrected KDE ──────────────────────────────────────────────────

def reflected_kde(data: np.ndarray, grid: np.ndarray, bw_method=None) -> np.ndarray:
    """
    Boundary-corrected Gaussian KDE on the positive half-line.

    First-passage times are strictly positive and pile up near t=0, where an
    ordinary KDE leaks probability below zero and biases the near-origin density
    downward. Reflecting the sample at t=0 and doubling the estimate on t>=0
    removes that leakage, giving a smoother, more faithful density than the raw
    ``gaussian_kde``. ``bw_method`` is forwarded to ``gaussian_kde`` so the
    bandwidth can be made finer/consistent across panels if desired.
    """
    data = np.asarray(data).reshape(-1)
    data = data[data > 0]
    kde = gaussian_kde(np.concatenate([data, -data]), bw_method=bw_method)
    dens = 2.0 * kde(grid)
    dens[grid < 0] = 0.0
    return dens


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 02 — Δx / Δv_x marginal distributions
# ═════════════════════════════════════════════════════════════════════════════

def _norm_extras(extras):
    """Normalise an ``extras`` overlay list, filling in default colours/linestyle.

    Each item is a dict describing an additional model to overlay; the data key
    it must carry depends on the plot (``m`` for a distribution, ``acf`` for an
    ACF, ``m_fpt`` for an FPT). Returns ``[]`` if ``extras`` is falsy.
    """
    out = []
    for i, e in enumerate(extras or []):
        e = dict(e)
        e.setdefault("color", EXTRA_COLORS[i % len(EXTRA_COLORS)])
        e.setdefault("ls", "--")
        e.setdefault("label", f"model {i + 2}")
        out.append(e)
    return out


def draw_dist(ax, b, m, label="model", *, bins=120, xlim=None,
              model_color=MODEL_COLOR, xlabel="", show_legend=True,
              extras=None) -> None:
    """Draw a benchmark-vs-model 1-D density comparison onto ``ax``.

    Benchmark: subtle filled area + crisp outline. Model: single accent line.
    ``extras`` overlays further models, each a dict ``{"m", "label", "color",
    "ls"}`` (colour/ls optional).
    """
    b = _np(b).reshape(-1)
    m = _np(m).reshape(-1)
    extras = _norm_extras(extras)
    if xlim is None:
        # robust auto-range: clip to the 0.05–99.95 percentiles so a handful of
        # rare outliers don't stretch the axis and flatten the visible shape.
        both = np.concatenate([b, m] + [_np(e["m"]).reshape(-1) for e in extras])
        lo = float(np.percentile(both, 0.05))
        hi = float(np.percentile(both, 99.95))
        pad = 0.03 * (hi - lo)
        xlim = (lo - pad, hi + pad)
    edges = np.linspace(*xlim, bins + 1)

    ax.hist(b, bins=edges, density=True, histtype="stepfilled",
            color=BENCH_FILL, alpha=0.12, lw=0.0)
    ax.hist(b, bins=edges, density=True, histtype="step",
            color=BENCH_COLOR, lw=1.7, label="benchmark (MZ)")
    ax.hist(m, bins=edges, density=True, histtype="step",
            color=model_color, lw=2.1, label=label)
    for e in extras:
        ax.hist(_np(e["m"]).reshape(-1), bins=edges, density=True, histtype="step",
                color=e["color"], lw=2.1, ls=e["ls"], label=e["label"])

    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.02)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("probability density")
    if show_legend:
        ax.legend(loc="best")


def draw_dx_dvx(ax_dx, ax_dvx, b_dx, b_dvx, dx, dvx, label="model", *,
                bins=120, dx_xlim=None, dvx_xlim=None,
                model_color=MODEL_COLOR, show_legend=True, extras=None) -> None:
    """Two-panel Δx / Δv_x distributions onto supplied axes (plot 02).

    ``extras`` overlays further models, each a dict ``{"dx", "dvx", "label",
    "color", "ls"}`` (colour/ls optional)."""
    extras = _norm_extras(extras)
    ex_dx = [{**e, "m": e["dx"]} for e in extras]
    ex_dvx = [{**e, "m": e["dvx"]} for e in extras]
    draw_dist(ax_dx, b_dx, dx, label, bins=bins, xlim=dx_xlim,
              model_color=model_color, xlabel=r"$\Delta x$  (bond length)",
              show_legend=show_legend, extras=ex_dx)
    ax_dx.set_title(r"$P(\Delta x)$")

    draw_dist(ax_dvx, b_dvx, dvx, label, bins=bins, xlim=dvx_xlim,
              model_color=model_color,
              xlabel=r"$\Delta v_x$  (axis-relative velocity)",
              show_legend=show_legend, extras=ex_dvx)
    ax_dvx.set_title(r"$P(\Delta v_x)$")


def fig_dx_dvx(b_dx, b_dvx, dx, dvx, label="model", *, bins=120,
               dx_xlim=None, dvx_xlim=None, model_color=MODEL_COLOR,
               extras=None, figsize=(11, 4.2)) -> plt.Figure:
    """Standalone Δx / Δv_x distribution figure (plot 02)."""
    use_pub_style()
    fig, (ax_dx, ax_dvx) = plt.subplots(1, 2, figsize=figsize)
    draw_dx_dvx(ax_dx, ax_dvx, b_dx, b_dvx, dx, dvx, label, bins=bins,
                dx_xlim=dx_xlim, dvx_xlim=dvx_xlim, model_color=model_color,
                extras=extras)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 03 — Δx / Δv_x autocorrelation functions
# ═════════════════════════════════════════════════════════════════════════════

def draw_acf(ax, lags, b_acf, m_acf, label="model", *, model_color=MODEL_COLOR,
             xlim=None, title="", show_legend=True, extras=None) -> None:
    """Draw a benchmark-vs-model ACF comparison onto ``ax``.

    ``extras`` overlays further models, each a dict ``{"lags", "acf", "label",
    "color", "ls"}`` (colour/ls optional; ``lags`` optional, defaults to the
    primary ``lags``)."""
    extras = _norm_extras(extras)
    ax.axhline(0.0, color=GRID_COLOR, lw=0.8, zorder=0)
    ax.plot(lags, _np(b_acf), color=BENCH_COLOR, lw=2.0, label="benchmark (MZ)")
    ax.plot(lags, _np(m_acf), color=model_color, lw=2.0, ls="--", label=label)
    for e in extras:
        ax.plot(_np(e.get("lags", lags)), _np(e["acf"]), color=e["color"],
                lw=2.0, ls=e["ls"], label=e["label"])
    ax.set_xlabel("lag time  [ns]")
    ax.set_ylabel("autocorrelation")
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="best")


def draw_acf_dx_dvx(ax_dx, ax_dvx, lags, b_acf_dx, m_acf_dx, b_acf_dvx,
                    m_acf_dvx, label="model", *, model_color=MODEL_COLOR,
                    dx_xlim=None, dvx_xlim=(0.0, 10.0), show_legend=True,
                    extras=None) -> None:
    """Two-panel Δx / Δv_x ACF onto supplied axes (plot 03).

    ``extras`` overlays further models, each a dict ``{"lags", "acf_dx",
    "acf_dvx", "label", "color", "ls"}`` (colour/ls/lags optional)."""
    extras = _norm_extras(extras)
    ex_dx = [{**e, "acf": e["acf_dx"]} for e in extras]
    ex_dvx = [{**e, "acf": e["acf_dvx"]} for e in extras]
    draw_acf(ax_dx, lags, b_acf_dx, m_acf_dx, label, model_color=model_color,
             xlim=dx_xlim, title=r"ACF: $\Delta x$", show_legend=show_legend,
             extras=ex_dx)
    draw_acf(ax_dvx, lags, b_acf_dvx, m_acf_dvx, label, model_color=model_color,
             xlim=dvx_xlim, title=r"ACF: $\Delta v_x$", show_legend=show_legend,
             extras=ex_dvx)


def fig_acf_dx_dvx(lags, b_acf_dx, m_acf_dx, b_acf_dvx, m_acf_dvx, label="model",
                   *, model_color=MODEL_COLOR, dx_xlim=None, dvx_xlim=(0.0, 10.0),
                   extras=None, figsize=(11, 4.2)) -> plt.Figure:
    """Standalone Δx / Δv_x ACF figure (plot 03)."""
    use_pub_style()
    fig, (ax_dx, ax_dvx) = plt.subplots(1, 2, figsize=figsize)
    draw_acf_dx_dvx(ax_dx, ax_dvx, lags, b_acf_dx, m_acf_dx, b_acf_dvx,
                    m_acf_dvx, label, model_color=model_color,
                    dx_xlim=dx_xlim, dvx_xlim=dvx_xlim, extras=extras)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 14 — first-passage-time distributions (CO / OC)
# ═════════════════════════════════════════════════════════════════════════════

_TRANS_TITLE = {"CO": r"CO  (closed $\rightarrow$ open)",
                "OC": r"OC  (open $\rightarrow$ closed)"}


def _fpt_legend(name, data, ref=None):
    data = np.asarray(data).reshape(-1)
    if ref is None:
        return f"{name}   $\\mu$={data.mean():.1f} ns"
    w1 = wasserstein_distance(ref, data)
    ks = ks_2samp(ref, data)[0]
    return f"{name}   $\\mu$={data.mean():.1f} ns   $W_1$={w1:.2f}  KS={ks:.3f}"


def draw_fpt_row(ax_hist, ax_kde, row, label="model", *, bins=60,
                 model_color=MODEL_COLOR, xmax=None, bw_method=None,
                 show_means=True, extras=None) -> None:
    """Draw one transition's FPT histogram + reflected KDE onto two axes.

    ``ax_hist`` may be ``None`` to draw the reflected-KDE panel only. Additional
    models are overlaid from ``row["extras"]`` (or the ``extras`` argument), each
    a dict ``{"m_fpt", "label", "color", "ls"}`` (colour/ls optional)."""
    ttype = row["transition_type"]
    b_fpt = np.asarray(row["b_fpt"]).reshape(-1)
    available = row["available"]
    m_fpt = np.asarray(row["m_fpt"]).reshape(-1) if available else None
    extras = _norm_extras(row.get("extras") if extras is None else extras)
    for e in extras:
        e["_data"] = np.asarray(e["m_fpt"]).reshape(-1)

    if xmax is None:
        pool = [b_fpt] + ([m_fpt] if available else []) + [e["_data"] for e in extras]
        alld = np.concatenate(pool)
        hi = alld.max()
        xmax = min(hi, float(np.percentile(alld, 99.5)) * 1.15)
    edges = np.linspace(0.0, xmax, bins + 1)
    grid = np.linspace(0.0, xmax, 512)

    # histogram panel (optional)
    if ax_hist is not None:
        ax_hist.hist(b_fpt, bins=edges, density=True, histtype="stepfilled",
                     color=BENCH_FILL, alpha=0.12, lw=0.0)
        ax_hist.hist(b_fpt, bins=edges, density=True, histtype="step",
                     color=BENCH_COLOR, lw=1.7,
                     label=_fpt_legend("benchmark (MZ)", b_fpt))
        if available:
            ax_hist.hist(m_fpt, bins=edges, density=True, histtype="step",
                         color=model_color, lw=2.1,
                         label=_fpt_legend(label, m_fpt, ref=b_fpt))
        for e in extras:
            ax_hist.hist(e["_data"], bins=edges, density=True, histtype="step",
                         color=e["color"], lw=2.1, ls=e["ls"],
                         label=_fpt_legend(e["label"], e["_data"], ref=b_fpt))
        ax_hist.set_xlim(0, xmax)
        ax_hist.set_ylim(bottom=0)
        ax_hist.set_xlabel("first-passage time  [ns]")
        ax_hist.set_ylabel("probability density")
        ax_hist.set_title(f"{_TRANS_TITLE.get(ttype, ttype)} — histogram")
        ax_hist.legend(loc="best")

    # reflected-KDE panel
    ax_kde.plot(grid, reflected_kde(b_fpt, grid, bw_method), color=BENCH_COLOR,
                lw=2.0, label=_fpt_legend("benchmark (MZ)", b_fpt))
    if available:
        ax_kde.plot(grid, reflected_kde(m_fpt, grid, bw_method),
                    color=model_color, lw=2.0, ls="--",
                    label=_fpt_legend(label, m_fpt, ref=b_fpt))
    for e in extras:
        ax_kde.plot(grid, reflected_kde(e["_data"], grid, bw_method),
                    color=e["color"], lw=2.0, ls=e["ls"],
                    label=_fpt_legend(e["label"], e["_data"], ref=b_fpt))
    if show_means:
        ax_kde.axvline(b_fpt.mean(), color=BENCH_COLOR, ls=":", lw=1.1, alpha=0.6)
        if available:
            ax_kde.axvline(m_fpt.mean(), color=model_color, ls=":", lw=1.1, alpha=0.6)
        for e in extras:
            ax_kde.axvline(e["_data"].mean(), color=e["color"], ls=":", lw=1.1, alpha=0.6)
    ax_kde.set_xlim(0, xmax)
    ax_kde.set_ylim(bottom=0)
    ax_kde.set_xlabel("first-passage time  [ns]")
    ax_kde.set_ylabel("probability density")
    title = _TRANS_TITLE.get(ttype, ttype)
    ax_kde.set_title(title if ax_hist is None else f"{title} — reflected KDE")
    ax_kde.legend(loc="best")

    if not available:
        for ax in (ax_hist, ax_kde):
            if ax is not None:
                ax.text(0.5, 0.9, f"{label}: FPT data unavailable for {ttype}",
                        transform=ax.transAxes, ha="center", va="top",
                        color=model_color, fontsize=9)


def draw_fpt(axes, rows, label="model", *, bins=60, model_color=MODEL_COLOR,
             bw_method=None, xmax=None, kde_only=False, extras=None) -> None:
    """Draw all FPT rows onto an axes array (plot 14).

    ``axes`` is (n_rows, 2) for the histogram+KDE layout, or (n_rows,) / (n_rows,1)
    when ``kde_only``. ``extras`` is a per-row list of overlay-model lists (or a
    single list applied to every row); each overlay item is a dict
    ``{"m_fpt", "label", "color", "ls"}``."""
    axes = np.atleast_2d(axes)
    for i, row in enumerate(rows):
        ex = extras[i] if (extras is not None and isinstance(extras[0], list)) else extras
        if kde_only:
            draw_fpt_row(None, axes[i, 0], row, label=label, bins=bins,
                         model_color=model_color, bw_method=bw_method, xmax=xmax,
                         extras=ex)
        else:
            draw_fpt_row(axes[i, 0], axes[i, 1], row, label=label, bins=bins,
                         model_color=model_color, bw_method=bw_method, xmax=xmax,
                         extras=ex)


def fig_fpt(rows, label="model", *, bins=60, model_color=MODEL_COLOR,
            bw_method=None, xmax=None, kde_only=False, extras=None,
            figsize=None) -> plt.Figure:
    """Standalone FPT figure (plot 14), one row per transition.

    ``kde_only`` drops the histogram column and lays the transitions out in a
    single row of reflected-KDE panels. ``extras`` overlays further models
    (see :func:`draw_fpt_row`)."""
    use_pub_style()
    n = len(rows)
    if kde_only:
        if figsize is None:
            figsize = (6 * n, 4.2)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        draw_fpt(axes.reshape(n, 1), rows, label=label, bins=bins,
                 model_color=model_color, bw_method=bw_method, xmax=xmax,
                 kde_only=True, extras=extras)
    else:
        if figsize is None:
            figsize = (11, 3.9 * n)
        fig, axes = plt.subplots(n, 2, figsize=figsize, squeeze=False)
        draw_fpt(axes, rows, label=label, bins=bins, model_color=model_color,
                 bw_method=bw_method, xmax=xmax, extras=extras)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Combined "key diagnostics" figure (plot 16): 02 + 03 + 14 stacked
# ═════════════════════════════════════════════════════════════════════════════

def fig_key_diagnostics(roll, fpt_rows, label="model", *, model_color=MODEL_COLOR,
                        dist_bins=120, fpt_bins=60, bw_method=None,
                        dvx_acf_xlim=(0.0, 10.0), title=None) -> plt.Figure:
    """
    Combined figure of the three most-important diagnostics on one canvas:

        row 0 :  P(Δx)        |  P(Δv_x)            (plot 02)
        row 1 :  ACF Δx       |  ACF Δv_x           (plot 03)
        row 2 :  CO FPT hist  |  CO FPT KDE         (plot 14)
        row 3 :  OC FPT hist  |  OC FPT KDE         (plot 14)

    ``roll`` is the dict returned by :func:`prepare_rollout_arrays`; ``fpt_rows``
    the list returned by :func:`prepare_fpt_rows` (ordered CO, OC).
    """
    use_pub_style()
    nfpt = len(fpt_rows)
    nrows = 2 + nfpt
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.7 * nrows))

    draw_dx_dvx(axes[0, 0], axes[0, 1], roll["b_dx"], roll["b_dvx"],
                roll["dx"], roll["dvx"], label, bins=dist_bins,
                model_color=model_color)
    draw_acf_dx_dvx(axes[1, 0], axes[1, 1], roll["lags"],
                    roll["b_acf_dx"], roll["m_acf_dx"],
                    roll["b_acf_dvx"], roll["m_acf_dvx"], label,
                    model_color=model_color, dvx_xlim=dvx_acf_xlim)
    draw_fpt(axes[2:], fpt_rows, label=label, bins=fpt_bins,
             model_color=model_color, bw_method=bw_method)

    if title is None:
        title = f"Key reduced-dimer diagnostics — {label} vs benchmark"
    fig.suptitle(title, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Data-prep convenience (notebook-friendly, fully parameterised)
# ═════════════════════════════════════════════════════════════════════════════

def _read_dt(sim_dir: Path, dt: float | None) -> float:
    if dt is not None:
        return dt
    try:
        params = analysisTools.readParameters(str(sim_dir / "parameters"))
        return float(params["dt"]) * int(float(params.get("stride", 1)))
    except Exception:
        print(f"  Warning: could not read dt from {sim_dir}/parameters; using 0.05")
        return 0.05


def prepare_rollout_arrays(sim_dir, bench_dir, *, n_trajs=100, n_bench_trajs=100,
                           mtrajs_acf=50, boxsize=5.0, dt=None,
                           lag_dx=None, verbose=True) -> dict:
    """
    Load benchmark + model rollouts and compute everything the Δx/Δv_x
    distribution (02) and ACF (03) plots need. Every knob a notebook might want
    to sweep (trajectory counts, ACF averaging, box size, dt, ACF lag budget) is
    exposed here.

    Returns a dict with: dt, lags, b_dx/b_dvx/dx/dvx, and the four ACF arrays
    ``b_acf_dx, m_acf_dx, b_acf_dvx, m_acf_dvx``.
    """
    sim_dir = Path(sim_dir)
    bench_dir = Path(bench_dir)
    dt = _read_dt(sim_dir, dt)

    b_q, b_v, _ = load_trajectories(bench_dir / "simMoriZwanzig_", n_bench_trajs,
                                    verbose=verbose)
    q, v, _ = load_trajectories(sim_dir / "simMoriZwanzigReduced_", n_trajs,
                                verbose=verbose)

    b_dx, b_dvx = compute_dx_dvx(b_q, b_v, boxsize=boxsize)
    dx, dvx = compute_dx_dvx(q, v, boxsize=boxsize)

    T_min = min(b_q.shape[1], q.shape[1])
    if lag_dx is None:
        lag_dx = min(2000, T_min // 5)

    b_acf_dx = acf_tensor(b_dx, lag_dx, mtrajs_acf)
    m_acf_dx = acf_tensor(dx, lag_dx, mtrajs_acf)
    b_acf_dvx = acf_tensor(b_dvx, lag_dx, mtrajs_acf)
    m_acf_dvx = acf_tensor(dvx, lag_dx, mtrajs_acf)

    return {
        "dt": dt,
        "lags": np.arange(lag_dx) * dt,
        "b_dx": b_dx, "b_dvx": b_dvx, "dx": dx, "dvx": dvx,
        "b_acf_dx": b_acf_dx, "m_acf_dx": m_acf_dx,
        "b_acf_dvx": b_acf_dvx, "m_acf_dvx": m_acf_dvx,
    }


def prepare_fpt_rows(fpt_sim_dir, fpt_bench_dir, *, order=("CO", "OC"),
                     verbose=True) -> list[dict]:
    """
    Build FPT comparison rows for both transition types from a single model FPT
    directory (the sibling transition dir is auto-discovered, matching
    ``run_fpt_diagnostics``). Each row: transition_type, b_fpt, m_fpt, available.
    """
    fpt_sim_dir = Path(fpt_sim_dir)
    fpt_bench_dir = Path(fpt_bench_dir)
    this_type, other_type, other_dir = other_transition_fpt_dir(fpt_sim_dir)
    sim_dir_by_type = {this_type: fpt_sim_dir, other_type: other_dir}

    rows = []
    for ttype in order:
        b_fpt = load_fpt(find_fpt_file(fpt_bench_dir, ttype))
        sdir = sim_dir_by_type[ttype]
        m_fpt, available = None, False
        if sdir.exists():
            try:
                m_fpt = load_fpt(find_fpt_file(sdir, ttype))
                available = True
            except FileNotFoundError:
                pass
        if verbose:
            n = len(m_fpt) if available else 0
            print(f"  [{ttype}] bench {len(b_fpt)} | model {n}"
                  f"{'' if available else ' (unavailable)'}")
        rows.append({"transition_type": ttype, "b_fpt": b_fpt,
                     "m_fpt": m_fpt, "available": available,
                     "m_dir_expected": str(sdir)})
    return rows
