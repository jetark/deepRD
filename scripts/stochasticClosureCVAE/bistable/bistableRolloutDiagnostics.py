"""
Diagnostics scoreboard for reduced-bistable CVAE_SP rollouts — the single-particle
analogue of dimerRolloutDiagnostics.py.

Compares a model rollout (produced by benchmarkReducedBistableCVAEGen.py) against
the full-MD benchmark on:
  * x / y / z position marginals
  * x / y / z velocity marginals
  * x / y / z noise (r) marginals + ||r||
  * ACFs: x-position (the slow well-hopping coordinate), velocity, r
  * L->R and R->L first-passage times

FPTs are extracted with the core-set entry-to-arrival crossing analysis
(tests/benchmarkFPTs/compute_true_fpt.py::entry_to_arrival_fpt) applied to the
long, continuous rollout — NOT the cold-start `propagateFPT` protocol, which
resets v=0 / zero r-history and imposes a warm-up dead-time bias (see
tests/benchmarkFPTs/RESULTS.md and PROJECT_CONTEXT.md §7/§10). The benchmark
reference is computed the same way, so both sides are consistent. The collective
variable is the particle x-position; wells sit at x = ±MINIMA_DIST (1.5), and
L->R / R->L should be roughly symmetric.

--sim-dir is the path *relative to* the bistable cvaeRuns/ root, e.g.:

    python bistableRolloutDiagnostics.py --sim-dir piri/001_base/100x10000

Plots + text summaries are written to:
    deepRD/noiseSampler/training/results/bistable/<cond>/<id_name>/diags/<simrun>/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests" / "benchmarkFPTs"))

import deepRD.tools.analysisTools as analysisTools
import deepRD.tools.trajectoryTools as trajectoryTools
from deepRD.noiseSampler.diagnostics.rollout_diags import reflected_kde, compute_fpt_stats
from compute_true_fpt import entry_to_arrival_fpt

DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5"
CVAE_RUNS_ROOT = Path(DEFAULT_OUTPUT_ROOT) / "cvaeRuns"
DEFAULT_BENCH_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5/benchmark"
RESULTS_ROOT = _REPO_ROOT / "deepRD" / "noiseSampler" / "training" / "results" / "bistable"

MINIMA_DIST = 1.5           # external double-well minima at x = ±1.5
BENCH_COLOR = "tab:blue"
MODEL_COLOR = "tab:orange"
COORD = ["x", "y", "z"]


# ── data loading ───────────────────────────────────────────────────────────────

def _parse_single_particle(ds: np.ndarray):
    """
    ds: (T, 10 or 11) from loadTrajectory for the single distinguished particle.
    Benchmark files have 11 columns (an extra col 7); reduced rollouts have 10.
    Returns q, v, r each (T, 3).
    """
    ds = np.asarray(ds, dtype=np.float64)
    if ds.shape[1] == 11:  # drop the extra column so layout is [t, q, v, r]
        ds = np.concatenate([ds[:, :7], ds[:, -3:]], axis=1)
    q = ds[:, 1:4]
    v = ds[:, 4:7]
    r = ds[:, 7:10]
    return q, v, r


def load_trajectories(base_path, n_trajs, max_scan=2600, verbose=True):
    """Load up to n_trajs single-particle trajectory files. Returns q, v, r each (n, T, 3)."""
    base_path = str(base_path)
    qs, vs, rs = [], [], []
    loaded = 0
    for fnum in range(max_scan):
        if loaded >= n_trajs:
            break
        try:
            ds = trajectoryTools.loadTrajectory(base_path, fnum)
        except (FileNotFoundError, OSError, KeyError):
            continue
        q, v, r = _parse_single_particle(ds)
        qs.append(q); vs.append(v); rs.append(r)
        loaded += 1
        if verbose and loaded % 50 == 0:
            print(f"  {loaded}/{n_trajs} loaded", end="\r")
    if not qs:
        raise FileNotFoundError(f"No trajectory files found at {base_path}")
    # Trajectories may differ in length (benchmark vs rollout); truncate to min.
    T = min(a.shape[0] for a in qs)
    q = np.stack([a[:T] for a in qs]); v = np.stack([a[:T] for a in vs]); r = np.stack([a[:T] for a in rs])
    if verbose:
        print(f"  Loaded {q.shape[0]} trajectories x {q.shape[1]} steps from {base_path}     ")
    return q, v, r


# ── ACF ────────────────────────────────────────────────────────────────────────

def _acf_1d(series, trunc):
    """
    Unnormalized autocovariance of an ALREADY-CENTERED 1D series (length trunc).
    The caller must subtract the ensemble (global) mean before calling — do NOT
    re-center per trajectory here. For the bistable x-position this is critical:
    benchmark trajectories are short (500 ns) and often sit in one well, so a
    per-trajectory mean removes the slow inter-well hopping signal and makes the
    ACF decay artificially fast (a false model-vs-benchmark mismatch). Using the
    global mean keeps the hopping memory — the connected ACF.
    """
    n = len(series)
    s_pad = np.concatenate([series, np.zeros(n)])
    f = np.fft.fft(s_pad)
    corr = np.fft.ifft(f * np.conj(f)).real[:trunc]
    corr /= np.linspace(n, n - trunc + 1, trunc)
    return corr


def acf_component(arr, comp, lag, mtrajs, rng):
    """Normalized connected ACF of a single vector component (global-mean centered)."""
    n = arr.shape[0]
    gmean = float(arr[..., comp].mean())   # ensemble mean over all trajs & time
    idx = rng.choice(n, min(mtrajs, n), replace=False)
    acc = np.zeros(lag)
    for i in idx:
        acc += _acf_1d(arr[i, :, comp] - gmean, lag)
    return acc / (acc[0] if acc[0] != 0 else 1.0)


def acf_vector(arr, lag, mtrajs, rng):
    """Normalized connected ACF summed over all 3 components (per-dim global-mean centered)."""
    n = arr.shape[0]
    gmean = arr.reshape(-1, arr.shape[-1]).mean(axis=0)   # per-component ensemble mean
    idx = rng.choice(n, min(mtrajs, n), replace=False)
    acc = np.zeros(lag)
    for i in idx:
        for d in range(arr.shape[-1]):
            acc += _acf_1d(arr[i, :, d] - float(gmean[d]), lag)
    return acc / (acc[0] if acc[0] != 0 else 1.0)


# ── FPT (core-set, cold-start-free) ──────────────────────────────────────────────

def coreset_fpts(x, margin, dt):
    """
    x: (n_traj, T) particle x-position. Returns dict with 'LR' and 'RL' FPT arrays.
    Wells at ±MINIMA_DIST; left core x < -MINIMA_DIST+margin, right core x > MINIMA_DIST-margin.
    L->R maps to entry_to_arrival_fpt direction 'CO' (A=left, B=right); R->L to 'OC'.
    """
    lo = -MINIMA_DIST + margin
    hi = MINIMA_DIST - margin
    return {
        "LR": entry_to_arrival_fpt(x, lo, hi, "CO", dt),
        "RL": entry_to_arrival_fpt(x, lo, hi, "OC", dt),
    }


# ── plots ────────────────────────────────────────────────────────────────────────

def fig_marginals(bench, model, var_name, label, xlims, bins=120):
    """1x3 marginal histograms (x/y/z) benchmark vs model."""
    b = bench.reshape(-1, 3); m = model.reshape(-1, 3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in range(3):
        edges = np.linspace(*xlims[c], bins + 1)
        axes[c].hist(b[:, c], bins=edges, density=True, histtype="step", color=BENCH_COLOR, label="benchmark")
        axes[c].hist(m[:, c], bins=edges, density=True, histtype="step", color=MODEL_COLOR, label=label)
        axes[c].set_title(f"{var_name}_{COORD[c]}")
        axes[c].set_xlabel(f"{var_name}_{COORD[c]}"); axes[c].grid(alpha=0.25)
        if c == 0:
            axes[c].set_ylabel("density"); axes[c].legend()
    fig.suptitle(f"{var_name} marginal distributions")
    fig.tight_layout()
    return fig


def fig_acf(lags, curves, title, label, xlabel="lag [ns]"):
    """curves: list of (bench_acf, model_acf, panel_title, xlim) tuples."""
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4), squeeze=False)
    for ax, (b, m, ptitle, xlim) in zip(axes[0], curves):
        L = len(b)
        ax.plot(lags[:L], b, lw=2, color=BENCH_COLOR, label="benchmark")
        ax.plot(lags[:L], m, "--", lw=2, color=MODEL_COLOR, label=label)
        ax.set_title(ptitle); ax.set_xlabel(xlabel); ax.set_ylabel("ACF"); ax.grid(alpha=0.3); ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_fpt(fpt_rows, label, bins=80):
    """2 rows (LR, RL) x 2 cols (histogram, reflected KDE)."""
    fig, axes = plt.subplots(len(fpt_rows), 2, figsize=(12, 4 * len(fpt_rows)), squeeze=False)
    for i, row in enumerate(fpt_rows):
        b, m, name = row["b_fpt"], row["m_fpt"], row["name"]
        tmax = float(np.percentile(np.concatenate([b, m]), 99.5))
        edges = np.linspace(0, tmax, bins + 1)
        axes[i, 0].hist(b, bins=edges, density=True, histtype="step", color=BENCH_COLOR, label="benchmark")
        axes[i, 0].hist(m, bins=edges, density=True, histtype="step", color=MODEL_COLOR, label=label)
        axes[i, 0].set_xlim(0, tmax); axes[i, 0].set_xlabel("FPT [ns]"); axes[i, 0].set_ylabel("density")
        axes[i, 0].set_title(f"{name}: FPT histogram"); axes[i, 0].legend(); axes[i, 0].grid(alpha=0.25)
        tgrid = np.linspace(0, tmax, 512)
        axes[i, 1].plot(tgrid, reflected_kde(b, tgrid), lw=2, color=BENCH_COLOR, label="benchmark")
        axes[i, 1].plot(tgrid, reflected_kde(m, tgrid), "--", lw=2, color=MODEL_COLOR, label=label)
        axes[i, 1].set_xlim(0, tmax); axes[i, 1].set_xlabel("FPT [ns]"); axes[i, 1].set_ylabel("density")
        axes[i, 1].set_title(f"{name}: FPT KDE (reflected)"); axes[i, 1].legend(); axes[i, 1].grid(alpha=0.25)
    fig.suptitle("First-passage-time distributions (L->R / R->L)")
    fig.tight_layout()
    return fig


def fig_fpt_survival(fpt_rows, label):
    fig, axes = plt.subplots(1, len(fpt_rows), figsize=(6 * len(fpt_rows), 4.5), squeeze=False)
    for ax, row in zip(axes[0], fpt_rows):
        for fpt, color, lbl in [(row["b_fpt"], BENCH_COLOR, "benchmark"), (row["m_fpt"], MODEL_COLOR, label)]:
            ts = np.sort(fpt)
            surv = 1.0 - np.arange(1, len(ts) + 1) / len(ts)
            ax.step(ts, surv, where="post", lw=2, color=color, label=lbl)
        ax.set_yscale("log"); ax.set_xlabel("FPT [ns]"); ax.set_ylabel("survival P(T>t)")
        ax.set_title(f"{row['name']}: survival"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.suptitle("FPT survival curves")
    fig.tight_layout()
    return fig


def fig_key_panel(b_q, b_v, q, v, lags, acf_pack, fpt_rows, label):
    """Combined panel of the key comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    # position-x and velocity-x marginals
    for ax, bench, model, name, xlim in [
        (axes[0, 0], b_q, q, "position x", (-3, 3)),
        (axes[0, 1], b_v, v, "velocity x", (-1, 1)),
    ]:
        edges = np.linspace(*xlim, 121)
        ax.hist(bench.reshape(-1, 3)[:, 0], bins=edges, density=True, histtype="step", color=BENCH_COLOR, label="benchmark")
        ax.hist(model.reshape(-1, 3)[:, 0], bins=edges, density=True, histtype="step", color=MODEL_COLOR, label=label)
        ax.set_title(name); ax.grid(alpha=0.25); ax.legend()
    # x-position ACF and velocity ACF
    for ax, key, name, xlim in [
        (axes[0, 2], "qx", "x-position ACF", None),
        (axes[1, 0], "v", "velocity ACF", None),
    ]:
        b, m = acf_pack[key]
        L = len(b)
        ax.plot(lags[:L], b, lw=2, color=BENCH_COLOR, label="benchmark")
        ax.plot(lags[:L], m, "--", lw=2, color=MODEL_COLOR, label=label)
        ax.set_title(name); ax.set_xlabel("lag [ns]"); ax.set_ylabel("ACF"); ax.grid(alpha=0.3); ax.legend()
    # FPT KDEs
    for ax, row in zip((axes[1, 1], axes[1, 2]), fpt_rows):
        b, m = row["b_fpt"], row["m_fpt"]
        tmax = float(np.percentile(np.concatenate([b, m]), 99.5))
        tgrid = np.linspace(0, tmax, 512)
        ax.plot(tgrid, reflected_kde(b, tgrid), lw=2, color=BENCH_COLOR, label="benchmark")
        ax.plot(tgrid, reflected_kde(m, tgrid), "--", lw=2, color=MODEL_COLOR, label=label)
        ax.set_xlim(0, tmax); ax.set_xlabel("FPT [ns]"); ax.set_ylabel("density")
        ax.set_title(f"FPT {row['name']}  (bench {row['stats']['bench_mean']:.0f} / model {row['stats'][f'{label}_mean']:.0f} ns)")
        ax.legend(); ax.grid(alpha=0.25)
    fig.suptitle(f"Bistable key diagnostics — {label}", fontsize=14)
    fig.tight_layout()
    return fig


# ── summaries ────────────────────────────────────────────────────────────────────

def marginal_summary(b_q, b_v, b_r, q, v, r, label):
    lines = ["=" * 68, f"BISTABLE ROLLOUT DIAGNOSTICS — model label: {label}", "=" * 68]
    for name, bt, mt in [("q", b_q, q), ("v", b_v, v), ("r", b_r, r)]:
        bn, mn = bt.reshape(-1, 3), mt.reshape(-1, 3)
        lines.append(f"\n── {name}  (benchmark / {label}) ──")
        for c in range(3):
            w = wasserstein_distance(bn[:, c], mn[:, c])
            lines.append(f"   {name}_{COORD[c]}   bench {bn[:,c].mean():+.5f} ± {bn[:,c].std():.5f}   "
                         f"{label} {mn[:,c].mean():+.5f} ± {mn[:,c].std():.5f}   W1={w:.5f}")
    # per-step MSD (per dim)
    bstep = np.diff(b_q, axis=1); mstep = np.diff(q, axis=1)
    lines.append("\n── per-step MSD of q (×10⁻⁵) ──")
    for c in range(3):
        lines.append(f"   {COORD[c]}   bench {(bstep[...,c]**2).mean()*1e5:.4f}   {label} {(mstep[...,c]**2).mean()*1e5:.4f}")
    return "\n".join(lines) + "\n"


def fpt_summary(fpt_rows, label):
    lines = ["=" * 68, f"FIRST-PASSAGE-TIME DIAGNOSTICS (core-set) — model label: {label}", "=" * 68,
             "method: core-set entry-to-arrival on rollout x-position (cold-start-free)"]
    for row in fpt_rows:
        s = row["stats"]
        lines.append(f"\n── {row['name']} " + "─" * 40)
        lines.append(f"   N          bench {s['bench_n']:6d}   {label} {s[f'{label}_n']:6d}")
        lines.append(f"   mean  [ns] bench {s['bench_mean']:8.3f}   {label} {s[f'{label}_mean']:8.3f}")
        lines.append(f"   std   [ns] bench {s['bench_std']:8.3f}   {label} {s[f'{label}_std']:8.3f}")
        lines.append(f"   median[ns] bench {s['bench_median']:8.3f}   {label} {s[f'{label}_median']:8.3f}")
        lines.append(f"   Wasserstein-1: {s['wasserstein']:.4f}   KS: {s['ks_stat']:.4f} (p={s['ks_pvalue']:.3g})")
    # symmetry check
    lr, rl = fpt_rows[0]["stats"], fpt_rows[1]["stats"]
    lines.append(f"\n── symmetry (L->R vs R->L) ──")
    lines.append(f"   benchmark mean: L->R {lr['bench_mean']:.2f}  R->L {rl['bench_mean']:.2f}  "
                 f"ratio {lr['bench_mean']/rl['bench_mean']:.3f}")
    lines.append(f"   {label} mean: L->R {lr[f'{label}_mean']:.2f}  R->L {rl[f'{label}_mean']:.2f}  "
                 f"ratio {lr[f'{label}_mean']/rl[f'{label}_mean']:.3f}")
    return "\n".join(lines) + "\n"


# ── main ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-dir", required=True,
                   help="Path relative to bistable cvaeRuns/ (e.g. piri/001_base/100x10000), "
                        "or an absolute path to any rollout directory (e.g. an old "
                        "benchmarkReducedGen_<cond>/ folder).")
    p.add_argument("--diag-dir", default=None,
                   help="Explicit output directory for figures/summaries (overrides the "
                        "default results/bistable/<cond>/<id>/diags/<simrun>/ location).")
    p.add_argument("--benchmark-dir", default=DEFAULT_BENCH_DIR)
    p.add_argument("--n-trajs", type=int, default=100, help="reduced trajectories to load")
    p.add_argument("--n-bench-trajs", type=int, default=400, help="benchmark trajectories to load")
    p.add_argument("--mtrajs-acf", type=int, default=20)
    p.add_argument("--margin", type=float, default=0.5, help="core-set margin (cores at x<∓(1.5-margin))")
    p.add_argument("--label", default=None)
    p.add_argument("--dt", type=float, default=None, help="output dt in ns (default: read from parameters)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rel = Path(args.sim_dir)
    if rel.is_absolute() and rel.exists():
        sim_dir = rel                      # absolute path to any rollout dir
    else:
        sim_dir = CVAE_RUNS_ROOT / rel     # relative to cvaeRuns/
    if not sim_dir.exists():
        print(f"Error: directory does not exist: {sim_dir}"); sys.exit(1)

    parts = rel.parts
    if args.diag_dir is not None:
        diag_dir = Path(args.diag_dir)
    elif not rel.is_absolute() and len(parts) == 3:
        cond, id_name, simrun = parts
        diag_dir = RESULTS_ROOT / cond / id_name / "diags" / simrun
    else:
        diag_dir = sim_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or (parts[0] if not rel.is_absolute() and parts else sim_dir.name)

    if args.dt is not None:
        dt = args.dt
    else:
        params = analysisTools.readParameters(str(sim_dir / "parameters"))
        dt = float(params["dt"]) * int(float(params.get("stride", 1)))

    print("Loading benchmark trajectories...")
    b_q, b_v, b_r = load_trajectories(Path(args.benchmark_dir) / "simMoriZwanzig_", args.n_bench_trajs)
    print("Loading reduced model trajectories...")
    q, v, r = load_trajectories(sim_dir / "simMoriZwanzigReduced_", args.n_trajs)
    print(f"  benchmark {b_q.shape}  |  model {q.shape}  |  dt={dt} ns")

    # ── ACFs ──
    print("Computing ACFs...")
    # x-position ACF (slow well-hopping): benchmark trajs are short (500 ns) so we
    # (a) use the connected/global-mean estimator (acf_component), (b) compare at
    # MATCHED length by chopping the long rollout into benchmark-length segments,
    # and (c) average over ALL benchmark trajs + all segments (20 short trajs are
    # far too noisy for the hopping ACF). These three together are what make the
    # benchmark and a faithful model agree; getting any wrong fakes a mismatch.
    T_b = b_q.shape[1]
    lag_qx = min(2400, T_b // 4)
    nseg = max(q.shape[1] // T_b, 1)
    q_seg = q[:, :nseg * T_b, :].reshape(q.shape[0] * nseg, T_b, q.shape[2])
    acf_qx = (acf_component(b_q,  0, lag_qx, b_q.shape[0], rng),
              acf_component(q_seg, 0, lag_qx, min(q_seg.shape[0], 2000), rng))
    lag_v = min(1000, b_v.shape[1] // 4, v.shape[1] // 4)    # velocity (fast, zero-mean)
    lag_r = min(100, b_r.shape[1] // 4, r.shape[1] // 4)     # noise r (fast)
    lags = np.arange(max(lag_qx, lag_v, lag_r)) * dt
    acf_v = (acf_vector(b_v, lag_v, args.mtrajs_acf, rng), acf_vector(v, lag_v, args.mtrajs_acf, rng))
    acf_r = (acf_vector(b_r, lag_r, args.mtrajs_acf, rng), acf_vector(r, lag_r, args.mtrajs_acf, rng))

    # ── FPTs (core-set) ──
    print("Computing core-set FPTs...")
    b_fpt = coreset_fpts(b_q[..., 0], args.margin, dt)
    m_fpt = coreset_fpts(q[..., 0], args.margin, dt)
    fpt_rows = []
    for key, name in [("LR", "L->R"), ("RL", "R->L")]:
        row = {"name": name, "b_fpt": b_fpt[key], "m_fpt": m_fpt[key]}
        row["stats"] = compute_fpt_stats(b_fpt[key], m_fpt[key], label=label)
        fpt_rows.append(row)

    # ── save plots ──
    def _save(fig, name):
        fig.savefig(diag_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("Generating plots...")
    _save(fig_marginals(b_q, q, "q", label, [(-3, 3)] * 3), "01_position_marginals")
    _save(fig_marginals(b_v, v, "v", label, [(-1, 1)] * 3), "02_velocity_marginals")
    _save(fig_marginals(b_r, r, "r", label, [(-0.08, 0.08)] * 3), "03_r_marginals")
    _save(fig_acf(lags, [
        (acf_qx[0], acf_qx[1], "x-position ACF (well hopping)", None),
    ], "Position ACF", label), "04_acf_position")
    _save(fig_acf(lags, [
        (acf_v[0], acf_v[1], "velocity ACF", (0, min(50.0, lag_v * dt))),
        (acf_r[0], acf_r[1], "noise r ACF", (0, min(5.0, lag_r * dt))),
    ], "Velocity / noise ACF", label), "05_acf_velocity_r")
    _save(fig_fpt(fpt_rows, label), "06_fpt_distributions")
    _save(fig_fpt_survival(fpt_rows, label), "07_fpt_survival")
    _save(fig_key_panel(b_q, b_v, q, v, lags, {"qx": acf_qx, "v": acf_v}, fpt_rows, label),
          "16_key_diagnostics")

    # ── summaries ──
    (diag_dir / "summary.txt").write_text(marginal_summary(b_q, b_v, b_r, q, v, r, label))
    (diag_dir / "fpt_summary.txt").write_text(fpt_summary(fpt_rows, label))
    print((diag_dir / "summary.txt").read_text())
    print((diag_dir / "fpt_summary.txt").read_text())
    print(f"\nAll diagnostics written to: {diag_dir}")


if __name__ == "__main__":
    main()
