"""
Ensemble / comparison scoreboard for reduced-dimer CVAE (and E3) rollouts.

Parallel to dimerRolloutDiagnostics.py, which stays a clean SINGLE-model
scoreboard. This script scores MANY rollouts and reports the two-way variance
decomposition used for model reproducibility:

  * inter-MODEL   — one rollout per distinct trained model (training-seed spread)
  * inter-ROLLOUT — several independent rollouts of ONE checkpoint (RNG floor)

It uses the SAME estimators as the single-model scoreboard, all imported from the
source ``rollout_diags`` module (no duplicated logic): core-set entry-to-arrival
FPT (margin 0.3), Δx / Δv_x / r-parallel marginals, and the Δx-ACF timescale. The
benchmark FPT comes from benchmarkFPTreference/ (itself core-set).

Manifest (JSON)::

    {
      "title": "CVAE_LF DAG1 — reproducibility",
      "n_trajs": 60, "n_bench_trajs": 500,
      "out": "scoreboard_dag1.txt",
      "fpt_bench_dir": "<abs>/benchmarkFPTreference",       # optional
      "inter_model":  {"s1": "<abs rollout dir>", "s2": "...", ...},
      "inter_rollout": {"model": "s1", "dirs": {"r1": "<abs>", "r2": "..."}}
    }

Each rollout dir must contain simMoriZwanzigReduced_*.h5 (+ a parameters file for
dt). Paths are absolute, or relative to --cvae-runs-root if given.

    python dimerDiagnosticsComparison.py --manifest manifest_dag1.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler.diagnostics.rollout_diags import (
    load_trajectories, compute_dx_dvx, _decompose, acf_tensor,
    entry_to_arrival_fpt, find_fpt_file, load_fpt,
    FPT_X0, FPT_RAD, FPT_MARGIN,
)

DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5"
DEFAULT_FPT_BENCH_DIR = f"{DEFAULT_OUTPUT_ROOT}/benchmarkFPTreference"
DEFAULT_BENCH_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark"
BOXSIZE = 5.0
LO, HI = FPT_X0 + FPT_MARGIN, FPT_X0 + 2 * FPT_RAD - FPT_MARGIN


def _read_dt(sim_dir: Path, fallback=0.05):
    try:
        p = analysisTools.readParameters(str(sim_dir / "parameters"))
        return float(p["dt"]) * int(float(p.get("stride", 1)))
    except Exception:
        return fallback


def _sub(a, m=400000):
    a = np.asarray(a).reshape(-1)
    return a if len(a) <= m else a[np.linspace(0, len(a) - 1, m).astype(int)]


def _exp_tau(acf, dt, max_lag_ns=100.0):
    """Simple e-folding timescale of a (normalized) ACF, in ns."""
    acf = np.asarray(acf).reshape(-1)
    nmax = min(len(acf), int(max_lag_ns / dt) if dt > 0 else len(acf))
    a = acf[:nmax]
    below = np.where(a < 1.0 / np.e)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    # linear interpolation between i-1 and i for the 1/e crossing
    y0, y1 = a[i - 1], a[i]
    frac = (y0 - 1.0 / np.e) / (y0 - y1) if y0 != y1 else 0.0
    return float((i - 1 + frac) * dt)


def score(sim_dir: Path, base_prefix: str, n_trajs: int, dt: float,
          label: str, fpt_bench_dir: Path, ref=None, use_ref_fpt=False):
    """Score one rollout dir on marginals + core-set FPT. If use_ref_fpt, take
    CO/OC from the benchmark reference .xyz instead of recomputing (bench row)."""
    q, v, r = load_trajectories(str(sim_dir / base_prefix), n_trajs, max_scan=max(4200, n_trajs + 200))
    dx, dvx = compute_dx_dvx(q, v, boxsize=BOXSIZE)
    dec = _decompose(r[..., :3], r[..., 3:], q[..., :3], q[..., 3:], boxsize=BOXSIZE)
    rpar = dec["rel_par_scalar"]

    dx_np = dx.detach().cpu().numpy()
    dvx_np = dvx.detach().cpu().numpy()
    rpar_np = rpar.detach().cpu().numpy()

    if use_ref_fpt:
        co = load_fpt(find_fpt_file(fpt_bench_dir, "CO"))
        oc = load_fpt(find_fpt_file(fpt_bench_dir, "OC"))
    else:
        co = entry_to_arrival_fpt(dx_np, LO, HI, "CO", dt)
        oc = entry_to_arrival_fpt(dx_np, LO, HI, "OC", dt)

    acf_dx = acf_tensor(dx, min(1000, dx.shape[1] // 5), mTrajs=min(50, dx.shape[0]))
    out = dict(
        label=label, n_traj=int(dx.shape[0]),
        dx_mean=float(dx_np.mean()), dx_std=float(dx_np.std()),
        dvx_std=float(dvx_np.std()), rpar_std=float(rpar_np.std()),
        tau_x=_exp_tau(acf_dx, dt),
        co=float(np.mean(co)), oc=float(np.mean(oc)),
        k=float(np.mean(oc) / np.mean(co)), nco=len(co), noc=len(oc),
        _dx=_sub(dx_np), _dvx=_sub(dvx_np), _rpar=_sub(rpar_np),
    )
    if ref is not None:
        out["ks_dx"] = float(ks_2samp(out["_dx"], ref["_dx"])[0])
        out["ks_dvx"] = float(ks_2samp(out["_dvx"], ref["_dvx"])[0])
        out["ks_rpar"] = float(ks_2samp(out["_rpar"], ref["_rpar"])[0])
    return out


HDR = (f"{'model':<14s} {'CO':>7s} {'OC':>7s} {'k':>6s} {'dxmean':>7s} {'dxstd':>7s} "
       f"{'dvxstd':>7s} {'rparstd':>8s} {'tau_x':>6s} {'KSdx':>6s} {'KSdvx':>6s} {'KSrp':>6s} {'nCO':>6s}")


def row(s):
    return (f"{s['label']:<14s} {s['co']:>7.2f} {s['oc']:>7.2f} {s['k']:>6.3f} "
            f"{s['dx_mean']:>7.3f} {s['dx_std']:>7.3f} {s['dvx_std']:>7.3f} {s['rpar_std']:>8.4f} "
            f"{s['tau_x']:>6.2f} {s.get('ks_dx', np.nan):>6.3f} {s.get('ks_dvx', np.nan):>6.3f} "
            f"{s.get('ks_rpar', np.nan):>6.3f} {s['nco']:>6d}")


def spread(rows, keys, title):
    out = ["", f"--- {title} ---",
           f"{'qty':<9s} {'mean':>9s} {'std':>9s} {'CV%':>7s} {'min':>9s} {'max':>9s}"]
    for k in keys:
        a = np.array([r[k] for r in rows], dtype=float)
        cv = 100 * a.std() / abs(a.mean()) if a.mean() != 0 else float("nan")
        out.append(f"{k:<9s} {a.mean():>9.3f} {a.std():>9.4f} {cv:>7.2f} {a.min():>9.3f} {a.max():>9.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Ensemble reproducibility scoreboard for dimer rollouts.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cvae-runs-root", default=None,
                    help="If given, manifest dirs are resolved relative to this (e.g. .../cvaeRuns).")
    ap.add_argument("--bench-dir", default=DEFAULT_BENCH_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="Directory for the output scoreboard (default: manifest's dir).")
    args = ap.parse_args()

    m = json.loads(Path(args.manifest).read_text())
    n_trajs = m["n_trajs"]
    n_bench = m.get("n_bench_trajs", 500)
    fpt_bench_dir = Path(m.get("fpt_bench_dir", DEFAULT_FPT_BENCH_DIR))
    root = Path(args.cvae_runs_root) if args.cvae_runs_root else None

    def resolve(d):
        p = Path(d)
        return (root / p) if (root and not p.is_absolute()) else p

    keys = ["co", "oc", "k", "dx_mean", "dx_std", "dvx_std", "rpar_std", "tau_x"]

    # ── benchmark row: marginals from raw MD, FPT from the core-set reference ──
    print("scoring benchmark ...", flush=True)
    bench_dir = Path(args.bench_dir)
    dt_b = _read_dt(bench_dir)
    b = score(bench_dir, "simMoriZwanzig_", n_bench, dt_b, "benchmark",
              fpt_bench_dir, ref=None, use_ref_fpt=True)

    lines = [f"=== {m.get('title', 'dimer comparison')} ===",
             f"benchmark  CO {b['co']:.2f} OC {b['oc']:.2f} k {b['k']:.3f} | "
             f"dx {b['dx_mean']:.3f}/{b['dx_std']:.3f} dvx_std {b['dvx_std']:.4f} "
             f"rpar_std {b['rpar_std']:.4f} tau_x {b['tau_x']:.2f} (core-set margin {FPT_MARGIN})", ""]

    for block, header in (("inter_model", "INTER-MODEL (training-seed spread): one rollout per trained model"),
                          ("inter_rollout", None)):
        if not m.get(block):
            continue
        if block == "inter_model":
            entries = m[block]
            lines += [f"## {header}", HDR]
        else:
            ir = m[block]
            entries = ir["dirs"]
            lines += ["", f"## INTER-ROLLOUT (rollout-RNG spread): {len(entries)} rollouts of model '{ir['model']}'", HDR]
        rows = []
        for lab, d in entries.items():
            print(f"scoring {block} {lab} ...", flush=True)
            sim_dir = resolve(d)
            s = score(sim_dir, "simMoriZwanzigReduced_", n_trajs, _read_dt(sim_dir),
                      lab, fpt_bench_dir, ref=b)
            rows.append(s)
            lines.append(row(s))
        if len(rows) >= 2:
            tag = "training-seed variance" if block == "inter_model" else "rollout-RNG noise floor"
            lines += spread(rows, keys, f"{block.replace('_', '-')} spread ({tag})")

    txt = "\n".join(lines)
    print("\n" + txt)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.manifest).resolve().parent
    out_path = out_dir / m.get("out", "scoreboard_comparison.txt")
    out_path.write_text(txt + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
