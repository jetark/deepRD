"""
Diagnostics script for reduced CVAE dimer rollouts.

--sim-dir takes the path *relative to* DEFAULT_OUTPUT_ROOT/cvaeRuns/, e.g.:

    python dimerRolloutDiagnostics.py --sim-dir local_dqipipimririm/001_base/100x10000
    python dimerRolloutDiagnostics.py --sim-dir pipimririm/002_run/50x5000_Tr=0.5

Pass --fpt to also compare first-passage-time statistics against the benchmark.
By DEFAULT this uses the project-standard CORE-SET (entry-to-arrival) estimator
applied directly to the rollout Δx — cold-start-free, no separate propagateFPT
run needed (see tests/benchmarkFPTs/RESULTS.md, "Update 2026-07-09"). It compares
against dimerGlobal/boxsize5/benchmarkFPTreference/ (correctly-labeled CO/OC
reference, itself core-set — the original benchmarkFPTcomparison/ "CO" file is
mislabeled and must not be used):

    python dimerRolloutDiagnostics.py --sim-dir local_dqipipimririm/001_base/100x10000 --fpt

For back-compatibility, pass --fpt-external together with --fpt-sim-dir to
instead ingest pre-computed FPT .xyz files (e.g. the reset-based propagateFPT
output of benchmarkFPTreducedDimer{CVAE,E3}Gen.py). --fpt-sim-dir must point to
the FPT output directory for ONE transition type (name matching
FPT_{OC|CO}_<nsims>x<tfinal>[_<tag>]); the sibling directory for the other
transition type is auto-discovered alongside it. NOTE: propagateFPT numbers
carry a cold-start dead-time bias and are kept only for reproducing legacy runs.

    python dimerRolloutDiagnostics.py --sim-dir local_dqipipimririm/001_base/100x10000 \
        --fpt --fpt-external --fpt-sim-dir local_dqipipimririm/001_base/FPT_OC_10000x10000

Pass --fpt-only to skip the standard rollout diagnostics and generate only the
FPT plots (implies --fpt; core-set mode still loads the rollout Δx internally):

    python dimerRolloutDiagnostics.py --sim-dir local_dqipipimririm/001_base/100x10000 --fpt-only

Plots and a text summary are written to:
    noiseSampler/training/results/<cond>/<id_name>/diags/<simrun>/
Override with --diag-dir <path> for non-standard destinations (e.g. a
tests/<campaign>/ scoreboard). --sim-dir may also be an ABSOLUTE path for
rollouts living outside cvaeRuns/ (output then defaults to
<sim-dir>/diagnostics/ unless --diag-dir is given).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must come before pyplot import

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from deepRD.noiseSampler.diagnostics.rollout_diags import (
    run_all_diagnostics, run_fpt_diagnostics, run_fpt_diagnostics_coreset,
    save_combined_key_diagnostics, load_trajectories, compute_dx_dvx,
)
import deepRD.tools.analysisTools as analysisTools

DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5"
CVAE_RUNS_ROOT      = Path(DEFAULT_OUTPUT_ROOT) / "cvaeRuns"
DEFAULT_BENCH_DIR   = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark"
RESULTS_ROOT        = _REPO_ROOT / "deepRD" / "noiseSampler" / "training" / "results"

# dimer/boxsize5/benchmarkFPTcomparison/simMoriZwanzigFPTs_CO_*.xyz is mislabeled
# (its stats match the OC direction, not CO -- see tests/benchmarkFPTs/RESULTS.md).
# benchmarkFPTreference/ holds correctly-labeled CO/OC files derived directly from
# the raw benchmark trajectories (tests/benchmarkFPTs/compute_true_fpt.py).
DEFAULT_FPT_BENCH_DIR = f"{DEFAULT_OUTPUT_ROOT}/benchmarkFPTreference"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rollout diagnostics for a CVAE dimer simulation run."
    )
    parser.add_argument(
        "--sim-dir", required=True,
        help=(
            "Path relative to cvaeRuns/, e.g. "
            "local_dqipipimririm/001_base/100x10000 or "
            "pipimririm/002_run/50x5000_Tr=0.5"
        ),
    )
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCH_DIR)
    parser.add_argument("--n-trajs",       type=int, default=100,
                        help="Number of reduced trajectories to load.")
    parser.add_argument("--n-bench-trajs", type=int, default=100,
                        help="Number of benchmark trajectories to load.")
    parser.add_argument("--mtrajs-acf",    type=int, default=50,
                        help="Trajectories subsampled for ACF estimation.")
    parser.add_argument("--label",   default=None,
                        help="Short label for plot legends (default: sim-dir leaf name).")
    parser.add_argument("--dt",      type=float, default=None,
                        help="Output time step in ns. If omitted, read from parameters file.")
    parser.add_argument("--boxsize", type=float, default=5.0)
    parser.add_argument("--fpt", action="store_true", default=False,
                        help=(
                            "Also generate first-passage-time diagnostics vs the benchmark. "
                            "Default: core-set (entry-to-arrival) on the rollout Δx (cold-start-free)."
                        ))
    parser.add_argument("--fpt-only", action="store_true", default=False,
                        help=(
                            "Only generate first-passage-time diagnostics, skipping the standard "
                            "rollout diagnostics. Implies --fpt."
                        ))
    parser.add_argument("--fpt-external", action="store_true", default=False,
                        help=(
                            "Back-compat: ingest pre-computed FPT .xyz (e.g. reset-based "
                            "propagateFPT) via --fpt-sim-dir instead of the default core-set "
                            "method. propagateFPT numbers carry a cold-start bias."
                        ))
    parser.add_argument("--fpt-margin", type=float, default=0.3,
                        help="Core-set margin for the default FPT method (default 0.3).")
    parser.add_argument("--fpt-sim-dir", default=None,
                        help=(
                            "Required only with --fpt-external. Path relative to cvaeRuns/ to "
                            "the FPT simulation output directory for ONE transition type, e.g. "
                            "local_dqipipimririm/001_base/FPT_OC_10000x10000. The sibling directory "
                            "for the other transition type (same parent, same name with OC/CO "
                            "swapped) is auto-discovered and included if it exists."
                        ))
    parser.add_argument("--fpt-bench-dir", default=DEFAULT_FPT_BENCH_DIR,
                        help=(
                            "Directory containing the benchmark simMoriZwanzigFPTs_{OC,CO}_*.xyz "
                            "files. Defaults to the corrected reference (benchmarkFPTreference/); "
                            "the original dimer/boxsize5/benchmarkFPTcomparison/ 'CO' file is "
                            "mislabeled -- see tests/benchmarkFPTs/RESULTS.md."
                        ))
    parser.add_argument("--diag-dir", default=None,
                        help=(
                            "Non-standard output directory for the scoreboard (plots + summaries). "
                            "Default: results/<cond>/<id_name>/diags/<simrun>/. Use this for test "
                            "runs whose scoreboards should land elsewhere (e.g. tests/<campaign>/...)."
                        ))
    return parser.parse_args()


def main():
    args = parse_args()

    rel = Path(args.sim_dir)
    # Absolute --sim-dir is taken as-is (test rollouts outside cvaeRuns/);
    # relative paths keep the standard cvaeRuns/ convention.
    sim_dir = rel if rel.is_absolute() else CVAE_RUNS_ROOT / rel

    if not sim_dir.exists():
        print(f"Error: directory does not exist: {sim_dir}")
        sys.exit(1)

    if args.diag_dir:
        # Explicit non-standard output location (e.g. a tests/<campaign>/ folder).
        diag_dir = Path(args.diag_dir)
    elif not rel.is_absolute() and len(rel.parts) == 3:
        # Standard: results/<cond>/<id_name>/diags/<simrun>/
        cond, id_name, simrun = rel.parts
        diag_dir = RESULTS_ROOT / cond / id_name / "diags" / simrun
    else:
        # absolute sim-dir or unexpected depth — fall back alongside trajectories
        diag_dir = sim_dir / "diagnostics"

    label = args.label or sim_dir.name
    roll = None
    fpt_rows = None

    if not args.fpt_only:
        roll = run_all_diagnostics(
            sim_dir       = sim_dir,
            bench_dir     = Path(args.benchmark_dir),
            n_trajs       = args.n_trajs,
            n_bench_trajs = args.n_bench_trajs,
            mTrajs_acf    = args.mtrajs_acf,
            label         = label,
            boxsize       = args.boxsize,
            dt            = args.dt,
            diag_dir      = diag_dir,
        )
        print(f"\nDiagnostics saved to: {diag_dir}")

    if args.fpt or args.fpt_only:
        fpt_bench_dir = Path(args.fpt_bench_dir)

        if args.fpt_external:
            # ── Back-compat: ingest pre-computed FPT .xyz (reset-based propagateFPT). ──
            if not args.fpt_sim_dir:
                print("Error: --fpt-external requires --fpt-sim-dir, e.g. "
                      "--fpt-sim-dir local_dqipipimririm/001_base/FPT_OC_10000x10000")
                sys.exit(1)
            fpt_sim_dir = CVAE_RUNS_ROOT / Path(args.fpt_sim_dir)
            if not fpt_sim_dir.exists():
                print(f"Error: FPT sim directory does not exist: {fpt_sim_dir}")
                sys.exit(1)
            fpt_rows = run_fpt_diagnostics(
                fpt_sim_dir   = fpt_sim_dir,
                fpt_bench_dir = fpt_bench_dir,
                label         = label,
                diag_dir      = diag_dir,
            )
        else:
            # ── Default: core-set (entry-to-arrival) directly on the rollout Δx. ──
            if roll is not None:
                model_dx, dt = roll["dx"], roll["dt"]
            else:
                # --fpt-only: load the rollout Δx ourselves (no full diagnostics run).
                dt = args.dt
                if dt is None:
                    params = analysisTools.readParameters(str(sim_dir / "parameters"))
                    dt = float(params["dt"]) * int(float(params.get("stride", 1)))
                q, v, _ = load_trajectories(str(sim_dir / "simMoriZwanzigReduced_"), args.n_trajs)
                model_dx, _ = compute_dx_dvx(q, v, boxsize=args.boxsize)
            fpt_rows = run_fpt_diagnostics_coreset(
                model_dx      = model_dx,
                dt            = dt,
                fpt_bench_dir = fpt_bench_dir,
                label         = label,
                diag_dir      = diag_dir,
                margin        = args.fpt_margin,
            )
        print(f"\nFPT diagnostics saved to: {diag_dir}")

    # Combined "key diagnostics" figure (plot 16): needs both the rollout
    # arrays (02/03) and the FPT rows (14).
    if roll is not None and fpt_rows is not None:
        save_combined_key_diagnostics(roll, fpt_rows, diag_dir, label=label)
        print(f"\nCombined key-diagnostics figure (16) saved to: {diag_dir}")


if __name__ == "__main__":
    main()
