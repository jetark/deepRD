"""
End-to-end driver: train -> calibrate -> sweep+select -> generate rollout.

Chains the four pipeline stages for one seed with the frozen deep4 recipe,
reproducing the paper's E3 dimer noise sampler (default: seed 302, gain selected
on the velocity marginal, published choice ~0.45). Each stage runs as its own
process (``python -m ...``) so GPU training and CPU-multiprocess rollout stay
isolated and every stage is independently re-runnable.

Usage
-----
    # full reproduction from scratch (needs a GPU for the training stage)
    python -m deepRD.noiseSampler.e3cvae.reproduce.run_pipeline --seed 302

    # skip training and reuse an existing checkpoint (e.g. the shipped one)
    python -m deepRD.noiseSampler.e3cvae.reproduce.run_pipeline --seed 302 \
        --run-dir deepRD/noiseSampler/training/results_e3/ens_e3_s302 --skip-train
"""
import argparse
import subprocess
import sys
from pathlib import Path

PKG = "deepRD.noiseSampler.e3cvae.reproduce"
RESULTS_ROOT = Path("deepRD/noiseSampler/training/results_e3")


def run(stage, cmd):
    print(f"\n===== [{stage}] {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run([sys.executable, *[str(c) for c in cmd]], check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Model run dir (default: results_e3/repro_s<seed>).")
    ap.add_argument("--gains-path", type=Path, default=None,
                    help="Where to write/read fd_gains (default: <run-dir>/fd_gains.json).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-train", action="store_true",
                    help="Reuse an existing checkpoint in --run-dir.")
    ap.add_argument("--skip-calibrate", action="store_true")
    # sweep + selection
    ap.add_argument("--gain-list", type=float, nargs="+",
                    default=[0.0, 0.40, 0.42, 0.45, 0.47, 0.50])
    ap.add_argument("--select-tol", type=float, default=0.01)
    ap.add_argument("--sweep-sims", type=int, default=20)
    ap.add_argument("--sweep-tfinal", type=float, default=300.0)
    ap.add_argument("--sweep-equil", type=int, default=2500)
    # production rollout
    ap.add_argument("--gain", type=float, default=None,
                    help="Force the production gain (default: use the swept selection).")
    ap.add_argument("--num-simulations", type=int, default=60)
    ap.add_argument("--tfinal", type=float, default=8000.0)
    ap.add_argument("--equilibration-steps", type=int, default=5000)
    ap.add_argument("--output-name", default=None,
                    help="Rollout output dir name (default: repro_s<seed>_fd_g<gain>).")
    ap.add_argument("--run-name", default="60x8000")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()

    run_dir = args.run_dir or (RESULTS_ROOT / f"repro_s{args.seed}")
    gains_path = args.gains_path or (run_dir / "fd_gains.json")
    selected_file = run_dir / "selected_gain.txt"

    if not args.skip_train:
        run("train", ["-m", f"{PKG}.train", "--seed", args.seed,
                      "--out", run_dir, "--device", args.device])

    if not args.skip_calibrate:
        run("calibrate", ["-m", f"{PKG}.calibrate_gains", "--run-dir", run_dir,
                          "--out-gains", gains_path])

    if args.gain is None:
        run("sweep", ["-m", f"{PKG}.gain_sweep", "--run-dir", run_dir,
                      "--gains-path", gains_path, "--gain-list", *args.gain_list,
                      "--num-sims", args.sweep_sims, "--tfinal", args.sweep_tfinal,
                      "--equil", args.sweep_equil, "--workers", args.workers,
                      "--select-tol", args.select_tol,
                      "--write-selected", selected_file])
        gain = float(selected_file.read_text().strip())
        print(f"\n[pipeline] selected gain = {gain}")
    else:
        gain = args.gain
        print(f"\n[pipeline] using forced gain = {gain}")

    output_name = args.output_name or f"repro_s{args.seed}_fd_g{str(gain).replace('.', '')}"
    run("rollout", ["-m", f"{PKG}.generate_rollout", "--run-dir", run_dir,
                    "--gains-path", gains_path, "--gain", gain,
                    "--output-name", output_name, "--run-name", args.run_name,
                    "--num-simulations", args.num_simulations, "--tfinal", args.tfinal,
                    "--equilibration-steps", args.equilibration_steps,
                    "--num-workers", args.workers, "--overwrite"])
    print(f"\n[pipeline] DONE. seed={args.seed} gain={gain} "
          f"rollout=<output-root>/{output_name}/{args.run_name}")


if __name__ == "__main__":
    main()
