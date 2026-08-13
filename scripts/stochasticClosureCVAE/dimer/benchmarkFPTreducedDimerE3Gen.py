"""
First-passage-time (FPT) estimator for reduced-dimer rollouts using the e3nn
E3DimerCVAE. Mirrors benchmarkFPTreducedDimerCVAEGen.py in FPT physics/setup,
and benchmarkReducedDimerE3Gen.py in E3 model loading / sampler construction.

The E3 model is trained on dqpipimririm, but the integrator receives
conditionedOn="E3_base" (or "E3_lag2" for lag2 models) because that branch
returns the required 30D/42D global state -- see benchmarkReducedDimerE3Gen.py.

Important: uses the unbiased pairBistable potential (not pairBistableBias), to
match how the benchmark data was generated (see tests/labeling_diagnostics).

Note: imports tests/e3_improvement/config_compat.py, a runtime monkey-patch
(no deepRD/ source touched) working around a real bug where load_config()
requires config.yaml fields (n_lags, cond_dim) that predate every existing
E3 checkpoint's saved config -- see tests/e3_improvement/ANALYSIS.md.

Output layout
-------------
  cvaeRuns/<cond>/<id>_<name>/
      e3_model_config.yaml / e3_normalizer.json   <- provenance copy (shared, written once)
      FPT_<transitionType>_<N>x<tfinal>/                 <- simulation sub-dir (no --output-name)
      FPT_<transitionType>_<N>x<tfinal>_<tag>/           <- simulation sub-dir (with --output-name)
          parameters
          simMoriZwanzigFPTs_<transitionType>_<cond>_box<boxsize>_nsims<N>.xyz

Usage
-----
    python benchmarkFPTreducedDimerE3Gen.py \\
        --run-dir deepRD/noiseSampler/training/results_e3/e3_dimer_dqpipimririm_axial_20260617 \\
        --output-root /group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5 \\
        --output-name e3_dqipipimririm/001_base \\
        --num-simulations 10000 --tfinal 10000 --transition-type CO --Tr 0.85
"""
import argparse
import multiprocessing
import random
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "e3_improvement"))
import config_compat  # noqa: E402,F401  (monkey-patches load_config, see module docstring)

import deepRD
import deepRD.tools.analysisTools as analysisTools
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal
from deepRD.noiseSampler.e3cvae.diagnostics import (
    E3DimerRolloutSampler,
    E3Lag2RolloutSampler,
    Lag2IntegratorGlobal,
    load_trained_model,
)
from deepRD.potentials import pairBistable

DEFAULT_RUN_DIR = (
    "deepRD/noiseSampler/training/results_e3/"
    "e3_dimer_dqpipimririm_axial_20260617"
)
DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5"
DEFAULT_BENCHMARK_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark"

BOXSIZE = 5.0
STRIDE = 1

# Pair-potential geometry (shared across all dimer benchmark/CVAE/E3 scripts)
PARTICLE_DIAMETER = 0.5
X0 = 1.0 * PARTICLE_DIAMETER    # closed-state (first minima) separation
RAD = 1.0 * PARTICLE_DIAMETER   # half the distance between minima; open state at X0 + 2*RAD = 1.5
SCALEFACTOR = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate reduced-dimer first passage times from a trained E3DimerCVAE."
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                        help="Directory with config.yaml/checkpoint.pt/e3_normalizer.json.")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-name", required=True,
        help="Sub-path under output-root/cvaeRuns/, e.g. e3_dqipipimririm/001_base "
             "(mirrors the model directory, shared with the trajectory rollout).",
    )
    parser.add_argument("--num-simulations", type=int, default=10000)
    parser.add_argument("--tfinal", type=float, default=10000.0,
                        help="Max simulation time before a run is declared failed.")
    parser.add_argument("--equilibration-steps", type=int, default=0)
    parser.add_argument("--transition-type", choices=["CO", "OC"], default="CO",
                        help="CO: closed->open (first minima to second). OC: open->closed.")
    parser.add_argument("--minima-threshold", type=float, default=0.05,
                        help="Distance tolerance to declare the final separation reached.")
    parser.add_argument("--Tr", type=float, default=1.0, help="Decoder vector-noise temperature.")
    parser.add_argument("--Tz", type=float, default=1.0, help="Latent prior temperature.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu", help="Rollouts use CPU so multiprocessing can spawn freely.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers. Defaults to cpu_count - 1.")
    return parser.parse_args()


def resolve_output_dirs(args) -> tuple[Path, Path]:
    """
    Return (base_dir, sim_dir).

    base_dir : cvaeRuns/<output-name>/         — mirrors the model directory,
               shared with the trajectory rollouts from benchmarkReducedDimerE3Gen.py.
    sim_dir  : base_dir/FPT_<type>_<N>x<tfinal>_<tag>/    — only when --output-name is given
               base_dir/FPT_<type>_<N>x<tfinal>/          — otherwise
    """
    base_dir = Path(args.output_root) / "cvaeRuns" / args.output_name
    tfinal_str = f"{args.tfinal:g}"
    run_tag = f"FPT_{args.transition_type}_{args.num_simulations}x{tfinal_str}"
    sim_dir = base_dir / run_tag
    return base_dir, sim_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            print(f"Folder {output_dir} already exists. Previous files may be overwritten. Continue? y/n")
            if input().strip().lower() != "y":
                sys.exit(0)
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_sampler(run_dir, device, boxsize, Tr, Tz):
    config, normalizer, model, checkpoint = load_trained_model(Path(run_dir), device)
    lag2 = getattr(config.model, "lag2", False)
    sampler_cls = E3Lag2RolloutSampler if lag2 else E3DimerRolloutSampler
    sampler = sampler_cls(model, normalizer, boxsize, device, Tr=Tr, Tz=Tz)
    return sampler, config, checkpoint


def run_parallel_sim(simnumber, args_dict, parameters, initial_separation, final_separation):
    args = argparse.Namespace(**args_dict)
    device = torch.device(args.device)
    set_seed(int(simnumber))
    if device.type == "cpu":
        torch.set_num_threads(1)

    sampler, config, _ = build_sampler(args.run_dir, device, parameters["boxsize"], args.Tr, args.Tz)
    lag2 = getattr(config.model, "lag2", False)
    integrator_cls = Lag2IntegratorGlobal if lag2 else langevinNoiseSamplerDimerGlobal
    conditioned_on = "E3_lag2" if lag2 else "E3_base"

    particle1 = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle2 = deepRD.particle([initial_separation, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle_list = deepRD.particleList([particle1, particle2])

    integrator = integrator_cls(
        parameters["dt"],
        STRIDE,
        args.tfinal,
        parameters["Gamma"],
        sampler,
        parameters["KbT"],
        parameters["boxsize"],
        parameters["boundaryType"],
        args.equilibration_steps,
        conditioned_on,
    )
    integrator.setPairPotential(pairBistable(X0, RAD, SCALEFACTOR))

    status, time = integrator.propagateFPT(particle_list, initial_separation, final_separation, args.minima_threshold)
    return status, time


def multiprocessing_handler(sim_numbers, worker, filename, num_workers):
    num_success = 0
    num_failed = 0
    with open(filename, "w") as file, multiprocessing.Pool(processes=num_workers) as pool:
        for index, (status, time) in zip(sim_numbers, pool.imap(worker, sim_numbers)):
            if status == "success":
                file.write(str(time) + "\n")
                file.flush()
                num_success += 1
                print(f"Simulation {index}, done. Success!")
            else:
                num_failed += 1
                print(f"Simulation {index}, done. Failed :(")
    return num_success, num_failed


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    _cfg, _, _, _ = load_trained_model(run_dir, torch.device("cpu"))
    lag2 = getattr(_cfg.model, "lag2", False)
    del _cfg

    base_dir, sim_dir = resolve_output_dirs(args)
    benchmark_dir = Path(args.benchmark_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(sim_dir, args.overwrite)

    parameters = analysisTools.readParameters(str(benchmark_dir / "parameters"))
    if float(parameters["boxsize"]) != BOXSIZE:
        raise ValueError(f"Benchmark boxsize {parameters['boxsize']} does not match expected {BOXSIZE}.")

    if args.transition_type == "CO":
        initial_separation = X0
        final_separation = X0 + 2.0 * RAD
    else:
        initial_separation = X0 + 2.0 * RAD
        final_separation = X0

    parameter_dictionary = {
        "numFiles": args.num_simulations,
        "dt": parameters["dt"],
        "Gamma": parameters["Gamma"],
        "KbT": parameters["KbT"],
        "mass": parameters["mass"],
        "tfinal": args.tfinal,
        "stride": STRIDE,
        "boxsize": parameters["boxsize"],
        "boundaryType": parameters["boundaryType"],
        "equilibrationSteps": args.equilibration_steps,
        "conditionedOn": "E3_lag2" if lag2 else "E3_base",
        "modelConditioning": "dqpipimririm",
        "modelType": "E3DimerCVAE",
        "runDir": str(run_dir),
        "Tr": args.Tr,
        "Tz": args.Tz,
        "transitionType": args.transition_type,
        "initialSeparation": initial_separation,
        "finalSeparation": final_separation,
        "minimaThreshold": args.minima_threshold,
    }
    analysisTools.writeParameters(str(sim_dir / "parameters"), parameter_dictionary)

    if (run_dir / "config.yaml").exists():
        shutil.copy2(run_dir / "config.yaml", base_dir / "e3_model_config.yaml")
    if (run_dir / "e3_normalizer.json").exists():
        shutil.copy2(run_dir / "e3_normalizer.json", base_dir / "e3_normalizer.json")

    filename = sim_dir / (
        f"simMoriZwanzigFPTs_{args.transition_type}_dqpipimririm"
        f"_box{int(parameters['boxsize'])}_nsims{args.num_simulations}.xyz"
    )
    sim_numbers = list(range(args.start_index, args.start_index + args.num_simulations))
    num_workers = args.num_workers if args.num_workers is not None else max(multiprocessing.cpu_count() - 1, 1)

    print(f"E3 FPT simulation for {args.transition_type} | dqpipimririm begins ...")
    print(f"Model directory  : {run_dir}")
    print(f"Output directory : {sim_dir}")
    print(f"Simulations      : {len(sim_numbers)}  |  tfinal: {args.tfinal}  |  equil: {args.equilibration_steps}")
    print(f"Initial sep      : {initial_separation}  |  Final sep: {final_separation}  |  threshold: {args.minima_threshold}")
    print(f"Tr={args.Tr}  Tz={args.Tz}  |  Workers: {num_workers}")

    worker = partial(
        run_parallel_sim,
        args_dict=vars(args),
        parameters=parameters,
        initial_separation=initial_separation,
        final_separation=final_separation,
    )

    num_success, num_failed = multiprocessing_handler(sim_numbers, worker, filename, num_workers)

    print(f"\nDone. {num_success} successful, {num_failed} failed (out of {len(sim_numbers)}).")
    print(f"FPT data written to: {filename}")


if __name__ == "__main__":
    main()
