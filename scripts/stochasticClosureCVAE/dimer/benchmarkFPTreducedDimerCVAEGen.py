"""
First-passage-time (FPT) estimator for reduced-dimer CVAE rollouts, using the
noiseSampler/training config system (results/<cond>/<NNN>_<name>/).

Mirrors benchmarkFPTreducedDimerGen.py in FPT physics/setup, and
benchmarkReducedDimerCVAEGen.py in model loading / output-directory layout.
Fixed defaults: boxsize=5, stride=1, device=cpu.

Output layout
-------------
  cvaeRuns/<cond>/<id>_<config.name>/
      model_config.yaml                          <- provenance copy (shared, written once)
      FPT_<transitionType>_<N>x<tfinal>/                 <- simulation sub-dir (no --output-name)
      FPT_<transitionType>_<N>x<tfinal>_<tag>/           <- simulation sub-dir (with --output-name)
          parameters
          simMoriZwanzigFPTs_<transitionType>_<cond>_box<boxsize>_nsims<N>.xyz

Usage
-----
    python benchmarkFPTreducedDimerCVAEGen.py --cond pipimririm --id 001
    python benchmarkFPTreducedDimerCVAEGen.py --cond local_dqidpipimmrimm --id 001 \\
        --num-simulations 500 --tfinal 20000 --transition-type OC
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

import deepRD
import deepRD.tools.analysisTools as analysisTools
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal
from deepRD.noiseSampler.cvae.checkpoints import load_run
from deepRD.potentials import pairBistable

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
RESULTS_ROOT = _REPO_ROOT / "deepRD" / "noiseSampler" / "training" / "results"

DEFAULT_BENCHMARK_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark"
DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5"

BOXSIZE = 5.0
STRIDE = 1

# Pair-potential geometry (shared across all dimer benchmark/CVAE scripts)
PARTICLE_DIAMETER = 0.5
X0 = 1.0 * PARTICLE_DIAMETER    # closed-state (first minima) separation
RAD = 1.0 * PARTICLE_DIAMETER   # half the distance between minima
SCALEFACTOR = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate reduced-dimer first passage times from a CVAE trained with the config system."
    )
    parser.add_argument(
        "--cond", required=True,
        help="Conditioning type matching the results sub-directory (e.g. pipimririm, local_dqidpipimmrimm).",
    )
    parser.add_argument(
        "--id", required=True,
        help="3-digit run-ID prefix to match results/<cond>/<id>_*/ (e.g. '001').",
    )
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--output-name", default=None,
        help=(
            "Optional tag appended to the simulation sub-directory name. "
            "Sub-directory is always FPT_<type>_<N>x<tfinal>/ (no tag) or "
            "FPT_<type>_<N>x<tfinal>_<tag>/ (with tag)."
        ),
    )
    parser.add_argument("--num-simulations", type=int, default=10000)
    parser.add_argument("--tfinal", type=float, default=10000.0,
                        help="Max simulation time before a run is declared failed.")
    parser.add_argument("--equilibration-steps", type=int, default=0)
    parser.add_argument("--transition-type", choices=["CO", "OC"], default="CO",
                        help="CO: closed->open (first minima to second). OC: open->closed.")
    parser.add_argument("--minima-threshold", type=float, default=0.05,
                        help="Distance tolerance to declare the final separation reached.")
    parser.add_argument("--Tr", type=float, default=1.0, help="Decoder noise temperature.")
    parser.add_argument("--Tz", type=float, default=1.0, help="Latent prior temperature.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers. Defaults to cpu_count - 1.")
    return parser.parse_args()


def find_run_dir(cond: str, run_id: str) -> Path:
    """Return the unique run directory matching results/<cond>/<run_id>_*/."""
    cond_dir = RESULTS_ROOT / cond
    if not cond_dir.exists():
        raise FileNotFoundError(f"Conditioning directory not found: {cond_dir}")
    matches = [p for p in sorted(cond_dir.iterdir()) if p.is_dir() and p.name.startswith(run_id)]
    if not matches:
        raise FileNotFoundError(f"No run directory with prefix '{run_id}' in {cond_dir}.")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous: multiple run directories match '{run_id}' in {cond_dir}: {matches}"
        )
    return matches[0]


def resolve_output_dirs(args, run_dir: Path) -> tuple[Path, Path]:
    """
    Return (base_dir, sim_dir).

    base_dir : cvaeRuns/<cond>/<run_dir.name>/           — mirrors the model directory,
               shared with the trajectory rollouts produced by benchmarkReducedDimerCVAEGen.py.
    sim_dir  : base_dir/FPT_<type>_<N>x<tfinal>_<tag>/    — only when --output-name is given
               base_dir/FPT_<type>_<N>x<tfinal>/          — otherwise
    """
    base_dir = Path(DEFAULT_OUTPUT_ROOT) / "cvaeRuns" / args.cond / run_dir.name
    tfinal_str = f"{args.tfinal:g}"
    run_tag = f"FPT_{args.transition_type}_{args.num_simulations}x{tfinal_str}"
    if args.output_name:
        run_tag = f"{run_tag}_{args.output_name}"
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


def run_parallel_sim(simnumber, args_dict, parameters, initial_separation, final_separation):
    args = argparse.Namespace(**args_dict)
    set_seed(int(simnumber))
    torch.set_num_threads(1)

    config, model, _ = load_run(Path(args.run_dir), map_location="cpu")
    model.set_temps(Tr=args.Tr, Tz=args.Tz)
    conditioned_on = config.data.conditioning

    particle1 = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle2 = deepRD.particle([initial_separation, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle_list = deepRD.particleList([particle1, particle2])

    integrator = langevinNoiseSamplerDimerGlobal(
        parameters["dt"],
        STRIDE,
        args.tfinal,
        parameters["Gamma"],
        model,
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
    """Runs FPT simulations in parallel and writes successful passage times to file."""
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

    run_dir = find_run_dir(args.cond, args.id)
    args.run_dir = str(run_dir)  # absolute path for worker subprocesses

    config, _, _ = load_run(run_dir, map_location="cpu")
    conditioned_on = config.data.conditioning

    base_dir, sim_dir = resolve_output_dirs(args, run_dir)
    benchmark_dir = Path(args.benchmark_dir)

    # Create base dir silently (shared across simulation runs of this model).
    base_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(sim_dir, args.overwrite)

    parameters = analysisTools.readParameters(str(benchmark_dir / "parameters"))

    if float(parameters["boxsize"]) != BOXSIZE:
        raise ValueError(
            f"Benchmark boxsize {parameters['boxsize']} does not match expected {BOXSIZE}."
        )

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
        "conditionedOn": conditioned_on,
        "modelType": config.model.model_type,
        "runDir": str(run_dir),
        "Tr": args.Tr,
        "Tz": args.Tz,
        "transitionType": args.transition_type,
        "initialSeparation": initial_separation,
        "finalSeparation": final_separation,
        "minimaThreshold": args.minima_threshold,
    }
    analysisTools.writeParameters(str(sim_dir / "parameters"), parameter_dictionary)
    # Provenance copy lives in base_dir (shared, written once).
    shutil.copy2(run_dir / "config.yaml", base_dir / "model_config.yaml")

    filename = sim_dir / (
        f"simMoriZwanzigFPTs_{args.transition_type}_{conditioned_on}"
        f"_box{parameters['boxsize']}_nsims{args.num_simulations}.xyz"
    )
    sim_numbers = list(range(args.start_index, args.start_index + args.num_simulations))

    num_workers = args.num_workers if args.num_workers is not None else max(multiprocessing.cpu_count() - 1, 1)

    print(f"FPT simulation for {args.transition_type} | {conditioned_on} begins ...")
    print(f"Model directory  : {run_dir}")
    print(f"Output directory : {sim_dir}")
    print(f"Model type       : {config.model.model_type}")
    print(f"Simulations      : {len(sim_numbers)}  |  tfinal: {args.tfinal}  |  equil: {args.equilibration_steps}")
    print(f"Initial sep      : {initial_separation}  |  Final sep: {final_separation}  |  threshold: {args.minima_threshold}")
    print(f"Workers          : {num_workers}")

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
