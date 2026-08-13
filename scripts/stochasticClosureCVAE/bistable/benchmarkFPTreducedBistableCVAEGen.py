"""
First-passage-time (FPT) estimator for reduced-bistable CVAE_SP rollouts, using
the noiseSampler/training config system (results/bistable/<cond>/<NNN>_<name>/).

Bistable analogue of benchmarkFPTreducedDimerCVAEGen.py: a single distinguished
particle in a double-well external potential; the FPT is the time to cross from
one well to the other. Mirrors benchmarkReducedBistableCVAEGen.py in model
loading / output-directory layout. Fixed defaults: boxsize=5, stride=1, device=cpu.

Following the project FPT rule, equilibration defaults to 0 (a nonzero
equilibration lets the particle leave the starting well before the clock starts,
biasing times toward zero).

Output layout
-------------
  cvaeRuns/<cond>/<id>_<config.name>/
      model_config.yaml                          <- provenance copy (shared, written once)
      FPT_<direction>_<N>x<tfinal>[_<tag>]/       <- simulation sub-dir
          parameters
          simMoriZwanzigFPTs_<direction>_<cond>_box<boxsize>_nsims<N>.xyz

Usage
-----
    python benchmarkFPTreducedBistableCVAEGen.py --cond piri --id 001
    python benchmarkFPTreducedBistableCVAEGen.py --cond piririm --id 001 \\
        --num-simulations 2000 --tfinal 20000 --direction RL
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
from deepRD.diffusionIntegrators import langevinNoiseSampler
from deepRD.noiseSampler.cvae.checkpoints import load_run
from deepRD.potentials import bistable

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
# Bistable models live under their own top-level folder: results/bistable/<cond>/...
RESULTS_ROOT = _REPO_ROOT / "deepRD" / "noiseSampler" / "training" / "results" / "bistable"

DEFAULT_BENCHMARK_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5/benchmark"
DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5"

BOXSIZE = 5.0
STRIDE = 1

# External bistable-potential geometry (shared across bistable benchmark scripts).
MINIMA_DIST = 1.5
KCONSTANTS = np.array([1.0, 1.0, 1.0])
SCALEFACTOR = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate reduced-bistable first passage times from a CVAE trained with the config system."
    )
    parser.add_argument("--cond", required=True,
                        help="Conditioning type matching the results sub-directory (e.g. piri, pipimri, piririm).")
    parser.add_argument("--id", required=True,
                        help="3-digit run-ID prefix to match results/bistable/<cond>/<id>_*/ (e.g. '001').")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-name", default=None,
                        help="Optional tag appended to the FPT simulation sub-directory name.")
    parser.add_argument("--num-simulations", type=int, default=2000)
    parser.add_argument("--tfinal", type=float, default=10000.0,
                        help="Max simulation time before a run is declared failed.")
    parser.add_argument("--equilibration-steps", type=int, default=0)
    parser.add_argument("--direction", choices=["LR", "RL"], default="LR",
                        help="LR: left well (-MINIMA_DIST) -> right well (+MINIMA_DIST). RL: the reverse.")
    parser.add_argument("--minima-threshold", type=float, default=0.3,
                        help="Distance tolerance to declare the target well reached.")
    parser.add_argument("--Tr", type=float, default=1.0, help="Decoder noise temperature.")
    parser.add_argument("--Tz", type=float, default=1.0, help="Latent prior temperature.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers. Defaults to cpu_count - 1.")
    return parser.parse_args()


def find_run_dir(cond: str, run_id: str) -> Path:
    """Return the unique run directory matching results/bistable/<cond>/<run_id>_*/."""
    cond_dir = RESULTS_ROOT / cond
    if not cond_dir.exists():
        raise FileNotFoundError(f"Conditioning directory not found: {cond_dir}")
    matches = [p for p in sorted(cond_dir.iterdir()) if p.is_dir() and p.name.startswith(run_id)]
    if not matches:
        raise FileNotFoundError(f"No run directory with prefix '{run_id}' in {cond_dir}.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous: multiple run directories match '{run_id}' in {cond_dir}: {matches}")
    return matches[0]


def resolve_output_dirs(args, run_dir: Path) -> tuple[Path, Path]:
    """Return (base_dir, sim_dir); base_dir mirrors the model directory name."""
    base_dir = Path(DEFAULT_OUTPUT_ROOT) / "cvaeRuns" / args.cond / run_dir.name
    run_tag = f"FPT_{args.direction}_{args.num_simulations}x{args.tfinal:g}"
    if args.output_name:
        run_tag = f"{run_tag}_{args.output_name}"
    return base_dir, base_dir / run_tag


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


def run_parallel_sim(simnumber, args_dict, parameters, initial_position, final_position):
    args = argparse.Namespace(**args_dict)
    set_seed(int(simnumber))
    torch.set_num_threads(1)

    config, model, _ = load_run(Path(args.run_dir), map_location="cpu")
    model.set_temps(Tr=args.Tr, Tz=args.Tz)
    conditioned_on = config.data.conditioning

    particle = deepRD.particle(list(initial_position), velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle_list = deepRD.particleList([particle])

    integrator = langevinNoiseSampler(
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
    integrator.setExternalPotential(bistable(MINIMA_DIST, KCONSTANTS, SCALEFACTOR))

    status, time = integrator.propagateFPT(particle_list, final_position, args.minima_threshold)
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
    args.run_dir = str(run_dir)

    config, _, _ = load_run(run_dir, map_location="cpu")
    conditioned_on = config.data.conditioning

    base_dir, sim_dir = resolve_output_dirs(args, run_dir)
    benchmark_dir = Path(args.benchmark_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(sim_dir, args.overwrite)

    parameters = analysisTools.readParameters(str(benchmark_dir / "parameters"))
    if float(parameters["boxsize"]) != BOXSIZE:
        raise ValueError(f"Benchmark boxsize {parameters['boxsize']} does not match expected {BOXSIZE}.")

    if args.direction == "LR":
        initial_position = np.array([-MINIMA_DIST, 0.0, 0.0])
        final_position = np.array([MINIMA_DIST, 0.0, 0.0])
    else:
        initial_position = np.array([MINIMA_DIST, 0.0, 0.0])
        final_position = np.array([-MINIMA_DIST, 0.0, 0.0])

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
        "direction": args.direction,
        "initialPosition": initial_position.tolist(),
        "finalPosition": final_position.tolist(),
        "minimaThreshold": args.minima_threshold,
    }
    analysisTools.writeParameters(str(sim_dir / "parameters"), parameter_dictionary)
    shutil.copy2(run_dir / "config.yaml", base_dir / "model_config.yaml")

    filename = sim_dir / (
        f"simMoriZwanzigFPTs_{args.direction}_{conditioned_on}"
        f"_box{parameters['boxsize']}_nsims{args.num_simulations}.xyz"
    )
    sim_numbers = list(range(args.start_index, args.start_index + args.num_simulations))
    num_workers = args.num_workers if args.num_workers is not None else max(multiprocessing.cpu_count() - 1, 1)

    print(f"FPT simulation for {args.direction} | {conditioned_on} begins ...")
    print(f"Model directory  : {run_dir}")
    print(f"Output directory : {sim_dir}")
    print(f"Model type       : {config.model.model_type}")
    print(f"Simulations      : {len(sim_numbers)}  |  tfinal: {args.tfinal}  |  equil: {args.equilibration_steps}")
    print(f"Initial pos      : {initial_position.tolist()}  |  Final pos: {final_position.tolist()}  |  threshold: {args.minima_threshold}")
    print(f"Workers          : {num_workers}")

    worker = partial(
        run_parallel_sim,
        args_dict=vars(args),
        parameters=parameters,
        initial_position=initial_position,
        final_position=final_position,
    )

    num_success, num_failed = multiprocessing_handler(sim_numbers, worker, filename, num_workers)

    print(f"\nDone. {num_success} successful, {num_failed} failed (out of {len(sim_numbers)}).")
    print(f"FPT data written to: {filename}")


if __name__ == "__main__":
    main()
