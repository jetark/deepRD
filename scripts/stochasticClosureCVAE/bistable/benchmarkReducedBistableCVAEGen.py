"""
Long reduced-bistable rollout generator for CVAE_SP models trained with the
noiseSampler/training config system (results/bistable/<cond>/<NNN>_<name>/).

Bistable analogue of benchmarkReducedDimerCVAEGen.py: single distinguished
particle in an external bistable (double-well) potential, ABOBA integration with
the noise term r sampled from a config-loaded CVAE. Fixed defaults:
boxsize=5, stride=1, device=cpu.

Output layout
-------------
  cvaeRuns/<cond>/<id>_<config.name>/
      model_config.yaml              <- provenance copy (always)
      <N>x<tfinal>/                  <- simulation sub-dir (no --output-name)
          parameters
          simMoriZwanzigReduced_*
      <N>x<tfinal>_<tag>/            <- simulation sub-dir (with --output-name)

Usage
-----
    python benchmarkReducedBistableCVAEGen.py --cond piri --id 001
    python benchmarkReducedBistableCVAEGen.py --cond piririm --id 001 \\
        --num-simulations 50 --tfinal 5000 --output-name "short_test"
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
import deepRD.tools.trajectoryTools as trajectoryTools
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
        description="Generate long reduced bistable rollouts from a CVAE trained with the config system."
    )
    parser.add_argument("--cond", required=True,
                        help="Conditioning type matching the results sub-directory (e.g. piri, pipimri, piririm).")
    parser.add_argument("--id", required=True,
                        help="3-digit run-ID prefix to match results/bistable/<cond>/<id>_*/ (e.g. '001').")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-name", default=None,
                        help="Optional tag appended to the simulation sub-directory name.")
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--tfinal", type=float, default=10000.0)
    parser.add_argument("--equilibration-steps", type=int, default=10000)
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
    run_tag = f"{args.num_simulations}x{args.tfinal:g}"
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


def run_parallel_sim(simnumber, args_dict, parameters, basefilename):
    args = argparse.Namespace(**args_dict)
    set_seed(int(simnumber))
    torch.set_num_threads(1)

    config, model, _ = load_run(Path(args.run_dir), map_location="cpu")
    model.set_temps(Tr=args.Tr, Tz=args.Tz)
    conditioned_on = config.data.conditioning

    particle = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
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

    t, X, V, Raux = integrator.propagate(particle_list, outputAux=True)
    traj = trajectoryTools.convert2trajectory(t, [X, V, Raux])
    trajectoryTools.writeTrajectory(traj, basefilename, simnumber)
    print(f"Simulation {simnumber}, done.")


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
    }
    analysisTools.writeParameters(str(sim_dir / "parameters"), parameter_dictionary)
    shutil.copy2(run_dir / "config.yaml", base_dir / "model_config.yaml")

    basefilename = str(sim_dir / "simMoriZwanzigReduced_")
    sim_numbers = list(range(args.start_index, args.start_index + args.num_simulations))
    num_workers = args.num_workers if args.num_workers is not None else max(multiprocessing.cpu_count() - 1, 1)

    print(f"Simulation for r_{{n+1}} | {conditioned_on} begins ...")
    print(f"Model directory  : {run_dir}")
    print(f"Output directory : {sim_dir}")
    print(f"Model type       : {config.model.model_type}")
    print(f"Simulations      : {len(sim_numbers)}  |  tfinal: {args.tfinal}  |  equil: {args.equilibration_steps}")
    print(f"Workers          : {num_workers}")

    worker = partial(run_parallel_sim, args_dict=vars(args), parameters=parameters, basefilename=basefilename)

    if num_workers == 1:
        for simnumber in sim_numbers:
            worker(simnumber)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(worker, sim_numbers)

    print(f"\nDone. Trajectories written to: {sim_dir}")


if __name__ == "__main__":
    main()
