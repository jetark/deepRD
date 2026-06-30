import argparse
import multiprocessing
import os
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
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal
from deepRD.noiseSampler.e3cvae.diagnostics import (
    E3DimerRolloutSampler,
    E3Lag2RolloutSampler,
    Lag2IntegratorGlobal,
    load_trained_model,
)
from deepRD.potentials import pairBistable


"""
Long reduced-dimer rollout generator for the e3nn E3DimerCVAE.

This mirrors scripts/stochasticClosureCVAE/benchmarkReducedDimerGen.py:
    - reads benchmark parameters from stochasticClosure/dimer/boxsize5/benchmark
    - starts from the same cold dimer initial condition
    - uses tfinal=10000, equilibrationSteps=10000, stride=1 by default
    - writes simMoriZwanzigReduced_* trajectories with auxiliary r output
    - important to use pariBistable potential (with no bias)

The E3 model is trained on dqpipimririm, but the integrator receives
conditionedOn="E3_base" because that branch returns the required 30D global
state:
    q1, q2, v1_n, v2_n, v1_nm1, v2_nm1, r1_n, r2_n, r1_nm1, r2_nm1.
"""


DEFAULT_RUN_DIR = (
    "deepRD/noiseSampler/training/results/"
    "e3_dimer_dqpipimririm_axial_20260617"
)
DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5"
DEFAULT_BENCHMARK_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate long reduced dimer rollouts with E3DimerCVAE.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Directory with config.yaml/checkpoint/normalizer.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-name", default="benchmarkReducedGen_E3DimerCVAE_dqpipimririm_axial")
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--boxsize", type=float, default=5.0)
    parser.add_argument("--tfinal", type=float, default=10000.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--equilibration-steps", type=int, default=10000)
    parser.add_argument("--device", default=None, help="Defaults to cuda if visible, else cpu.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--Tr", type=float, default=1.0, help="Decoder vector-noise temperature.")
    parser.add_argument("--Tz", type=float, default=1.0, help="Latent prior temperature.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--serial", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            print(f"Folder {output_dir} already exists. Previous data files might be overwritten. Continue, y/n?")
            proceed = input().strip().lower()
            if proceed != "y":
                sys.exit(0)
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def make_parameter_dictionary(args, parameters, conditioned_on):
    return {
        "numFiles": args.num_simulations,
        "dt": parameters["dt"],
        "Gamma": parameters["Gamma"],
        "KbT": parameters["KbT"],
        "mass": parameters["mass"],
        "tfinal": args.tfinal,
        "stride": args.stride,
        "boxsize": parameters["boxsize"],
        "boundaryType": parameters["boundaryType"],
        "equilibrationSteps": args.equilibration_steps,
        "conditionedOn": conditioned_on,
        "modelConditioning": "dqpipimririm",
        "modelType": "E3DimerCVAE",
        "runDir": str(args.run_dir),
        "Tr": args.Tr,
        "Tz": args.Tz,
    }


def build_sampler(run_dir, device, boxsize, Tr, Tz):
    config, normalizer, model, checkpoint = load_trained_model(Path(run_dir), device)
    lag2 = getattr(config.model, "lag2", False)
    sampler_cls = E3Lag2RolloutSampler if lag2 else E3DimerRolloutSampler
    sampler = sampler_cls(model, normalizer, boxsize, device, Tr=Tr, Tz=Tz)
    return sampler, config, checkpoint


def run_parallel_sim(simnumber, args_dict, parameters, basefilename):
    args = argparse.Namespace(**args_dict)
    device = resolve_device(args.device)
    set_seed(int(simnumber))

    # Keep each process from oversubscribing CPU threads inside small per-step torch calls.
    if device.type == "cpu":
        torch.set_num_threads(1)

    sampler, config, _ = build_sampler(args.run_dir, device, parameters["boxsize"], args.Tr, args.Tz)
    lag2 = getattr(config.model, "lag2", False)
    integrator_cls = Lag2IntegratorGlobal if lag2 else langevinNoiseSamplerDimerGlobal
    conditioned_on = "E3_lag2" if lag2 else "E3_base"

    particle_diameter = 0.5
    x0 = 1.0 * particle_diameter
    rad = 1.0 * particle_diameter
    scalefactor = 2

    particle1 = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle2 = deepRD.particle([x0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle_list = deepRD.particleList([particle1, particle2])

    integrator = integrator_cls(
        parameters["dt"],
        args.stride,
        args.tfinal,
        parameters["Gamma"],
        sampler,
        parameters["KbT"],
        parameters["boxsize"],
        parameters["boundaryType"],
        args.equilibration_steps,
        conditioned_on,
    )
    integrator.setPairPotential(pairBistable(x0, rad, scalefactor))

    t, X, V, Raux = integrator.propagate(particle_list, outputAux=True)
    traj = trajectoryTools.convert2trajectory(t, [X, V, Raux])
    trajectoryTools.writeTrajectory(traj, basefilename, simnumber)
    print(f"Simulation {simnumber}, done.")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_root) / args.output_name
    benchmark_dir = Path(args.benchmark_dir)
    # Detect lag2 from saved config so parameter file is correctly labelled.
    _cfg, _, _, _ = load_trained_model(Path(args.run_dir), torch.device("cpu"))
    lag2 = getattr(_cfg.model, "lag2", False)
    conditioned_on = "E3_lag2" if lag2 else "E3_base"
    del _cfg

    prepare_output_dir(output_dir, args.overwrite)
    parameters = analysisTools.readParameters(str(benchmark_dir / "parameters"))

    if float(args.boxsize) != float(parameters["boxsize"]):
        raise ValueError(
            f"Requested boxsize {args.boxsize} does not match benchmark boxsize "
            f"{parameters['boxsize']}"
        )

    parameterfilename = output_dir / "parameters"
    parameter_dictionary = make_parameter_dictionary(args, parameters, conditioned_on)
    analysisTools.writeParameters(str(parameterfilename), parameter_dictionary)

    # Save the trained E3 config beside the generated trajectories for provenance.
    run_dir = Path(args.run_dir)
    if (run_dir / "config.yaml").exists():
        shutil.copy2(run_dir / "config.yaml", output_dir / "e3_model_config.yaml")
    if (run_dir / "e3_normalizer.json").exists():
        shutil.copy2(run_dir / "e3_normalizer.json", output_dir / "e3_normalizer.json")

    basefilename = str(output_dir / "simMoriZwanzigReduced_")
    sim_numbers = list(range(args.start_index, args.start_index + args.num_simulations))

    if args.num_workers is None:
        if args.serial or device.type == "cuda":
            num_workers = 1
        else:
            num_workers = max(multiprocessing.cpu_count() - 1, 1)
    else:
        num_workers = args.num_workers

    print("Simulation for ri+1|E3_base with E3DimerCVAE begins ...")
    print(f"Output directory: {output_dir}")
    print(f"Run directory: {args.run_dir}")
    print(f"Device: {device}")
    print(f"Simulations: {len(sim_numbers)} | tfinal: {args.tfinal} | equilibration: {args.equilibration_steps}")
    print(f"Workers: {num_workers}")

    args_dict = vars(args)
    worker = partial(
        run_parallel_sim,
        args_dict=args_dict,
        parameters=parameters,
        basefilename=basefilename,
    )

    if num_workers == 1:
        for simnumber in sim_numbers:
            worker(simnumber)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(worker, sim_numbers)


if __name__ == "__main__":
    main()
