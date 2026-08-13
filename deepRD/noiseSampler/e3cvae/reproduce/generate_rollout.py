"""
Long reduced-dimer rollout generator for the FD-corrected E3DimerCVAE.

Mirrors ``scripts/stochasticClosureCVAE/dimer/benchmarkReducedDimerE3Gen.py`` but
swaps in the fluctuation-dissipation sampler (``E3DimerRolloutSamplerFD``). The
output directory (``<output-root>/<output-name>/<run-name>/`` with a
``parameters`` file and ``simMoriZwanzigReduced_*`` trajectories) is exactly what
the standard ``dimerRolloutDiagnostics.py`` / FPT tooling consumes, for an
apples-to-apples comparison against the benchmark.

Each worker re-seeds NumPy/torch from its simulation index, so a given
``--num-simulations`` reproduces the same set of trajectories.

Usage
-----
    python -m deepRD.noiseSampler.e3cvae.reproduce.generate_rollout \
        --run-dir deepRD/noiseSampler/training/results_e3/repro_s302 \
        --gains-path fd_gains_s302.json --gain 0.45 \
        --output-name repro_s302_fd_g045 --run-name 60x8000 \
        --num-simulations 60 --tfinal 8000 --equilibration-steps 5000 \
        --num-workers 11 --overwrite
"""
import argparse
import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch

import deepRD
import deepRD.tools.analysisTools as analysisTools
import deepRD.tools.trajectoryTools as trajectoryTools
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal
from deepRD.noiseSampler.e3cvae.diagnostics import load_trained_model
from deepRD.noiseSampler.e3cvae.reproduce.fd_sampler import E3DimerRolloutSamplerFD
from deepRD.potentials import pairBistable


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gains-path", required=True)
    p.add_argument("--gain", type=float, default=0.45)
    p.add_argument("--output-root",
                   default="/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5/cvaeRuns")
    p.add_argument("--output-name", default="repro_s302_fd")
    p.add_argument("--run-name", default="60x8000")
    p.add_argument("--benchmark-dir",
                   default="/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark")
    p.add_argument("--num-simulations", type=int, default=60)
    p.add_argument("--boxsize", type=float, default=5.0)
    p.add_argument("--tfinal", type=float, default=8000.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--equilibration-steps", type=int, default=5000)
    p.add_argument("--Tr", type=float, default=1.0)
    p.add_argument("--Tz", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run_sim(simnumber, args_dict, params, basefilename):
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(1)
    np.random.seed(int(simnumber))
    torch.manual_seed(int(simnumber))
    device = torch.device("cpu")
    _, normalizer, model, _ = load_trained_model(Path(args.run_dir), device)
    sampler = E3DimerRolloutSamplerFD(model, normalizer, params["boxsize"], device,
                                      Tr=args.Tr, Tz=args.Tz,
                                      gains_path=args.gains_path, gain=args.gain)
    diameter = 0.5
    x0 = 1.0 * diameter
    p1 = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=params["mass"])
    p2 = deepRD.particle([x0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=params["mass"])
    plist = deepRD.particleList([p1, p2])
    integ = langevinNoiseSamplerDimerGlobal(
        params["dt"], args.stride, args.tfinal, params["Gamma"], sampler, params["KbT"],
        params["boxsize"], params["boundaryType"], args.equilibration_steps, "E3_base")
    integ.setPairPotential(pairBistable(x0, 1.0 * diameter, 2))
    t, X, V, R = integ.propagate(plist, outputAux=True)
    traj = trajectoryTools.convert2trajectory(t, [X, V, R])
    trajectoryTools.writeTrajectory(traj, basefilename, simnumber)
    print(f"sim {simnumber} done", flush=True)


def main():
    args = parse_args()
    out_dir = Path(args.output_root) / args.output_name / args.run_name
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = analysisTools.readParameters(str(Path(args.benchmark_dir) / "parameters"))

    pd = {
        "numFiles": args.num_simulations, "dt": params["dt"], "Gamma": params["Gamma"],
        "KbT": params["KbT"], "mass": params["mass"], "tfinal": args.tfinal,
        "stride": args.stride, "boxsize": params["boxsize"],
        "boundaryType": params["boundaryType"], "equilibrationSteps": args.equilibration_steps,
        "conditionedOn": "E3_base", "modelConditioning": "dqpipimririm",
        "modelType": "E3DimerCVAE_FD", "runDir": str(args.run_dir),
        "Tr": args.Tr, "Tz": args.Tz, "fd_gain": args.gain,
    }
    analysisTools.writeParameters(str(out_dir / "parameters"), pd)

    basefilename = str(out_dir / "simMoriZwanzigReduced_")
    print(f"Generating {args.num_simulations} FD rollouts (gain={args.gain}) -> {out_dir}")
    worker = partial(run_sim, args_dict=vars(args), params=params, basefilename=basefilename)
    nums = list(range(args.num_simulations))
    if args.num_workers <= 1:
        for n in nums:
            worker(n)
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            pool.map(worker, nums)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
