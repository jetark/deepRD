"""
FD-gain sweep + automatic selection on the velocity marginal.

Runs the real reduced-dimer integrator (langevinNoiseSamplerDimerGlobal +
pairBistable) with the FD-corrected E3 sampler for a list of scalar gains, using
the same cold-start + equilibration protocol as the production generator (so
gain=0 reproduces the baseline E3 rollout). For each gain it reports the
stationary velocity/relative-velocity/aux marginal std ratios against the
benchmark.

Selection criterion (paper protocol): pick the SMALLEST gain whose bond-velocity
marginal std ``dvx_std`` lands within ``--select-tol`` (default 1%) of the
benchmark. The velocity marginal is what Fix-A is calibrated to correct; the
first-passage kinetics are NOT consulted here (they are scored downstream on the
full rollout). This makes the choice reproducible without a human reading the
table.

Usage
-----
    python -m deepRD.noiseSampler.e3cvae.reproduce.gain_sweep \
        --run-dir deepRD/noiseSampler/training/results_e3/repro_s302 \
        --gains-path fd_gains_s302.json \
        --gain-list 0.0 0.40 0.42 0.45 0.47 0.50 \
        --num-sims 20 --tfinal 300 --equil 2500 --workers 11 \
        --write-selected selected_gain_s302.txt
"""
import argparse
import multiprocessing as mp
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


def minimal_image(rel, boxsize):
    return rel - boxsize * np.round(rel / boxsize)


def run_one(simnumber, run_dir, params, gain, gains_path, Tr, Tz, tfinal, equil, boxsize):
    torch.set_num_threads(1)
    np.random.seed(simnumber)
    torch.manual_seed(simnumber)
    device = torch.device("cpu")
    _, normalizer, model, _ = load_trained_model(Path(run_dir), device)
    sampler = E3DimerRolloutSamplerFD(model, normalizer, boxsize, device,
                                      Tr=Tr, Tz=Tz, gains_path=gains_path, gain=gain)

    diameter = 0.5
    x0 = 1.0 * diameter
    p1 = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=params["mass"])
    p2 = deepRD.particle([x0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=params["mass"])
    plist = deepRD.particleList([p1, p2])
    integ = langevinNoiseSamplerDimerGlobal(
        params["dt"], 1, tfinal, params["Gamma"], sampler, params["KbT"],
        boxsize, params["boundaryType"], equil, "E3_base")
    integ.setPairPotential(pairBistable(x0, 1.0 * diameter, 2))

    _, X, V, R = integ.propagate(plist, outputAux=True)
    X = np.asarray(X); V = np.asarray(V); R = np.asarray(R)
    rel = minimal_image(X[:, 1, :] - X[:, 0, :], boxsize)
    dx = np.linalg.norm(rel, axis=-1)
    e = rel / np.clip(dx[:, None], 1e-12, None)
    dvx = np.sum((V[:, 1, :] - V[:, 0, :]) * e, axis=-1)
    return {"v": V.reshape(-1, 3), "dx": dx, "dvx": dvx, "r": R.reshape(-1, 3)}


def benchmark_reference(dataset_dir, n_ref, n_total, boxsize):
    base = str(Path(dataset_dir) / "simMoriZwanzig_")
    V, DX, DVX, RC = [], [], [], []
    got = 0
    for fnum in range(n_total):
        if got >= n_ref:
            break
        try:
            traj = trajectoryTools.loadTrajectory(base, fnum)
        except Exception:
            continue
        q = np.stack([traj[0::2, 1:4], traj[1::2, 1:4]], axis=1)
        v = np.stack([traj[0::2, 4:7], traj[1::2, 4:7]], axis=1)
        r = np.stack([traj[0::2, 8:11], traj[1::2, 8:11]], axis=1)
        rel = minimal_image(q[:, 1, :] - q[:, 0, :], boxsize)
        dxb = np.linalg.norm(rel, axis=-1)
        e = rel / np.clip(dxb[:, None], 1e-12, None)
        V.append(v.reshape(-1, 3)); DX.append(dxb)
        DVX.append(np.sum((v[:, 1, :] - v[:, 0, :]) * e, axis=-1))
        RC.append(r.reshape(-1, 3)); got += 1
    return (np.concatenate(V), np.concatenate(DX), np.concatenate(DVX), np.concatenate(RC))


def sweep(run_dir, gains_path, gain_list, benchmark_dir, num_sims=20, tfinal=300.0,
          equil=2500, Tr=1.0, Tz=1.0, workers=6, n_ref=60):
    """Run the sweep; return (ref, {gain: (v_ratio, dvx_ratio, r_ratio, dxmean)})."""
    params = analysisTools.readParameters(str(Path(benchmark_dir) / "parameters"))
    boxsize = float(params["boxsize"])
    Vb, DXb, DVXb, RCb = benchmark_reference(benchmark_dir, n_ref, int(params["numFiles"]), boxsize)
    ref = {"vstd": float(Vb.std()), "dvxstd": float(DVXb.std()), "rstd": float(RCb.std()),
           "dxmean": float(DXb.mean()), "dxstd": float(DXb.std())}
    print(f"steps/sim ~ {int(tfinal/params['dt'])}  equil={equil}  num_sims={num_sims}  Tr={Tr}")
    print(f"{'BENCHMARK':>10} | v_std={ref['vstd']:.5f} (1.000x) | dvx_std={ref['dvxstd']:.5f} "
          f"(1.000x) | dx={ref['dxmean']:.4f} | r_std={ref['rstd']:.5f} (1.000x)")

    out = {}
    for gain in gain_list:
        worker = partial(run_one, run_dir=run_dir, params=params, gain=gain,
                         gains_path=gains_path, Tr=Tr, Tz=Tz, tfinal=tfinal,
                         equil=equil, boxsize=boxsize)
        if workers <= 1:
            outs = [worker(i) for i in range(num_sims)]
        else:
            with mp.Pool(workers) as pool:
                outs = pool.map(worker, list(range(num_sims)))
        V = np.concatenate([o["v"] for o in outs])
        DX = np.concatenate([o["dx"] for o in outs])
        DVX = np.concatenate([o["dvx"] for o in outs])
        RC = np.concatenate([o["r"] for o in outs])
        rec = {"v": float(V.std()) / ref["vstd"], "dvx": float(DVX.std()) / ref["dvxstd"],
               "r": float(RC.std()) / ref["rstd"], "dxmean": float(DX.mean())}
        out[gain] = rec
        print(f"{'gain='+format(gain, '.2f'):>10} | v_std={V.std():.5f} ({rec['v']:.3f}x) "
              f"| dvx_std={DVX.std():.5f} ({rec['dvx']:.3f}x) | dx={DX.mean():.4f} "
              f"| r_std={RC.std():.5f} ({rec['r']:.3f}x)")
    return ref, out


def select_gain(results, tol=0.01):
    """Smallest gain with |dvx_std/bench - 1| <= tol; else closest to 1.0."""
    ok = sorted(g for g, r in results.items() if abs(r["dvx"] - 1.0) <= tol)
    if ok:
        return ok[0], True
    closest = min(results, key=lambda g: abs(results[g]["dvx"] - 1.0))
    return closest, False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gains-path", required=True)
    ap.add_argument("--gain-list", type=float, nargs="+",
                    default=[0.0, 0.40, 0.42, 0.45, 0.47, 0.50])
    ap.add_argument("--num-sims", type=int, default=20)
    ap.add_argument("--tfinal", type=float, default=300.0)
    ap.add_argument("--equil", type=int, default=2500)
    ap.add_argument("--Tr", type=float, default=1.0)
    ap.add_argument("--Tz", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-ref", type=int, default=60)
    ap.add_argument("--select-tol", type=float, default=0.01,
                    help="Max |dvx_std/bench - 1| for a gain to qualify (default 1%).")
    ap.add_argument("--benchmark-dir",
                    default="/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark")
    ap.add_argument("--write-selected", default=None,
                    help="If set, write the selected gain (bare float) to this file.")
    args = ap.parse_args()

    _, results = sweep(args.run_dir, args.gains_path, args.gain_list, args.benchmark_dir,
                       num_sims=args.num_sims, tfinal=args.tfinal, equil=args.equil,
                       Tr=args.Tr, Tz=args.Tz, workers=args.workers, n_ref=args.n_ref)
    gain, within_tol = select_gain(results, tol=args.select_tol)
    flag = "within tol" if within_tol else f"NO gain within {args.select_tol:.0%}; closest"
    print(f"\n[select] gain = {gain:.2f}  ({flag}; dvx_std/bench = {results[gain]['dvx']:.3f}x, "
          f"v_std/bench = {results[gain]['v']:.3f}x)")
    if args.write_selected:
        Path(args.write_selected).write_text(f"{gain}\n")
        print(f"[select] wrote {gain} -> {args.write_selected}")


if __name__ == "__main__":
    main()
