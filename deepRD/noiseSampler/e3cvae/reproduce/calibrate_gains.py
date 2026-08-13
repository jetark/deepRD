"""
Teacher-forced fluctuation-dissipation calibration for a trained E3DimerCVAE.

Estimates the per-channel friction *deficit* the model must have added back at
rollout time to match the benchmark noise-velocity coupling ("Fix-A"). For each
held-out benchmark transition n -> n+1 we feed the model the TRUE conditioning
(q, v_n, v_{n-1}, r_n, r_{n-1}), draw its next-step auxiliary noise several times,
and regress the conditional-mean noise onto the driving velocity in three
channels:

    relative-axial : dr_par  = (r1-r2).e   vs  dvx = (v1-v2).e
    relative-perp  : dr_perp                vs  dv_perp
    com-isotropic  : sr = r1+r2             vs  sv = v1+v2

The slope (Cov / Var) is the effective per-step friction the noise supplies. The
gain written for each channel is ``gamma_add = slope_benchmark - slope_model``,
i.e. exactly the friction the FD sampler adds. Output JSON is consumed by
``fd_sampler.E3DimerRolloutSamplerFD`` / ``gain_sweep.py`` / ``generate_rollout.py``.

The held-out split reuses the training seed, so calibration sees only data the
model was never trained on. No source files are modified.

Usage
-----
    python -m deepRD.noiseSampler.e3cvae.reproduce.calibrate_gains \
        --run-dir deepRD/noiseSampler/training/results_e3/repro_s302 \
        --out-gains fd_gains_s302.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepRD.noiseSampler.cvae.datasets import extract_vars, load_datasets
from deepRD.noiseSampler.e3cvae.diagnostics import (
    load_trained_model, split_by_trajectory)
from deepRD.noiseSampler.e3cvae.tools import (
    build_dimer_graph_batch, construct_dqpipimririm_tensors)
from deepRD.noiseSampler.e3cvae.training import move_graph_batch_to_device


def minimal_image(rel, boxsize):
    return rel - boxsize * np.round(rel / boxsize)


def axis_unit(q1, q2, boxsize):
    rel = minimal_image(q2 - q1, boxsize)
    n = np.linalg.norm(rel, axis=-1, keepdims=True)
    return rel / np.clip(n, 1e-12, None)


@torch.no_grad()
def model_sample_batch(model, normalizer, flat, sl, boxsize, device, Tr, Tz, n_draws):
    """Teacher-forced draws of model r_{n+1} for a slice of transitions.

    Returns (r1, r2) physical noise, each shape [n_draws, N_slice, 3].
    """
    def sc_v(x):
        return torch.tensor(x[sl] / normalizer.velocity_scale, dtype=torch.float32)

    def sc_r(x):
        return torch.tensor(x[sl] / normalizer.auxiliary_scale, dtype=torch.float32)

    feat = model.graph_featurisation() if hasattr(model, "graph_featurisation") else {}
    batch = build_dimer_graph_batch(
        q1=torch.tensor(flat["q1"][sl], dtype=torch.float32),
        q2=torch.tensor(flat["q2"][sl], dtype=torch.float32),
        v1=sc_v(flat["v1_n"]), v2=sc_v(flat["v2_n"]),
        r1=sc_r(flat["r1_n"]), r2=sc_r(flat["r2_n"]),
        v1_prev=sc_v(flat["v1_nm1"]), v2_prev=sc_v(flat["v2_nm1"]),
        r1_prev=sc_r(flat["r1_nm1"]), r2_prev=sc_r(flat["r2_nm1"]),
        boxsize=boxsize, **feat)
    batch = move_graph_batch_to_device(batch, device)
    outs = []
    for _ in range(n_draws):
        r_norm, _, _ = model.sample_torch(batch, Tr=Tr, Tz=Tz)  # [2N,3] normalised
        r_phys = (r_norm * normalizer.auxiliary_scale).cpu().numpy().reshape(-1, 2, 3)
        outs.append(r_phys)
    outs = np.stack(outs, axis=0)              # [n_draws, N, 2, 3]
    return outs[:, :, 0, :], outs[:, :, 1, :]


def calibrate(run_dir, max_samples=60000, n_draws=8, Tr=1.0, Tz=1.0, batch=8192,
              sample_seed=0, device=None):
    """Return the friction-gain dict for the model in ``run_dir``."""
    device = device or torch.device("cpu")
    config, normalizer, model, ckpt = load_trained_model(Path(run_dir), device)
    boxsize = config.system.boxsize

    # Held-out benchmark transitions, using the SAME split as training.
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)
    raw, _ = load_datasets(config.data.dataset_dir,
                           n_datasets=config.data.train_trajectories,
                           n_total=config.data.total_trajectories, trajtype="bench")
    q, v, r = extract_vars(raw)
    structured = construct_dqpipimririm_tensors(q, v, r)
    _, val = split_by_trajectory(structured, config.data.val_fraction)
    flat = {k: (v_.reshape(-1, 3).numpy() if torch.is_tensor(v_)
                else np.asarray(v_).reshape(-1, 3)) for k, v_ in val.items()}

    N = flat["q1"].shape[0]
    idx = np.random.default_rng(sample_seed).permutation(N)[:min(max_samples, N)]
    for k in flat:
        flat[k] = flat[k][idx]
    N = flat["q1"].shape[0]

    e = axis_unit(flat["q1"], flat["q2"], boxsize)
    dv = flat["v1_n"] - flat["v2_n"]
    dvx = np.sum(dv * e, axis=-1)
    dv_perp = dv - dvx[:, None] * e
    sv = flat["v1_n"] + flat["v2_n"]

    # TRUE next-step noise
    dr_true = flat["r1_next"] - flat["r2_next"]
    dr_par_true = np.sum(dr_true * e, axis=-1)
    dr_perp_true_vec = dr_true - dr_par_true[:, None] * e
    sr_true = flat["r1_next"] + flat["r2_next"]

    # MODEL next-step noise (teacher-forced), averaged over draws -> conditional mean
    r1d, r2d = [], []
    for s in range(0, N, batch):
        sl = slice(s, min(s + batch, N))
        a1, a2 = model_sample_batch(model, normalizer, flat, sl, boxsize, device,
                                    Tr, Tz, n_draws)
        r1d.append(a1)
        r2d.append(a2)
    r1_mean = np.concatenate(r1d, axis=1).mean(axis=0)   # [N,3]
    r2_mean = np.concatenate(r2d, axis=1).mean(axis=0)
    dr_mean = r1_mean - r2_mean
    dr_par_mean = np.sum(dr_mean * e, axis=-1)
    dr_perp_mean_vec = dr_mean - dr_par_mean[:, None] * e
    sr_model = r1_mean + r2_mean

    # Friction slopes (Cov / Var) per channel, benchmark vs model.
    def slope(y, x_scalar):
        return float(np.cov(y, x_scalar, bias=True)[0, 1] / np.var(x_scalar))

    b_par_true = slope(dr_par_true, dvx)
    b_par_model = slope(dr_par_mean, dvx)
    var_dvperp = float(np.mean(np.sum(dv_perp * dv_perp, axis=-1)))
    b_perp_true = float(np.mean(np.sum(dv_perp * dr_perp_true_vec, axis=-1))) / var_dvperp
    b_perp_model = float(np.mean(np.sum(dv_perp * dr_perp_mean_vec, axis=-1))) / var_dvperp
    var_sv = float(np.mean(np.sum(sv * sv, axis=-1)))
    b_com_true = float(np.mean(np.sum(sv * sr_true, axis=-1))) / var_sv
    b_com_model = float(np.mean(np.sum(sv * sr_model, axis=-1))) / var_sv

    gains = {
        "gamma_par": b_par_true - b_par_model,
        "gamma_perp": b_perp_true - b_perp_model,
        "gamma_com": b_com_true - b_com_model,
        "b_par_true": b_par_true, "b_par_model": b_par_model,
        "b_perp_true": b_perp_true, "b_perp_model": b_perp_model,
        "b_com_true": b_com_true, "b_com_model": b_com_model,
        "run_dir": str(run_dir), "n_transitions": int(N),
        "best_val_loss": ckpt.get("best_val_loss"),
    }
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in gains.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-gains", required=True, help="Path to write fd_gains JSON.")
    ap.add_argument("--max-samples", type=int, default=60000)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--Tr", type=float, default=1.0)
    ap.add_argument("--Tz", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    gains = calibrate(args.run_dir, max_samples=args.max_samples, n_draws=args.n_draws,
                      Tr=args.Tr, Tz=args.Tz, batch=args.batch, sample_seed=args.sample_seed)

    print(f"[calibrate] {args.run_dir}  (best_val={gains['best_val_loss']}, "
          f"N={gains['n_transitions']})")
    for ch in ("par", "perp", "com"):
        print(f"  {ch:>4}: b_true={gains['b_'+ch+'_true']:+.5f}  "
              f"b_model={gains['b_'+ch+'_model']:+.5f}  "
              f"gamma_add={gains['gamma_'+ch]:+.6f}")
    Path(args.out_gains).write_text(json.dumps(gains, indent=2))
    print(f"[calibrate] wrote gains -> {args.out_gains}")


if __name__ == "__main__":
    main()
