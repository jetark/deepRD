import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import deepRD
import deepRD.tools.analysisTools as analysisTools
import deepRD.tools.trajectoryTools as trajectoryTools
from deepRD.diffusionIntegrators import langevinNoiseSamplerDimerGlobal
from deepRD.noiseSampler.cvae.config import build_model_from_config_e3, load_config
from deepRD.noiseSampler.cvae.datasets import extract_vars, load_datasets
from deepRD.noiseSampler.e3cvae.losses import e3_cvae_axial_loss, e3_cvae_isotropic_loss
from deepRD.noiseSampler.e3cvae.normalization import E3VectorNormalizer
from deepRD.noiseSampler.e3cvae.tools import (
    DimerE3Dataset,
    append_z_to_decoder_features,
    build_dimer_graph_batch,
    collate_dimer_e3_graphs,
    construct_dqpipimririm_tensors,
)
from deepRD.noiseSampler.e3cvae.training import move_graph_batch_to_device
from deepRD.potentials import pairBistableBias


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_trajectory(structured: dict, val_fraction: float) -> tuple[dict, dict]:
    n_traj = structured["q1"].shape[0]
    split = int((1.0 - val_fraction) * n_traj)
    if split <= 0 or split >= n_traj:
        raise ValueError(f"Bad validation split for n_traj={n_traj}, val_fraction={val_fraction}")
    return (
        {key: value[:split] for key, value in structured.items()},
        {key: value[split:] for key, value in structured.items()},
    )


def load_trained_model(run_dir: Path, device: torch.device):
    config = load_config(run_dir / "config.yaml")
    normalizer = E3VectorNormalizer.load_json(run_dir / config.paths.scaler_name)
    model = build_model_from_config_e3(config).to(device)
    checkpoint = torch.load(run_dir / config.paths.checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, normalizer, model, checkpoint


def load_validation_structured(config, normalizer: E3VectorNormalizer):
    set_seed(config.experiment.seed)
    raw_dataset, _ = load_datasets(
        config.data.dataset_dir,
        n_datasets=config.data.train_trajectories,
        n_total=config.data.total_trajectories,
        trajtype="bench",
    )
    q, v, r = extract_vars(raw_dataset)
    structured = construct_dqpipimririm_tensors(q, v, r)
    _, val_structured = split_by_trajectory(structured, config.data.val_fraction)
    return normalizer.transform(val_structured)


def subsample_structured(structured: dict, max_samples: int | None, seed: int) -> dict:
    if max_samples is None or max_samples <= 0:
        return structured

    n_traj, t_eff = structured["q1"].shape[:2]
    total = n_traj * t_eff
    if max_samples >= total:
        return structured

    generator = torch.Generator().manual_seed(seed)
    flat_idx = torch.randperm(total, generator=generator)[:max_samples]
    out = {}
    for key, value in structured.items():
        flat = value.reshape(total, value.shape[-1])
        out[key] = flat[flat_idx]
    return out


def _stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _axis_project(vectors: np.ndarray, units: np.ndarray) -> np.ndarray:
    return np.sum(vectors * units, axis=-1)


@torch.no_grad()
def one_step_diagnostics(model, loader, normalizer: E3VectorNormalizer, device: torch.device):
    aux_scale = normalizer.auxiliary_scale
    all_target = []
    all_recon = []
    all_gen = []
    all_bond_unit = []
    losses = []
    nlls = []
    kls = []
    log_sigma = []
    z_mu = []
    z_logvar = []

    for batch in loader:
        batch = move_graph_batch_to_device(batch, device)

        outputs = model(batch)
        loss, nll, kl = e3_cvae_axial_loss(outputs, batch, beta=1.0)
        losses.append(float(loss.item()))
        nlls.append(float(nll.item()))
        kls.append(float(kl.item()))

        enc_mu, enc_logvar = model.encoder(
            h=batch["h_enc"],
            edge_index=batch["edge_index"],
            edge_vec=batch["edge_vec"],
            edge_radial=batch["edge_radial"],
            batch_index=batch["batch_index"],
        )
        h_dec = append_z_to_decoder_features(
            batch["h_dec_base"],
            enc_mu.repeat_interleave(2, dim=0),
        )
        recon_mu, recon_log_sigma = model.decoder(
            h=h_dec,
            edge_index=batch["edge_index"],
            edge_vec=batch["edge_vec"],
            edge_radial=batch["edge_radial"],
        )
        gen, _, gen_log_sigma = model.sample_torch(batch)

        # Node tensors are [B*2, 3]. Reshape to [B, 2, 3] for dimer diagnostics.
        B = batch["num_graphs"]
        target = (batch["r_next"].reshape(B, 2, 3) * aux_scale).cpu().numpy()
        recon = (recon_mu.reshape(B, 2, 3) * aux_scale).cpu().numpy()
        generated = (gen.reshape(B, 2, 3) * aux_scale).cpu().numpy()
        bond_unit = batch["bond_unit_node"].reshape(B, 2, 3)[:, 0, :].cpu().numpy()

        all_target.append(target)
        all_recon.append(recon)
        all_gen.append(generated)
        all_bond_unit.append(bond_unit)
        log_sigma.append(recon_log_sigma.cpu().numpy())
        log_sigma.append(gen_log_sigma.cpu().numpy())
        z_mu.append(enc_mu.cpu().numpy())
        z_logvar.append(enc_logvar.cpu().numpy())

    target = np.concatenate(all_target, axis=0)
    recon = np.concatenate(all_recon, axis=0)
    generated = np.concatenate(all_gen, axis=0)
    bond_unit = np.concatenate(all_bond_unit, axis=0)

    target_flat = target.reshape(target.shape[0], 6)
    recon_flat = recon.reshape(recon.shape[0], 6)
    gen_flat = generated.reshape(generated.shape[0], 6)

    recon_err = recon_flat - target_flat
    gen_err = gen_flat - target_flat
    target_rms = np.sqrt(np.mean(target_flat * target_flat))

    target_rel = target[:, 1, :] - target[:, 0, :]
    recon_rel = recon[:, 1, :] - recon[:, 0, :]
    gen_rel = generated[:, 1, :] - generated[:, 0, :]
    target_com = 0.5 * (target[:, 0, :] + target[:, 1, :])
    recon_com = 0.5 * (recon[:, 0, :] + recon[:, 1, :])
    gen_com = 0.5 * (generated[:, 0, :] + generated[:, 1, :])

    metrics = {
        "num_graphs": int(target.shape[0]),
        "num_target_dimensions_per_graph": 6,
        "checkpoint_elbo": {
            "loss_mean": float(np.mean(losses)),
            "nll_mean": float(np.mean(nlls)),
            "kl_mean": float(np.mean(kls)),
        },
        "posterior_mean_reconstruction": {
            "rmse_6d": float(np.sqrt(np.mean(recon_err * recon_err))),
            "mae_6d": float(np.mean(np.abs(recon_err))),
            "relative_rmse_vs_target_rms": float(np.sqrt(np.mean(recon_err * recon_err)) / target_rms),
            "target_rms": float(target_rms),
            "prediction_rms": float(np.sqrt(np.mean(recon_flat * recon_flat))),
            "error_rms_by_component": np.sqrt(np.mean(recon_err * recon_err, axis=0)).tolist(),
        },
        "prior_generation_one_sample": {
            "rmse_6d_vs_realized_target": float(np.sqrt(np.mean(gen_err * gen_err))),
            "mae_6d_vs_realized_target": float(np.mean(np.abs(gen_err))),
            "relative_rmse_vs_target_rms": float(np.sqrt(np.mean(gen_err * gen_err)) / target_rms),
            "sample_rms": float(np.sqrt(np.mean(gen_flat * gen_flat))),
        },
        "marginals": {
            "target_components": _stats(target_flat),
            "recon_components": _stats(recon_flat),
            "generated_components": _stats(gen_flat),
            "target_node_norm": _stats(np.linalg.norm(target.reshape(-1, 3), axis=-1)),
            "recon_node_norm": _stats(np.linalg.norm(recon.reshape(-1, 3), axis=-1)),
            "generated_node_norm": _stats(np.linalg.norm(generated.reshape(-1, 3), axis=-1)),
            "target_rel_parallel": _stats(_axis_project(target_rel, bond_unit)),
            "recon_rel_parallel": _stats(_axis_project(recon_rel, bond_unit)),
            "generated_rel_parallel": _stats(_axis_project(gen_rel, bond_unit)),
            "target_rel_norm": _stats(np.linalg.norm(target_rel, axis=-1)),
            "recon_rel_norm": _stats(np.linalg.norm(recon_rel, axis=-1)),
            "generated_rel_norm": _stats(np.linalg.norm(gen_rel, axis=-1)),
            "target_com_norm": _stats(np.linalg.norm(target_com, axis=-1)),
            "recon_com_norm": _stats(np.linalg.norm(recon_com, axis=-1)),
            "generated_com_norm": _stats(np.linalg.norm(gen_com, axis=-1)),
        },
        "latent": {
            "z_mu": _stats(np.concatenate(z_mu, axis=0)),
            "z_logvar": _stats(np.concatenate(z_logvar, axis=0)),
        },
        "sigma_normalized": {
            "log_sigma": _stats(np.concatenate(log_sigma, axis=0)),
            "sigma": _stats(np.exp(np.concatenate(log_sigma, axis=0))),
        },
    }
    return metrics


def minimal_image_np(rel, boxsize):
    return rel - boxsize * np.round(rel / boxsize)


def bond_length(X, boxsize):
    rel = minimal_image_np(X[:, 1, :] - X[:, 0, :], boxsize)
    return np.linalg.norm(rel, axis=-1)


def axis_rel_vel(X, V, boxsize):
    rel = minimal_image_np(X[:, 1, :] - X[:, 0, :], boxsize)
    unit = rel / np.linalg.norm(rel, axis=-1, keepdims=True).clip(1e-12, None)
    return np.sum((V[:, 1, :] - V[:, 0, :]) * unit, axis=-1)


def acf_1d(series, max_lag):
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    var = np.mean(series * series)
    out = np.zeros(max_lag + 1, dtype=np.float64)
    if var < 1e-15:
        return out
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.mean(series[:-lag] * series[lag:]) / var
    return out


def extract_equilibrium_frames(dataset_dir: str, n_frames: int, n_total: int, boxsize: float):
    frames = []
    file_base = str(Path(dataset_dir) / "simMoriZwanzig_")
    for fnum in range(n_total):
        if len(frames) >= n_frames:
            break
        try:
            traj = trajectoryTools.loadTrajectory(file_base, fnum)
        except Exception:
            continue
        T = traj.shape[0] // 2
        n = T // 2
        if n < 1:
            continue
        q1 = traj[2 * n, 1:4]
        q2 = traj[2 * n + 1, 1:4]
        dx = np.linalg.norm(minimal_image_np(q2 - q1, boxsize))
        if 0.8 < dx < 2.0:
            frames.append(
                {
                    "q1": q1.copy(),
                    "q2": q2.copy(),
                    "v1": traj[2 * n, 4:7].copy(),
                    "v2": traj[2 * n + 1, 4:7].copy(),
                    "v1p": traj[2 * (n - 1), 4:7].copy(),
                    "v2p": traj[2 * (n - 1) + 1, 4:7].copy(),
                    "r1": traj[2 * n, 8:11].copy(),
                    "r2": traj[2 * n + 1, 8:11].copy(),
                    "r1p": traj[2 * (n - 1), 8:11].copy(),
                    "r2p": traj[2 * (n - 1) + 1, 8:11].copy(),
                }
            )
    if not frames:
        raise RuntimeError("No equilibrium frames found for rollout initialization.")
    return frames


def load_benchmark_reference(dataset_dir: str, n_ref: int, n_total: int):
    file_base = str(Path(dataset_dir) / "simMoriZwanzig_")
    bench_X = []
    bench_V = []
    for fnum in range(n_total):
        if len(bench_X) >= n_ref:
            break
        try:
            traj = trajectoryTools.loadTrajectory(file_base, fnum)
        except Exception:
            continue
        bench_X.append(np.stack([traj[0::2, 1:4], traj[1::2, 1:4]], axis=1))
        bench_V.append(np.stack([traj[0::2, 4:7], traj[1::2, 4:7]], axis=1))
    if not bench_X:
        raise RuntimeError("No benchmark trajectories found for rollout reference.")
    return bench_X, bench_V


class Lag2IntegratorGlobal(langevinNoiseSamplerDimerGlobal):
    """
    Extends the global dimer integrator to track two steps of r and v history
    in aux4 (r_nm2) and aux5 (v_nm2) for lag-2 E3 conditioning.

    Conditioning key: "E3_lag2" — returns 42D vector:
        q1 q2 v1_n v2_n v1_nm1 v2_nm1 v1_nm2 v2_nm2
        r1_n r2_n r1_nm1 r2_nm1 r1_nm2 r2_nm2
    """

    def integrateBOBGlobal(self, particleList, dt):
        # Full loop override so aux4/aux5 shift happens AFTER getConditionedVars
        # reads them (as v_nm2 / r_nm2) but BEFORE aux3/aux2 are updated.
        for i in range(int(len(particleList) // 2)):
            p1 = particleList[2 * i]
            p2 = particleList[2 * i + 1]
            exp1 = np.exp(-dt * self.Gamma / p1.mass)
            exp2 = np.exp(-dt * self.Gamma / p2.mass)
            ff1 = p1.nextVelocity * exp1 + (1 + exp1) * self.forceField[2 * i] * dt / (2 * p1.mass)
            ff2 = p2.nextVelocity * exp2 + (1 + exp2) * self.forceField[2 * i + 1] * dt / (2 * p2.mass)
            # getConditionedVars reads aux3 (v_nm1), aux5 (v_nm2), aux2 (r_nm1), aux4 (r_nm2)
            conditioned_vars = self.getConditionedVars(p1, p2, 0)
            noise = self.noiseSampler.sample(conditioned_vars)
            n1, n2 = noise[0:3], noise[3:6]
            self.rel1 = self.relDistance
            self.axv1 = self.axisRelVelocity
            # Shift lag-2 history AFTER reading, BEFORE updating lag-1
            p1.aux5 = 1.0 * p1.aux3
            p2.aux5 = 1.0 * p2.aux3
            p1.aux4 = 1.0 * p1.aux2
            p2.aux4 = 1.0 * p2.aux2
            # Update lag-1 history
            p1.aux2 = 1.0 * p1.aux1
            p2.aux2 = 1.0 * p2.aux1
            p1.aux1 = n1
            p2.aux1 = n2
            p1.aux3 = 1.0 * p1.nextVelocity
            p2.aux3 = 1.0 * p2.nextVelocity
            p1.nextVelocity = ff1 + n1
            p2.nextVelocity = ff2 + n2

    def getConditionedVars(self, particle1, particle2, index):
        if self.conditionedOn == "E3_lag2":
            return np.concatenate((
                particle1.nextPosition, particle2.nextPosition,
                particle1.nextVelocity, particle2.nextVelocity,
                particle1.aux3, particle2.aux3,    # v_nm1
                particle1.aux5, particle2.aux5,    # v_nm2
                particle1.aux1, particle2.aux1,    # r_n
                particle1.aux2, particle2.aux2,    # r_nm1
                particle1.aux4, particle2.aux4,    # r_nm2
            ))
        return super().getConditionedVars(particle1, particle2, index)


class E3DimerRolloutSampler:
    """
    Adapter from the 30D E3_base integrator conditioning vector to E3DimerCVAE.

    Conditioning layout:
        q1 q2 v1_n v2_n v1_nm1 v2_nm1 r1_n r2_n r1_nm1 r2_nm1
    """

    def __init__(self, model, normalizer: E3VectorNormalizer, boxsize: float, device: torch.device, Tr: float = 1.0, Tz: float = 1.0):
        self.model = model
        self.normalizer = normalizer
        self.boxsize = boxsize
        self.device = device
        self.Tr = Tr
        self.Tz = Tz

    @torch.no_grad()
    def sample(self, conditioned_vars):
        c = np.asarray(conditioned_vars, dtype=np.float32).reshape(30)
        q1, q2 = c[0:3], c[3:6]
        v1, v2 = c[6:9], c[9:12]
        v1p, v2p = c[12:15], c[15:18]
        r1, r2 = c[18:21], c[21:24]
        r1p, r2p = c[24:27], c[27:30]

        structured = {
            "q1": torch.tensor(q1, dtype=torch.float32)[None, :],
            "q2": torch.tensor(q2, dtype=torch.float32)[None, :],
            "v1_n": torch.tensor(v1, dtype=torch.float32)[None, :],
            "v2_n": torch.tensor(v2, dtype=torch.float32)[None, :],
            "v1_nm1": torch.tensor(v1p, dtype=torch.float32)[None, :],
            "v2_nm1": torch.tensor(v2p, dtype=torch.float32)[None, :],
            "r1_n": torch.tensor(r1, dtype=torch.float32)[None, :],
            "r2_n": torch.tensor(r2, dtype=torch.float32)[None, :],
            "r1_nm1": torch.tensor(r1p, dtype=torch.float32)[None, :],
            "r2_nm1": torch.tensor(r2p, dtype=torch.float32)[None, :],
        }
        structured = self.normalizer.transform(structured)
        batch = build_dimer_graph_batch(
            q1=structured["q1"],
            q2=structured["q2"],
            v1=structured["v1_n"],
            v2=structured["v2_n"],
            r1=structured["r1_n"],
            r2=structured["r2_n"],
            v1_prev=structured["v1_nm1"],
            v2_prev=structured["v2_nm1"],
            r1_prev=structured["r1_nm1"],
            r2_prev=structured["r2_nm1"],
            boxsize=self.boxsize,
        )
        batch = move_graph_batch_to_device(batch, self.device)
        r_norm, _, _ = self.model.sample_torch(batch, Tr=self.Tr, Tz=self.Tz)
        r_phys = self.normalizer.inverse_transform_aux(r_norm).reshape(2, 3)
        return r_phys.cpu().numpy().reshape(6)


class E3Lag2RolloutSampler(E3DimerRolloutSampler):
    """
    Adapter from the 42D E3_lag2 integrator conditioning vector to E3DimerCVAE
    with lag-2 conditioning (v_nm2, r_nm2 included).

    Conditioning layout (42D):
        q1 q2 v1_n v2_n v1_nm1 v2_nm1 v1_nm2 v2_nm2
        r1_n r2_n r1_nm1 r2_nm1 r1_nm2 r2_nm2
    """

    @torch.no_grad()
    def sample(self, conditioned_vars):
        c = np.asarray(conditioned_vars, dtype=np.float32).reshape(42)
        q1, q2 = c[0:3], c[3:6]
        v1, v2 = c[6:9], c[9:12]
        v1p, v2p = c[12:15], c[15:18]
        v1pp, v2pp = c[18:21], c[21:24]
        r1, r2 = c[24:27], c[27:30]
        r1p, r2p = c[30:33], c[33:36]
        r1pp, r2pp = c[36:39], c[39:42]

        structured = {
            "q1": torch.tensor(q1, dtype=torch.float32)[None, :],
            "q2": torch.tensor(q2, dtype=torch.float32)[None, :],
            "v1_n": torch.tensor(v1, dtype=torch.float32)[None, :],
            "v2_n": torch.tensor(v2, dtype=torch.float32)[None, :],
            "v1_nm1": torch.tensor(v1p, dtype=torch.float32)[None, :],
            "v2_nm1": torch.tensor(v2p, dtype=torch.float32)[None, :],
            "v1_nm2": torch.tensor(v1pp, dtype=torch.float32)[None, :],
            "v2_nm2": torch.tensor(v2pp, dtype=torch.float32)[None, :],
            "r1_n": torch.tensor(r1, dtype=torch.float32)[None, :],
            "r2_n": torch.tensor(r2, dtype=torch.float32)[None, :],
            "r1_nm1": torch.tensor(r1p, dtype=torch.float32)[None, :],
            "r2_nm1": torch.tensor(r2p, dtype=torch.float32)[None, :],
            "r1_nm2": torch.tensor(r1pp, dtype=torch.float32)[None, :],
            "r2_nm2": torch.tensor(r2pp, dtype=torch.float32)[None, :],
        }
        structured = self.normalizer.transform(structured)
        batch = build_dimer_graph_batch(
            q1=structured["q1"],
            q2=structured["q2"],
            v1=structured["v1_n"],
            v2=structured["v2_n"],
            r1=structured["r1_n"],
            r2=structured["r2_n"],
            v1_prev=structured["v1_nm1"],
            v2_prev=structured["v2_nm1"],
            r1_prev=structured["r1_nm1"],
            r2_prev=structured["r2_nm1"],
            boxsize=self.boxsize,
            v1_prev2=structured["v1_nm2"],
            v2_prev2=structured["v2_nm2"],
            r1_prev2=structured["r1_nm2"],
            r2_prev2=structured["r2_nm2"],
        )
        batch = move_graph_batch_to_device(batch, self.device)
        r_norm, _, _ = self.model.sample_torch(batch, Tr=self.Tr, Tz=self.Tz)
        r_phys = self.normalizer.inverse_transform_aux(r_norm).reshape(2, 3)
        return r_phys.cpu().numpy().reshape(6)


def run_one_rollout(sampler, frame, params, n_steps: int, seed: int, lag2: bool = False):
    set_seed(seed)
    p1 = deepRD.particle(frame["q1"].copy(), velocity=frame["v1"].copy(), mass=params["mass"])
    p2 = deepRD.particle(frame["q2"].copy(), velocity=frame["v2"].copy(), mass=params["mass"])
    plist = deepRD.particleList([p1, p2])

    if lag2:
        integrator_cls = Lag2IntegratorGlobal
        conditioned_on = "E3_lag2"
    else:
        integrator_cls = langevinNoiseSamplerDimerGlobal
        conditioned_on = "E3_base"

    integrator = integrator_cls(
        params["dt"],
        1,
        n_steps * params["dt"],
        params["Gamma"],
        sampler,
        params["KbT"],
        params["boxsize"],
        params["boundaryType"],
        0,
        conditioned_on,
    )
    integrator.setPairPotential(pairBistableBias(0.5, 0.5, 2))
    integrator.prepareSimulation(plist)

    p1.aux1 = frame["r1"].copy()
    p2.aux1 = frame["r2"].copy()
    p1.aux2 = frame["r1p"].copy()
    p2.aux2 = frame["r2p"].copy()
    p1.aux3 = frame["v1p"].copy()
    p2.aux3 = frame["v2p"].copy()
    if lag2:
        # Bootstrap lag-2 history from lag-1 (best available at init)
        p1.aux4 = frame["r1p"].copy()
        p2.aux4 = frame["r2p"].copy()
        p1.aux5 = frame["v1p"].copy()
        p2.aux5 = frame["v2p"].copy()
    integrator.calculateForceField(plist)

    X = [plist.positions.copy()]
    V = [plist.velocities.copy()]
    for _ in range(n_steps):
        integrator.integrateOne(plist)
        X.append(plist.positions.copy())
        V.append(plist.velocities.copy())
    return np.asarray(X), np.asarray(V)


def rollout_diagnostics(model, normalizer, config, device, n_sims: int, n_steps: int, max_lag: int):
    params = analysisTools.readParameters(str(Path(config.data.dataset_dir) / "parameters"))
    lag2 = getattr(model, "lag2", False)
    sampler_cls = E3Lag2RolloutSampler if lag2 else E3DimerRolloutSampler
    sampler = sampler_cls(model, normalizer, config.system.boxsize, device)
    frames = extract_equilibrium_frames(
        config.data.dataset_dir,
        n_frames=n_sims,
        n_total=config.data.total_trajectories,
        boxsize=config.system.boxsize,
    )
    bench_X, bench_V = load_benchmark_reference(
        config.data.dataset_dir,
        n_ref=max(n_sims, 4),
        n_total=config.data.total_trajectories,
    )

    X_list = []
    V_list = []
    failures = []
    for idx in range(n_sims):
        try:
            X, V = run_one_rollout(sampler, frames[idx % len(frames)], params, n_steps, seed=1000 + idx, lag2=lag2)
            X_list.append(X)
            V_list.append(V)
        except Exception as exc:
            failures.append(f"sim {idx}: {exc}")

    if not X_list:
        return {"ok": False, "failures": failures}

    boxsize = config.system.boxsize
    dx_model = np.concatenate([bond_length(X, boxsize) for X in X_list])
    dvx_model = np.concatenate([axis_rel_vel(X, V, boxsize) for X, V in zip(X_list, V_list)])
    dx_bench = np.concatenate([bond_length(X, boxsize) for X in bench_X])
    dvx_bench = np.concatenate([axis_rel_vel(X, V, boxsize) for X, V in zip(bench_X, bench_V)])
    acf_dx_model = np.mean([acf_1d(bond_length(X, boxsize), max_lag) for X in X_list], axis=0)
    acf_dvx_model = np.mean(
        [acf_1d(axis_rel_vel(X, V, boxsize), max_lag) for X, V in zip(X_list, V_list)],
        axis=0,
    )
    acf_dx_bench = np.mean([acf_1d(bond_length(X, boxsize), max_lag) for X in bench_X], axis=0)
    acf_dvx_bench = np.mean(
        [acf_1d(axis_rel_vel(X, V, boxsize), max_lag) for X, V in zip(bench_X, bench_V)],
        axis=0,
    )

    return {
        "ok": True,
        "failures": failures,
        "n_sims": int(len(X_list)),
        "n_steps": int(n_steps),
        "dx_model": _stats(dx_model),
        "dx_benchmark": _stats(dx_bench),
        "dvx_model": _stats(dvx_model),
        "dvx_benchmark": _stats(dvx_bench),
        "acf_dx_lag1": float(acf_dx_model[1]),
        "acf_dx_benchmark_lag1": float(acf_dx_bench[1]),
        "acf_dvx_lag1": float(acf_dvx_model[1]),
        "acf_dvx_benchmark_lag1": float(acf_dvx_bench[1]),
        "acf_dx_rmse_first_40": float(np.sqrt(np.mean((acf_dx_model[:40] - acf_dx_bench[:40]) ** 2))),
        "acf_dvx_rmse_first_40": float(np.sqrt(np.mean((acf_dvx_model[:40] - acf_dvx_bench[:40]) ** 2))),
    }


def write_report(path: Path, one_step: dict, rollout: dict | None, checkpoint: dict) -> None:
    lines = []
    lines.append("E3DimerCVAE diagnostics")
    lines.append("")
    lines.append(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    lines.append(f"checkpoint_best_val_loss: {checkpoint.get('best_val_loss')}")
    lines.append("")
    lines.append("One-step")
    lines.append(f"num_graphs: {one_step['num_graphs']}")
    lines.append(f"target_dimensions_per_graph: {one_step['num_target_dimensions_per_graph']}")
    rec = one_step["posterior_mean_reconstruction"]
    gen = one_step["prior_generation_one_sample"]
    lines.append(f"posterior_mean_rmse_6d: {rec['rmse_6d']:.6g}")
    lines.append(f"posterior_mean_relative_rmse: {rec['relative_rmse_vs_target_rms']:.6g}")
    lines.append(f"prior_sample_rmse_6d_vs_realized_target: {gen['rmse_6d_vs_realized_target']:.6g}")
    lines.append(f"target_component_std: {one_step['marginals']['target_components']['std']:.6g}")
    lines.append(f"recon_component_std: {one_step['marginals']['recon_components']['std']:.6g}")
    lines.append(f"generated_component_std: {one_step['marginals']['generated_components']['std']:.6g}")
    lines.append(f"target_rel_parallel_std: {one_step['marginals']['target_rel_parallel']['std']:.6g}")
    lines.append(f"recon_rel_parallel_std: {one_step['marginals']['recon_rel_parallel']['std']:.6g}")
    lines.append(f"generated_rel_parallel_std: {one_step['marginals']['generated_rel_parallel']['std']:.6g}")
    lines.append(f"elbo_loss_mean: {one_step['checkpoint_elbo']['loss_mean']:.6g}")
    lines.append(f"elbo_nll_mean: {one_step['checkpoint_elbo']['nll_mean']:.6g}")
    lines.append(f"elbo_kl_mean: {one_step['checkpoint_elbo']['kl_mean']:.6g}")
    if rollout is not None:
        lines.append("")
        lines.append("Short rollouts")
        lines.append(f"ok: {rollout.get('ok')}")
        if rollout.get("ok"):
            lines.append(f"n_sims: {rollout['n_sims']}")
            lines.append(f"n_steps: {rollout['n_steps']}")
            lines.append(f"dx_model_mean/std: {rollout['dx_model']['mean']:.6g} / {rollout['dx_model']['std']:.6g}")
            lines.append(
                f"dx_benchmark_mean/std: {rollout['dx_benchmark']['mean']:.6g} / "
                f"{rollout['dx_benchmark']['std']:.6g}"
            )
            lines.append(f"dvx_model_mean/std: {rollout['dvx_model']['mean']:.6g} / {rollout['dvx_model']['std']:.6g}")
            lines.append(
                f"dvx_benchmark_mean/std: {rollout['dvx_benchmark']['mean']:.6g} / "
                f"{rollout['dvx_benchmark']['std']:.6g}"
            )
            lines.append(f"acf_dx_rmse_first_40: {rollout['acf_dx_rmse_first_40']:.6g}")
            lines.append(f"acf_dvx_rmse_first_40: {rollout['acf_dvx_rmse_first_40']:.6g}")
        if rollout.get("failures"):
            lines.append(f"failures: {rollout['failures']}")
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run E3DimerCVAE one-step and short-rollout diagnostics.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all validation samples.")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--rollout-sims", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--rollout-max-lag", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "diagnostics_e3"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config, normalizer, model, checkpoint = load_trained_model(run_dir, device)
    val_structured = load_validation_structured(config, normalizer)
    val_structured = subsample_structured(val_structured, args.max_samples, config.experiment.seed + 99)
    val_ds = DimerE3Dataset(val_structured, flatten=(val_structured["q1"].ndim == 3))
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_dimer_e3_graphs, boxsize=config.system.boxsize),
    )

    one_step = one_step_diagnostics(model, loader, normalizer, device)
    rollout = None
    if args.rollout:
        rollout = rollout_diagnostics(
            model,
            normalizer,
            config,
            device,
            n_sims=args.rollout_sims,
            n_steps=args.rollout_steps,
            max_lag=args.rollout_max_lag,
        )

    result = {
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "beta": checkpoint.get("beta"),
        },
        "one_step": one_step,
        "rollout": rollout,
    }
    with (output_dir / "diagnostics.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    write_report(output_dir / "summary.txt", one_step, rollout, checkpoint)

    print((output_dir / "summary.txt").read_text())
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
