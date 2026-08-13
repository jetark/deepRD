"""
Train the paper's ``deep4`` E3-equivariant dimer CVAE from a single seed.

Loads the frozen recipe ``configs/deep4.yaml`` (E3DimerCVAE, irreps
32x0e + 16x1o + 8x2e, dqpipimririm conditioning, 4 decoder layers, 200 training
trajectories, 50 epochs) and varies only the seed, writing a run directory that
``calibrate_gains.py`` / ``generate_rollout.py`` (and the standard
``benchmarkReducedDimerE3Gen.py``) can load: ``config.yaml`` + ``checkpoint.pt``
+ ``e3_normalizer.json`` (+ ``final.pt`` / ``history.json``).

Reproducibility
---------------
``set_seed(seed)`` seeds Python/NumPy/torch. The training-trajectory subset is
drawn with ``np.random.choice`` from the (seeded) global NumPy RNG and the
train/val split is index-deterministic, so a given seed always selects the same
data. Weight init, batch shuffling and the reparameterisation noise are driven
by the seeded torch RNG. The result is statistically reproducible from the seed;
for bit-exact reproduction, load the saved ``checkpoint.pt`` directly (GPU kernel
reductions are not made bitwise-deterministic here).

The published paper model is seed 302 (``--seed 302``, the default).

Usage
-----
    python -m deepRD.noiseSampler.e3cvae.reproduce.train --seed 302
    python -m deepRD.noiseSampler.e3cvae.reproduce.train --seed 302 \
        --out deepRD/noiseSampler/training/results_e3/repro_s302 --device cuda
"""
import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepRD.noiseSampler.cvae.config import (
    build_model_from_config_e3, load_config, save_config)
from deepRD.noiseSampler.cvae.datasets import extract_vars, load_datasets
from deepRD.noiseSampler.e3cvae.normalization import E3VectorNormalizer
from deepRD.noiseSampler.e3cvae.tools import (
    DimerE3Dataset, collate_dimer_e3_graphs, construct_dqpipimririm_tensors)
from deepRD.noiseSampler.e3cvae.training import (
    move_graph_batch_to_device, train_e3_cvae)
from deepRD.noiseSampler.e3cvae.train_dimer import (
    equivariance_diagnostics, set_seed, split_by_trajectory, write_json)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deep4.yaml"
RESULTS_ROOT = Path("deepRD/noiseSampler/training/results_e3")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=302,
                   help="Training seed (paper model = 302).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="Frozen deep4 recipe (default: packaged configs/deep4.yaml).")
    p.add_argument("--out", type=Path, default=None,
                   help="Run directory (default: results_e3/repro_s<seed>).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override config epochs (default: config value, 50).")
    p.add_argument("--n-trajectories", type=int, default=None,
                   help="Override number of training trajectories (default: 200).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    out = a.out or (RESULTS_ROOT / f"repro_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)

    set_seed(a.seed)
    cfg = load_config(a.config)
    cfg.experiment.seed = a.seed
    if a.epochs is not None:
        cfg.training.epochs = a.epochs
    if a.n_trajectories is not None:
        cfg.data.train_trajectories = a.n_trajectories
    cfg.paths.output_root = str(out)
    save_config(cfg, out / "config.yaml")

    epochs = cfg.training.epochs
    n_traj = cfg.data.train_trajectories
    print(f"[train] seed={a.seed} model={cfg.model.model_type} "
          f"irreps='{cfg.model.hidden_irreps}' n_dec_layers={cfg.model.n_dec_layers} "
          f"epochs={epochs} train_trajectories={n_traj}", flush=True)

    # Deterministic (seeded) data subset + trajectory split -- same as flagship.
    raw, _ = load_datasets(cfg.data.dataset_dir, n_datasets=n_traj,
                           n_total=cfg.data.total_trajectories, trajtype="bench")
    q, v, r = extract_vars(raw)
    lag2 = getattr(cfg.model, "lag2", False)
    structured = construct_dqpipimririm_tensors(q, v, r, lag2=lag2)
    tr_s, va_s = split_by_trajectory(structured, cfg.data.val_fraction)

    norm = E3VectorNormalizer.fit(tr_s)
    norm.save_json(out / cfg.paths.scaler_name)
    tr_s = norm.transform(tr_s)
    va_s = norm.transform(va_s)

    collate = partial(collate_dimer_e3_graphs, boxsize=cfg.system.boxsize)
    pin = a.device.startswith("cuda")
    tl = DataLoader(DimerE3Dataset(tr_s, flatten=True, lag2=lag2),
                    batch_size=cfg.training.batch_size, shuffle=True, num_workers=4,
                    collate_fn=collate, pin_memory=pin)
    vl = DataLoader(DimerE3Dataset(va_s, flatten=True, lag2=lag2),
                    batch_size=cfg.training.batch_size, shuffle=False, num_workers=4,
                    collate_fn=collate)

    model = build_model_from_config_e3(cfg).to(a.device)
    hist = train_e3_cvae(
        model, tl, val_loader=vl, epochs=epochs, lr=cfg.training.learning_rate,
        beta_max=cfg.training.beta_max, warmup_epochs=cfg.training.beta_warmup_epochs,
        grad_clip=cfg.training.grad_clip, weight_decay=cfg.training.weight_decay,
        save_path=out / "checkpoint.pt", early_stopping=cfg.training.early_stopping,
        patience=cfg.training.patience, min_delta=cfg.training.min_delta,
        validate_every=cfg.training.validate_every, device=a.device,
        free_bits=getattr(cfg.training, "free_bits", 0.0))

    ckpt = out / cfg.paths.checkpoint_name          # loader needs a weights-only checkpoint
    if not ckpt.exists():
        torch.save({"model_state": model.state_dict()}, ckpt)
    eq = equivariance_diagnostics(
        model, move_graph_batch_to_device(next(iter(vl)), a.device))
    write_json(out / "history.json", hist)
    torch.save({"model_state": model.state_dict(), "config": cfg, "history": hist,
                "normalizer": norm.state_dict(), "equivariance": eq}, out / "final.pt")
    print(f"[train] DONE -> {out}  best_val={hist.get('best_val_loss')} "
          f"best_epoch={hist.get('best_epoch')} equiv_err={eq}", flush=True)


if __name__ == "__main__":
    main()
