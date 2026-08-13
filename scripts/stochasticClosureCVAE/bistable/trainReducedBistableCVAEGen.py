"""
Train a standard-prior CVAE (CVAE_SP) noise sampler for the 1D bistable system,
using the noiseSampler/training config system — the same convention as the dimer
CVAE scripts, but under a dedicated bistable/ folder so bistable and dimer models
never collide (results/bistable/<cond>/<NNN>_<name>/ with config.yaml + checkpoint.pt +
scalers.pkl, loadable via deepRD.noiseSampler.cvae.checkpoints.load_run).

Pipeline (start to finish):
    load benchmark trajectories  ->  build_conditioning_and_scalers
    ->  normalize + RCDataset split  ->  train CVAE_SP  ->  save run dir

Run-dir layout (mirrors the dimer CVAE runs, under a dedicated bistable/ folder):
    deepRD/noiseSampler/training/results/bistable/<cond>/<NNN>_<name>/
        config.yaml        (CVAEConfig; consumed by load_run / the rollout scripts)
        checkpoint.pt       ({"model_state": ...})
        scalers.pkl         ({"scaler_r": ..., "scaler_c": ...})
        history.json        (training/validation loss curves)

Usage:
    python trainReducedBistableCVAEGen.py --cond piri
    python trainReducedBistableCVAEGen.py --cond all --n-trajectories 400 --epochs 60
    python trainReducedBistableCVAEGen.py --cond piri --name smoke --n-trajectories 20 --epochs 8
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepRD.noiseSampler.cvae.checkpoints import create_run_dir
from deepRD.noiseSampler.cvae.config import (
    CVAEConfig,
    DataSection,
    ExperimentSection,
    ModelSection,
    PathsSection,
    SystemSection,
    TrainingSection,
    build_model_from_config,
)
from deepRD.noiseSampler.cvae.datasets import (
    RCDataset,
    build_conditioning_and_scalers,
    extract_vars,
    get_nlags,
    load_datasets,
)
from deepRD.noiseSampler.cvae.models import CVAE
from deepRD.noiseSampler.cvae.training import train_model

# Standard-prior bistable conditionings (single distinguished particle, r in R^3).
BISTABLE_CONDS = ("piri", "pipimri", "piririm")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
# Bistable models live under their own top-level folder, separate from the dimer
# runs: results/bistable/<cond>/<NNN>_<name>/.
RESULTS_ROOT = _REPO_ROOT / "deepRD" / "noiseSampler" / "training" / "results" / "bistable"

DEFAULT_DATA_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/bistable/boxsize5/benchmark/"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(cond, args):
    idim, cdim = CVAE.assign_dims(system_type="bistable", cond_type=cond)
    return CVAEConfig(
        experiment=ExperimentSection(name=args.name, seed=args.seed, frame="global"),
        system=SystemSection(system_type="bistable", boxsize=args.boxsize, dt=args.dt, step=args.step),
        data=DataSection(
            conditioning=cond,
            n_lags=get_nlags(cond),
            train_trajectories=args.n_trajectories,
            val_fraction=args.val_fraction,
            dataset_dir=args.data_dir,
            total_trajectories=args.n_total,
            scaler_type="standard",
            weights="uniform",
        ),
        model=ModelSection(
            model_type=args.model_type,
            input_dim=idim,
            latent_dim=args.zdim,
            cond_dim=cdim,
            hidden_dims=list(args.hidden),
            # CVAE_SP = fixed N(0,I) prior; CVAE = learned/conditional prior p(z|c).
            standard_prior=(args.model_type == "CVAE_SP"),
        ),
        training=TrainingSection(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            beta_max=args.beta_max,
            beta_warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            early_stopping=True,
            patience=args.patience,
            min_delta=args.min_delta,
            validate_every=args.validate_every,
            num_workers=args.num_workers,
        ),
        paths=PathsSection(
            output_root=str(RESULTS_ROOT),
            checkpoint_name="checkpoint.pt",
            scaler_name="scalers.pkl",
        ),
    )


def train_one(cond, args, device):
    config = make_config(cond, args)
    run_dir = create_run_dir(config, output_root=str(RESULTS_ROOT))  # writes config.yaml
    print(f"[{cond}] run dir: {run_dir}")

    # ---- load benchmark data ----
    print(f"[{cond}] loading {args.n_trajectories} benchmark trajectories from {args.data_dir}")
    dataset, parameters = load_datasets(
        args.data_dir, n_datasets=args.n_trajectories, n_total=args.n_total, trajtype="bench"
    )
    q, v, r = extract_vars(dataset)
    print(f"[{cond}] data shapes q{tuple(q.shape)} v{tuple(v.shape)} r{tuple(r.shape)}")

    # ---- build conditioning + fit scalers ----
    r_next, c, scaler_r, scaler_c = build_conditioning_and_scalers(
        q, v, r,
        system_type="bistable",
        cond_type=cond,
        step=args.step,
        parameters=parameters,
    )
    r_next_norm = torch.tensor(scaler_r.transform(r_next), dtype=torch.float32)
    c_norm = torch.tensor(scaler_c.transform(c), dtype=torch.float32)

    # ---- random train/val split (single-step (r_next, c) pairs) ----
    N = r_next_norm.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_val = int(args.val_fraction * N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = RCDataset(r_next_norm[train_idx], c_norm[train_idx])
    val_ds = RCDataset(r_next_norm[val_idx], c_norm[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"[{cond}] train={len(train_ds)} val={len(val_ds)} cdim={config.model.cond_dim}")

    # ---- model (built from config, exactly as load_run/rollout will rebuild it) ----
    model = build_model_from_config(config).to(device)

    # ---- train (checkpoint.pt written on val improvement) ----
    ckpt_path = run_dir / config.paths.checkpoint_name
    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        beta_max=args.beta_max, warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip, save_path=ckpt_path,
        early_stopping=True, patience=args.patience, min_delta=args.min_delta,
        validate_every=args.validate_every, device=device,
    )

    # Guarantee the artifact exists even for very short runs that never improved.
    if not ckpt_path.exists():
        torch.save({"epoch": args.epochs, "model_state": model.state_dict()}, ckpt_path)

    with open(run_dir / config.paths.scaler_name, "wb") as f:
        pickle.dump({"scaler_r": scaler_r, "scaler_c": scaler_c}, f)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[{cond}] saved run dir -> {run_dir}")
    print(f"[{cond}] best_val_loss={history.get('best_val_loss')} best_epoch={history.get('best_epoch')}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond", default="all",
                   help="conditioning: one of piri/pipimri/piririm, or 'all'")
    p.add_argument("--name", default="base", help="run name (suffix of results/bistable/<cond>/<NNN>_<name>/)")
    p.add_argument("--model-type", default="CVAE", choices=["CVAE_SP", "CVAE"],
                   help="CVAE = conditional/learned prior p(z|c) (default recipe); "
                        "CVAE_SP = standard N(0,I) prior.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--n-total", type=int, default=2500, help="total benchmark files available")
    p.add_argument("--n-trajectories", type=int, default=200, help="trajectories sampled for training")
    p.add_argument("--step", type=int, default=1, help="time coarse-graining stride k")
    p.add_argument("--boxsize", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--zdim", type=int, default=3)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-max", type=float, default=1.1)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--validate-every", type=int, default=1)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  conds={args.cond}  n_trajectories={args.n_trajectories}  epochs={args.epochs}")

    conds = BISTABLE_CONDS if args.cond == "all" else (args.cond,)
    for cond in conds:
        assert cond in BISTABLE_CONDS, f"unknown cond {cond!r}, expected one of {BISTABLE_CONDS}"
        train_one(cond, args, device)


if __name__ == "__main__":
    main()
