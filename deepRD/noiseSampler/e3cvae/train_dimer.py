import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from e3nn import o3
from torch.utils.data import DataLoader

from deepRD.noiseSampler.cvae.config import (
    CVAEConfig,
    DataSection,
    ExperimentSection,
    ModelSection,
    PathsSection,
    SystemSection,
    TrainingSection,
    build_model_from_config_e3,
    save_config,
)
from deepRD.noiseSampler.cvae.datasets import extract_vars, load_datasets
from deepRD.noiseSampler.e3cvae.normalization import E3VectorNormalizer
from deepRD.noiseSampler.e3cvae.tools import (
    DimerE3Dataset,
    append_z_to_decoder_features,
    collate_dimer_e3_graphs,
    construct_dqpipimririm_tensors,
    rotate_batch_vectors,
)
from deepRD.noiseSampler.e3cvae.training import move_graph_batch_to_device, train_e3_cvae


def parse_args():
    parser = argparse.ArgumentParser(description="Train E3DimerCVAE on dimer dqpipimririm data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="deepRD/noiseSampler/training/results/e3_dimer")
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--n-total", type=int, default=2500)
    parser.add_argument("--boxsize", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--zdim", type=int, default=3)
    parser.add_argument("--hidden-irreps", default="32x0e + 16x1o + 8x2e")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args):
    return CVAEConfig(
        experiment=ExperimentSection(name="e3_dimer_dqpipimririm", seed=args.seed, frame="global"),
        system=SystemSection(system_type="dimer", boxsize=args.boxsize, dt=args.dt, step=1),
        data=DataSection(
            conditioning="dqpipimririm",
            train_trajectories=args.n_trajectories,
            val_fraction=args.val_fraction,
            dataset_dir=args.data_dir,
            total_trajectories=args.n_total,
            scaler_type="e3_vector_rms",
        ),
        model=ModelSection(
            model_type="E3DimerCVAE",
            input_dim=6,
            latent_dim=args.zdim,
            hidden_dims=[],
            hidden_irreps=args.hidden_irreps,
            standard_prior=True,
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
            output_root=args.output_dir,
            checkpoint_name="checkpoint.pt",
            scaler_name="e3_normalizer.json",
        ),
    )


def split_by_trajectory(structured, val_fraction):
    n_traj = structured["q1"].shape[0]
    split = int((1.0 - val_fraction) * n_traj)
    if split <= 0 or split >= n_traj:
        raise ValueError(
            f"Bad val split for n_traj={n_traj}, val_fraction={val_fraction}; "
            "need at least one train and one validation trajectory."
        )
    train_structured = {key: value[:split] for key, value in structured.items()}
    val_structured = {key: value[split:] for key, value in structured.items()}
    return train_structured, val_structured


@torch.no_grad()
def equivariance_diagnostics(model, batch):
    """
    Return max/relative errors for one decoder and encoder rotation check.
    """
    model.eval()
    R = o3.rand_matrix(device=batch["edge_vec"].device)
    batch_rot = rotate_batch_vectors(batch, R)

    z = torch.randn(
        batch["num_graphs"],
        model.zdim,
        device=batch["edge_vec"].device,
        dtype=batch["edge_vec"].dtype,
    )
    z_node = z.repeat_interleave(2, dim=0)

    h_dec = append_z_to_decoder_features(batch["h_dec_base"], z_node)
    h_dec_rot = append_z_to_decoder_features(batch_rot["h_dec_base"], z_node)
    mu, log_sigma = model.decoder(h_dec, batch["edge_index"], batch["edge_vec"], batch["edge_radial"])
    mu_rot, log_sigma_rot = model.decoder(
        h_dec_rot,
        batch_rot["edge_index"],
        batch_rot["edge_vec"],
        batch_rot["edge_radial"],
    )
    expected_mu_rot = mu @ R.T

    z_mu, z_logvar = model.encoder(
        batch["h_enc"],
        batch["edge_index"],
        batch["edge_vec"],
        batch["edge_radial"],
        batch["batch_index"],
    )
    z_mu_rot, z_logvar_rot = model.encoder(
        batch_rot["h_enc"],
        batch_rot["edge_index"],
        batch_rot["edge_vec"],
        batch_rot["edge_radial"],
        batch_rot["batch_index"],
    )

    return {
        "decoder_mu_rel_error": (
            (mu_rot - expected_mu_rot).norm() / (expected_mu_rot.norm() + 1e-12)
        ).item(),
        "decoder_mu_max_error": (mu_rot - expected_mu_rot).abs().max().item(),
        "decoder_log_sigma_max_error": (log_sigma_rot - log_sigma).abs().max().item(),
        "encoder_z_mu_max_error": (z_mu_rot - z_mu).abs().max().item(),
        "encoder_z_logvar_max_error": (z_logvar_rot - z_logvar).abs().max().item(),
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args)
    save_config(config, output_dir / "config.yaml")

    raw_dataset, _ = load_datasets(
        args.data_dir,
        n_datasets=args.n_trajectories,
        n_total=args.n_total,
        trajtype="bench",
    )
    q, v, r = extract_vars(raw_dataset)
    structured = construct_dqpipimririm_tensors(q, v, r)
    train_structured, val_structured = split_by_trajectory(structured, args.val_fraction)

    normalizer = E3VectorNormalizer.fit(train_structured)
    normalizer.save_json(output_dir / config.paths.scaler_name)
    train_structured = normalizer.transform(train_structured)
    val_structured = normalizer.transform(val_structured)

    train_ds = DimerE3Dataset(train_structured, flatten=True)
    val_ds = DimerE3Dataset(val_structured, flatten=True)

    collate = partial(collate_dimer_e3_graphs, boxsize=args.boxsize)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
    )

    model = build_model_from_config_e3(config).to(args.device)
    print("E3 normalizer:", normalizer.state_dict())
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    history = train_e3_cvae(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        lr=config.training.learning_rate,
        beta_max=config.training.beta_max,
        warmup_epochs=config.training.beta_warmup_epochs,
        grad_clip=config.training.grad_clip,
        weight_decay=config.training.weight_decay,
        save_path=output_dir / config.paths.checkpoint_name,
        early_stopping=config.training.early_stopping,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        validate_every=config.training.validate_every,
        device=args.device,
    )

    val_batch = move_graph_batch_to_device(next(iter(val_loader)), args.device)
    eq_diag = equivariance_diagnostics(model, val_batch)
    diagnostics = {
        "history": history,
        "normalizer": normalizer.state_dict(),
        "equivariance": eq_diag,
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "history.json", history)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "history": history,
            "normalizer": normalizer.state_dict(),
            "equivariance": eq_diag,
            "args": vars(args),
        },
        output_dir / "final.pt",
    )

    print("Equivariance diagnostics:", eq_diag)
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
