from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


@dataclass
class ExperimentSection:
    name: str
    seed: int = 123
    frame: str = "global"  # "global" or "local"


@dataclass
class SystemSection:
    system_type: str
    boxsize: float
    dt: float
    step: int = 1


@dataclass
class DataSection:
    conditioning: str
    n_lags: int
    train_trajectories: int
    val_fraction: float
    dataset_dir: str | None = None
    total_trajectories: int | None = None
    scaler_type: str = "standard"
    weights: str | None = None


@dataclass
class ModelSection:
    model_type: str
    input_dim: int
    latent_dim: int
    cond_dim: int
    hidden_dims: list[int]
    activation: str = "silu"
    layer_norm: bool = True
    standard_prior: bool = True
    hidden_irreps: str | None = None
    isotropic: bool = False
    lag2: bool = False
    n_dec_layers: int = 2
    dec_nonlinear_head: bool = False
    add_dx_scalar: bool = False
    radial_num_basis: int = 16
    r_cut: float = 2.0
    lmax: int = 2
    add_linear_response_mean: bool = False
    gate_scale: float = 0.05


@dataclass
class TrainingSection:
    batch_size: int
    epochs: int
    learning_rate: float
    beta_max: float
    beta_warmup_epochs: int
    window_length: int = 1
    free_bits: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float | None = None
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-3
    validate_every: int = 1
    weights_for_training: bool = False
    num_workers: int = 0
    # MDN specific parameters
    lambda_gate: float | None = None
    lambda_comp: float | None = None


@dataclass
class PathsSection:
    output_root: str = "results/cvae"
    checkpoint_name: str = "checkpoint.pt"
    scaler_name: str = "scalers.pkl"


@dataclass
class CVAEConfig:
    experiment: ExperimentSection
    system: SystemSection
    data: DataSection
    model: ModelSection
    training: TrainingSection
    paths: PathsSection = field(default_factory=PathsSection)


def config_to_dict(config: CVAEConfig) -> dict:
    return asdict(config)


def save_config(config: CVAEConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)


def load_config(path: str | Path) -> CVAEConfig:
    path = Path(path)

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    return CVAEConfig(
        experiment=ExperimentSection(**raw["experiment"]),
        system=SystemSection(**raw["system"]),
        data=DataSection(**raw["data"]),
        model=ModelSection(**raw["model"]),
        training=TrainingSection(**raw["training"]),
        paths=PathsSection(**raw.get("paths", {})),
    )


def build_model_from_config(config: CVAEConfig):
    """Builds the CVAE model based on the provided configuration."""
    
    if config.model.model_type == "CVAE":
        from deepRD.noiseSampler.cvaeSampler import CVAESampler as model_class
    elif config.model.model_type == "CVAE_LF":
        from deepRD.noiseSampler.cvaeSampler import CVAE_LF as model_class
    elif config.model.model_type == "CVAE_SP":
        from deepRD.noiseSampler.cvaeSampler import CVAE_SP as model_class
    elif config.model.model_type == "CVAE_MDN":
        from deepRD.noiseSampler.cvaeSampler import CVAE_MDN as model_class
    elif config.model.model_type == "CVAE_Inv":
        from deepRD.noiseSampler.cvaeSampler import CVAE_Inv as model_class
    elif config.model.model_type == "CVAE_E3":
        from deepRD.noiseSampler.cvaeSampler import CVAESampler_E3 as model_class
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    model = model_class(
        zdim=config.model.latent_dim,
        system_type=config.system.system_type,
        cond_type=config.data.conditioning,
        hidden=config.model.hidden_dims,
    )

    return model


def build_model_from_config_e3(config: CVAEConfig):
    """
    Build the e3nn-based E3DimerCVAE from the shared CVAEConfig schema.

    This intentionally accepts only the subset currently meaningful for the
    graph E3 model. The flat input_dim/hidden_dims fields are retained in the
    config for compatibility with existing YAMLs, but E3DimerCVAE uses
    latent_dim and an optional hidden_irreps attribute instead.
    """
    if config.model.model_type not in ("E3DimerCVAE", "CVAE_E3NN", "E3_CVAE"):
        raise ValueError(
            "build_model_from_config_e3 expects model_type in "
            "{'E3DimerCVAE', 'CVAE_E3NN', 'E3_CVAE'}, got "
            f"{config.model.model_type!r}"
        )
    if config.system.system_type != "dimer":
        raise ValueError("E3DimerCVAE currently supports only system_type='dimer'.")
    if config.data.conditioning != "dqpipimririm":
        raise ValueError("E3DimerCVAE currently expects conditioning='dqpipimririm'.")

    from deepRD.noiseSampler.e3cvae.model import E3DimerCVAE

    hidden_irreps = config.model.hidden_irreps or "32x0e + 16x1o + 8x2e"
    return E3DimerCVAE(
        zdim=config.model.latent_dim,
        hidden_irreps=hidden_irreps,
        isotropic=getattr(config.model, "isotropic", False),
        lag2=getattr(config.model, "lag2", False),
        n_dec_layers=getattr(config.model, "n_dec_layers", 2),
        dec_nonlinear_head=getattr(config.model, "dec_nonlinear_head", False),
        add_dx_scalar=getattr(config.model, "add_dx_scalar", False),
        radial_num_basis=getattr(config.model, "radial_num_basis", 16),
        r_cut=getattr(config.model, "r_cut", 2.0),
        lmax=getattr(config.model, "lmax", 2),
        add_linear_response_mean=getattr(config.model, "add_linear_response_mean", False),
        gate_scale=getattr(config.model, "gate_scale", 0.05),
    )
