from pathlib import Path
from datetime import datetime
import pickle
import torch
import shutil
from .config import CVAEConfig, load_config, build_model_from_config, save_config


def load_run(run_dir, map_location="cpu"):
    run_dir = Path(run_dir)

    config = load_config(run_dir / "config.yaml")

    with open(run_dir / config.paths.scaler_name, "rb") as f:
        scalers = pickle.load(f)

    model = build_model_from_config(config)
    checkpoint = torch.load(run_dir / config.paths.checkpoint_name, map_location=map_location)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.attach_normalizers(**scalers)
    model.eval()

    return config, model, scalers


def create_run_dir(config_or_path: CVAEConfig | str | Path, output_root: str | Path | None = None) -> Path:
    if isinstance(config_or_path, CVAEConfig):
        config = config_or_path
        run_stem = config.experiment.name
        root = Path(output_root or config.paths.output_root)
    else:
        config_path = Path(config_or_path)
        config = load_config(config_path)
        run_stem = config_path.stem
        root = Path(output_root or config.paths.output_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_stem + "_" + timestamp

    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    if isinstance(config_or_path, CVAEConfig):
        save_config(config, run_dir / "config.yaml")
    else:
        shutil.copy(config_path, run_dir / "config.yaml")

    return run_dir


def latest_run(output_root: str | Path = "results/cvae", pattern: str = "*") -> Path:
    runs = sorted(Path(output_root).glob(pattern))
    runs = [run for run in runs if run.is_dir() and (run / "config.yaml").exists()]
    if not runs:
        raise FileNotFoundError(f"No CVAE runs found in {output_root!s} matching {pattern!r}.")
    return runs[-1]
