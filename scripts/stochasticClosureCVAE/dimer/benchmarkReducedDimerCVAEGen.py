"""
Long reduced-dimer rollout generator for CVAE models trained with the
noiseSampler/training config system (results/<cond>/<NNN>_<name>/).

Mirrors benchmarkReducedDimerGen.py in physical setup and output format.
Fixed defaults: boxsize=5, stride=1, device=cpu.

Output layout
-------------
  cvaeRuns/<cond>/<id>_<config.name>/
      model_config.yaml              ← provenance copy (always)
      <N>x<tfinal>/                  ← simulation sub-dir (no --output-name)
          parameters
          simMoriZwanzigReduced_*
      <N>x<tfinal>_<tag>/            ← simulation sub-dir (with --output-name)
          parameters
          simMoriZwanzigReduced_*

Usage
-----
    python benchmarkReducedDimerCVAEGen.py --cond pipimririm --id 001
    python benchmarkReducedDimerCVAEGen.py --cond local_dqidpipimmrimm --id 001 \\
        --num-simulations 50 --tfinal 5000 --output-name "short_test"
"""
import argparse
import multiprocessing
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
from deepRD.noiseSampler.cvae.checkpoints import load_run
from deepRD.potentials import pairBistable

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
RESULTS_ROOT = _REPO_ROOT / "deepRD" / "noiseSampler" / "training" / "results"

DEFAULT_BENCHMARK_DIR = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark"
DEFAULT_OUTPUT_ROOT = "/group/ag_cmb/scratch/maojrs/stochasticClosure/dimerGlobal/boxsize5"

BOXSIZE = 5.0
STRIDE = 1


# ── Inference-time sampler-wrapper registry ───────────────────────────────────
# Each wrapper is applied AFTER the model is loaded and before it is handed to the
# integrator. Signature: wrap(model, run_dir, args) -> model. Add new inference
# fixes here (fixa/rbf/student) by importing their loader lazily inside the fn, so
# the canonical Gen script stays importable without any tests/ campaign module.
def _wrap_none(model, run_dir, args):
    return model


def _wrap_fixm(model, run_dir, args):
    """Conditional-mean bias correction (CVAE_LF_M). Run-dir mode only."""
    if run_dir is None:
        raise ValueError("--sampler-wrapper fixm requires run-dir mode (not --checkpoint).")
    if not args.wrapper_bias_path:
        raise ValueError("--sampler-wrapper fixm requires --wrapper-bias-path.")
    # Lazy import: keeps the canonical script free of a hard tests/ dependency.
    sys.path.insert(0, str(_REPO_ROOT / "tests" / "condmean_rollout"))
    from fixm_sampler import load_cvae_lf_m_run
    _, wrapped, _ = load_cvae_lf_m_run(
        str(run_dir), bias_path=args.wrapper_bias_path,
        gain=args.wrapper_gain, boxsize=BOXSIZE)
    return wrapped


WRAPPERS = {"none": _wrap_none, "fixm": _wrap_fixm}


def load_generic_model(args):
    """Load a bare checkpoint.pt (+ sibling scalers.pkl) — the generic-mode path."""
    import joblib
    from deepRD.noiseSampler.cvaeSampler import CVAE_LF, CVAESampler
    cls = CVAE_LF if args.model_type == "CVAE_LF" else CVAESampler
    m = cls(zdim=args.zdim, system_type="dimer", cond_type=args.cond, hidden=tuple(args.hidden))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    m.load_state_dict(state.get("model_state", state), strict=False)
    scalers = joblib.load(str(args.checkpoint).replace("checkpoint.pt", "scalers.pkl"))
    m.attach_normalizers(**scalers)
    m.set_temps(Tr=args.Tr, Tz=args.Tz)
    m.eval()
    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate long reduced dimer rollouts from a CVAE trained with the config system."
    )
    parser.add_argument(
        "--cond", required=True,
        help="Conditioning type. In run-dir mode: matches results/<cond>/. In generic "
             "(--checkpoint) mode: the model's cond_type.",
    )
    parser.add_argument(
        "--id", default=None,
        help="Run-dir mode: 3-digit run-ID prefix to match results/<cond>/<id>_*/ (e.g. '001'). "
             "Required unless --checkpoint (generic mode) is given.",
    )
    # ── Generic-checkpoint mode: bare checkpoint.pt, no results/<cond>/<id>/ dir. ──
    parser.add_argument(
        "--checkpoint", default=None,
        help="Generic mode: path to a bare checkpoint.pt (with sibling scalers.pkl) to roll out "
             "without the results/<cond>/<id> convention. Output goes to a flat cvaeRuns/<label>/ dir.",
    )
    parser.add_argument("--model-type", default="CVAE_LF", choices=["CVAE_LF", "CVAE"],
                        help="Generic mode only: model class (default CVAE_LF).")
    parser.add_argument("--zdim", type=int, default=3, help="Generic mode only: latent dim.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Generic mode only: decoder hidden dims.")
    parser.add_argument("--label", default=None,
                        help="Generic mode: output subdir name under cvaeRuns/ (default: checkpoint's parent name).")
    # ── Inference-time sampler wrapper (any post-load correction). ──
    parser.add_argument("--sampler-wrapper", default="none", choices=sorted(WRAPPERS),
                        help="Inference-time fix applied to the loaded model before rollout. "
                             "'none' (default) = raw model; 'fixm' = conditional-mean bias correction.")
    parser.add_argument("--wrapper-bias-path", default=None,
                        help="fixm: path to the bias .npz (from measure_bias.py).")
    parser.add_argument("--wrapper-gain", type=float, default=1.0,
                        help="Gain for the sampler wrapper (fixm/fixa), default 1.0.")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--output-name", default=None,
        help=(
            "Optional tag appended to the simulation sub-directory name. "
            "Sub-directory is always <N>x<tfinal>/ (no tag) or <N>x<tfinal>_<tag>/ (with tag)."
        ),
    )
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--tfinal", type=float, default=10000.0)
    parser.add_argument("--equilibration-steps", type=int, default=10000)
    parser.add_argument("--Tr", type=float, default=1.0, help="Decoder noise temperature.")
    parser.add_argument("--Tz", type=float, default=1.0, help="Latent prior temperature.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers. Defaults to cpu_count - 1.")
    parser.add_argument("--make-diags", action="store_true", default=True,
                        help="Run diagnostics automatically after simulations (default: True).")
    parser.add_argument("--no-diags", dest="make_diags", action="store_false",
                        help="Skip automatic diagnostics.")
    parser.add_argument("--n-bench-trajs", type=int, default=100,
                        help="Benchmark trajectories loaded for diagnostics.")
    return parser.parse_args()


def find_run_dir(cond: str, run_id: str) -> Path:
    """Return the unique run directory matching results/<cond>/<run_id>_*/."""
    cond_dir = RESULTS_ROOT / cond
    if not cond_dir.exists():
        raise FileNotFoundError(f"Conditioning directory not found: {cond_dir}")
    matches = [p for p in sorted(cond_dir.iterdir()) if p.is_dir() and p.name.startswith(run_id)]
    if not matches:
        raise FileNotFoundError(f"No run directory with prefix '{run_id}' in {cond_dir}.")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous: multiple run directories match '{run_id}' in {cond_dir}: {matches}"
        )
    return matches[0]


def resolve_output_dirs(args, run_dir: Path) -> tuple[Path, Path]:
    """
    Return (base_dir, sim_dir).

    base_dir : cvaeRuns/<cond>/<run_dir.name>/   — mirrors the model directory
    sim_dir  : base_dir/<N>x<tfinal>_<tag>/      — only when --output-name is given
               base_dir/                          — otherwise
    """
    base_dir = Path(DEFAULT_OUTPUT_ROOT) / "cvaeRuns" / args.cond / run_dir.name
    tfinal_str = f"{args.tfinal:g}"
    run_tag = f"{args.num_simulations}x{tfinal_str}"
    if args.output_name:
        run_tag = f"{run_tag}_{args.output_name}"
    sim_dir = base_dir / run_tag
    return base_dir, sim_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            print(f"Folder {output_dir} already exists. Previous files may be overwritten. Continue? y/n")
            if input().strip().lower() != "y":
                sys.exit(0)
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_parallel_sim(simnumber, args_dict, parameters, basefilename):
    args = argparse.Namespace(**args_dict)
    set_seed(int(simnumber))
    torch.set_num_threads(1)

    if args.checkpoint:                         # generic-checkpoint mode
        model = load_generic_model(args)
        run_dir = None
        conditioned_on = args.cond
    else:                                       # run-dir (config-system) mode
        config, model, _ = load_run(Path(args.run_dir), map_location="cpu")
        model.set_temps(Tr=args.Tr, Tz=args.Tz)
        run_dir = Path(args.run_dir)
        conditioned_on = config.data.conditioning

    # Apply the inference-time sampler wrapper (identity for 'none').
    model = WRAPPERS[args.sampler_wrapper](model, run_dir, args)

    particle_diameter = 0.5
    x0 = 1.0 * particle_diameter
    rad = 1.0 * particle_diameter
    scalefactor = 2

    particle1 = deepRD.particle([0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle2 = deepRD.particle([x0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], mass=parameters["mass"])
    particle_list = deepRD.particleList([particle1, particle2])

    integrator = langevinNoiseSamplerDimerGlobal(
        parameters["dt"],
        STRIDE,
        args.tfinal,
        parameters["Gamma"],
        model,
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

    generic = args.checkpoint is not None
    tfinal_str = f"{args.tfinal:g}"
    run_tag = f"{args.num_simulations}x{tfinal_str}"
    if args.output_name:
        run_tag = f"{run_tag}_{args.output_name}"

    if generic:                                 # ── generic-checkpoint mode ──
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"--checkpoint not found: {ckpt}")
        args.run_dir = None
        run_dir = None
        config = None
        conditioned_on = args.cond
        model_type = args.model_type
        label = args.label or ckpt.parent.name
        base_dir = Path(DEFAULT_OUTPUT_ROOT) / "cvaeRuns" / label
        sim_dir = base_dir / run_tag
    else:                                       # ── run-dir (config-system) mode ──
        if not args.id:
            raise SystemExit("--id is required in run-dir mode (or pass --checkpoint for generic mode).")
        run_dir = find_run_dir(args.cond, args.id)
        args.run_dir = str(run_dir)  # absolute path for worker subprocesses
        config, _, _ = load_run(run_dir, map_location="cpu")
        conditioned_on = config.data.conditioning
        model_type = config.model.model_type
        base_dir, sim_dir = resolve_output_dirs(args, run_dir)

    benchmark_dir = Path(args.benchmark_dir)

    # Create base dir silently (shared across simulation runs of this model).
    base_dir.mkdir(parents=True, exist_ok=True)
    # The sim dir may be the same as base_dir or a sub-directory.
    prepare_output_dir(sim_dir, args.overwrite)

    parameters = analysisTools.readParameters(str(benchmark_dir / "parameters"))

    if float(parameters["boxsize"]) != BOXSIZE:
        raise ValueError(
            f"Benchmark boxsize {parameters['boxsize']} does not match expected {BOXSIZE}."
        )

    parameter_dictionary = {
        "numFiles": args.num_simulations,
        "dt": parameters["dt"],
        "Gamma": parameters["Gamma"],
        "KbT": parameters["KbT"],
        "mass": parameters["mass"],
        "tfinal": args.tfinal,
        "stride": STRIDE,
        "boxsize": parameters["boxsize"],
        "boundaryType": parameters["boundaryType"],
        "equilibrationSteps": args.equilibration_steps,
        "conditionedOn": conditioned_on,
        "modelType": model_type,
        "runDir": str(run_dir) if run_dir else str(args.checkpoint),
        "samplerWrapper": args.sampler_wrapper,
        "Tr": args.Tr,
        "Tz": args.Tz,
    }
    analysisTools.writeParameters(str(sim_dir / "parameters"), parameter_dictionary)
    # Provenance copy lives in base_dir (shared, written once). Only run-dir mode
    # has a config.yaml to copy.
    if run_dir is not None:
        shutil.copy2(run_dir / "config.yaml", base_dir / "model_config.yaml")

    basefilename = str(sim_dir / "simMoriZwanzigReduced_")
    sim_numbers = list(range(args.start_index, args.start_index + args.num_simulations))

    num_workers = args.num_workers if args.num_workers is not None else max(multiprocessing.cpu_count() - 1, 1)

    print(f"Simulation for r_{{n+1}} | {conditioned_on} begins ...")
    print(f"Model source     : {run_dir if run_dir else args.checkpoint}")
    print(f"Output directory : {sim_dir}")
    print(f"Model type       : {model_type}  |  sampler-wrapper: {args.sampler_wrapper}")
    print(f"Simulations      : {len(sim_numbers)}  |  tfinal: {args.tfinal}  |  equil: {args.equilibration_steps}")
    print(f"Workers          : {num_workers}")

    worker = partial(
        run_parallel_sim,
        args_dict=vars(args),
        parameters=parameters,
        basefilename=basefilename,
    )

    if num_workers == 1:
        for simnumber in sim_numbers:
            worker(simnumber)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(worker, sim_numbers)

    if args.make_diags:
        import matplotlib
        matplotlib.use("Agg")
        from deepRD.noiseSampler.diagnostics.rollout_diags import run_all_diagnostics
        # run-dir mode: results/<cond>/<id_name>/diags/<simrun>/;  generic mode:
        # alongside the rollout (sim_dir/diagnostics/), since there is no run dir.
        diag_dir = (run_dir / "diags" / sim_dir.name) if run_dir is not None else (sim_dir / "diagnostics")
        print("\nRunning diagnostics...")
        run_all_diagnostics(
            sim_dir       = sim_dir,
            bench_dir     = Path(args.benchmark_dir),
            n_trajs       = args.num_simulations,
            n_bench_trajs = args.n_bench_trajs,
            label         = sim_dir.name,
            boxsize       = BOXSIZE,
            diag_dir      = diag_dir,
        )
        print(f"Diagnostics saved to: {diag_dir}")


if __name__ == "__main__":
    main()
