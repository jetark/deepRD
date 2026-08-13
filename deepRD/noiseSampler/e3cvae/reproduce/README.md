# Reproducing the E3 dimer noise sampler (deep4 + Fix-A)

End-to-end pipeline for the paper's SO(3)-equivariant conditional VAE noise
sampler on the bistable dimer: a `deep4` `E3DimerCVAE` trained from a fixed seed,
plus the fluctuation-dissipation ("Fix-A") correction whose scalar gain is
selected on the velocity marginal. The published model is **seed 302, gain 0.45**.

Everything here imports only the public `deepRD` API — no test-tree or scratch
dependencies. The established interpreter is
`/srv/data/jakut77/miniconda3/envs/pyta/bin/python`.

## Stages

| stage | module | in → out |
|-------|--------|----------|
| 1. train | `train` | seed → `results_e3/repro_s302/` (`config.yaml`, `checkpoint.pt`, `e3_normalizer.json`, `final.pt`) |
| 2. calibrate | `calibrate_gains` | checkpoint → `fd_gains.json` (3 friction gains, teacher-forced) |
| 3. sweep + select | `gain_sweep` | gains → smallest scalar gain landing `dvx_std ≈ 1.0×` benchmark |
| 4. rollout | `generate_rollout` | model + gains + gain → long reduced-dimer trajectories |

`fd_sampler.E3DimerRolloutSamplerFD` is the correction itself (used by stages 3–4);
`run_pipeline` chains all four.

## One-command reproduction

```bash
PY=/srv/data/jakut77/miniconda3/envs/pyta/bin/python
cd /home/mi/jakut77/cgr/deepRD

# full run reproducing the PUBLISHED model (training needs a GPU; rollout uses CPU workers)
$PY -m deepRD.noiseSampler.e3cvae.reproduce.run_pipeline --seed 302 --gain 0.45

# ...or let the sweep auto-select the gain by the criterion (lands on the 0.42
# end of the plateau; see "Selection criterion" below):
$PY -m deepRD.noiseSampler.e3cvae.reproduce.run_pipeline --seed 302
```

To reuse an already-trained checkpoint instead of retraining:

```bash
$PY -m deepRD.noiseSampler.e3cvae.reproduce.run_pipeline --seed 302 \
    --run-dir deepRD/noiseSampler/training/results_e3/ens_e3_s302 --skip-train
```

## Stage-by-stage (explicit)

```bash
RUN=deepRD/noiseSampler/training/results_e3/repro_s302

# 1. train the deep4 model from seed 302 (50 epochs, 200 trajectories)
$PY -m deepRD.noiseSampler.e3cvae.reproduce.train --seed 302 --out $RUN

# 2. calibrate the three friction gains (teacher-forced on the held-out split)
$PY -m deepRD.noiseSampler.e3cvae.reproduce.calibrate_gains \
    --run-dir $RUN --out-gains $RUN/fd_gains.json

# 3. sweep the scalar gain and auto-select on the velocity marginal
$PY -m deepRD.noiseSampler.e3cvae.reproduce.gain_sweep \
    --run-dir $RUN --gains-path $RUN/fd_gains.json \
    --gain-list 0.0 0.40 0.42 0.45 0.47 0.50 \
    --num-sims 20 --tfinal 300 --equil 2500 --workers 11 \
    --write-selected $RUN/selected_gain.txt

# 4. generate the production rollout at the selected gain (0.45)
$PY -m deepRD.noiseSampler.e3cvae.reproduce.generate_rollout \
    --run-dir $RUN --gains-path $RUN/fd_gains.json --gain 0.45 \
    --output-name repro_s302_fd_g045 --run-name 60x8000 \
    --num-simulations 60 --tfinal 8000 --equilibration-steps 5000 \
    --num-workers 11 --overwrite
```

The rollout lands in
`<output-root>/repro_s302_fd_g045/60x8000/` (default output-root is the cluster
`cvaeRuns` dir) as `simMoriZwanzigReduced_*` plus a `parameters` file — exactly
what `dimerRolloutDiagnostics.py` and the FPT tooling consume for scoring.

## Selection criterion

Fix-A is calibrated to restore the missing noise-velocity friction, so its scalar
gain is chosen on the **velocity marginal only**: the smallest gain whose
bond-velocity std `dvx_std` is within 1% of the benchmark (`--select-tol`).
First-passage kinetics are *not* used to pick the gain — they are scored
downstream on the full rollout.

For seed 302 the velocity marginal is corrected across a **plateau** (every gain
in 0.42–0.50 lands within ~1% on `dvx_std`), so the rule selects the low end of
that plateau, **gain 0.42**. The published model uses **gain 0.45**, the
within-plateau value whose first-passage rate ratio `k` best matches the
benchmark (4.28 vs 4.25); pass `--gain 0.45` to `run_pipeline`/`generate_rollout`
to reproduce it exactly. Because the sweep is noisy (20 short rollouts), treat
the auto-selected value as advisory near the plateau edge.

## Reproducibility notes

- **Data + training are seed-deterministic.** `set_seed(seed)` seeds
  Python/NumPy/torch; the 200 training trajectories are drawn with
  `np.random.choice` from the seeded global NumPy RNG and the train/val split is
  index-deterministic. Weight init, batch shuffling and the reparameterisation
  noise use the seeded torch RNG.
- **One caveat:** CUDA/cuDNN kernel reductions are not forced bitwise-deterministic,
  so a retrain from the seed reproduces a *statistically identical* model, not a
  bit-identical one. For an exact result, load the shipped `checkpoint.pt`
  directly (stage 1 is then skippable).
- **Config of record:** `configs/deep4.yaml` (E3DimerCVAE, irreps
  `32x0e + 16x1o + 8x2e`, `dqpipimririm` conditioning, 4 decoder layers, 200
  trajectories, 50 epochs, β warmup 10). `dataset_dir` must point at the dimer
  benchmark data.
```
