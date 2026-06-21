# E3DimerCVAE Implementation Summary

This directory contains the first e3nn-based E(3)/SO(3)-equivariant CVAE for
the 3D dimer Langevin noise sampler. The model is designed for the global-frame
conditioning scheme `dqpipimririm`, without a local-frame transform. All vector
quantities stay in Cartesian coordinates and equivariance is enforced by e3nn
irreps, spherical harmonics, and tensor-product message passing.

The current implementation targets the auxiliary-variable one-step model:

- conditioning at time `n`: positions, velocities, current auxiliary variables,
  and previous-step history
- target at time `n+1`: `r1_next`, `r2_next`
- graph structure: one two-node graph per dimer sample
- output: beadwise vector mean and axial Gaussian covariance aligned with the
  dimer bond axis

The public model class is `E3DimerCVAE` in `model.py`.

## Conditioning Scheme

The intended conditioning name is `dqpipimririm`. In this implementation it is
stored structurally instead of as one flattened MLP vector.

For each effective time index `n`, the structured fields are:

- `q1`, `q2`: bead positions at `n`
- `v1_n`, `v2_n`: current velocities
- `v1_nm1`, `v2_nm1`: previous velocities
- `r1_n`, `r2_n`: current auxiliary variables
- `r1_nm1`, `r2_nm1`: previous auxiliary variables
- `r1_next`, `r2_next`: target auxiliary variables at `n+1`

The effective time slice is `n = 1 .. T-2`, because both previous and next
auxiliary variables are required.

The bond vector used by the graph is:

```text
dq = minimal_image(q2 - q1, boxsize)
```

No local frame is built. The model sees global Cartesian vectors, and its
operations preserve the correct transformation laws.

## File Map

### `tools.py`

This file handles structured dimer data, graph batching, e3nn feature packing,
latent appending, and rotation utilities.

Key functions/classes:

- `radial_embedding(edge_vec, num_basis=16, r_cut=5.0)`
  - Converts edge lengths `|dq|` into Gaussian radial basis features.
  - Output shape: `[num_edges, 16]`.
  - These radial features are invariant scalar edge attributes used by the
    radial MLP in message passing.

- `split_dimer_particles(x)`
  - Splits interleaved dimer trajectories:
    `[bead1_t0, bead2_t0, bead1_t1, bead2_t1, ...]`.
  - Input shape: `[n_traj, 2*T, 3]`.
  - Output: two tensors `[n_traj, T, 3]`.

- `construct_dqpipimririm_tensors(q, v, r)`
  - Builds the structured global-frame conditioning dictionary from interleaved
    dimer trajectories.
  - Returns tensors with shape `[n_traj, T-2, 3]`.
  - This is the E3 equivalent of the old flat/local conditioning builder.

- `flatten_structured_dqpipimririm(data)`
  - Flattens `[n_traj, T_eff, 3]` fields to `[N, 3]`, where
    `N = n_traj * T_eff`.

- `DimerE3Dataset`
  - Dataset wrapper around the structured fields.
  - Each item is a dictionary of one-sample 3D vector fields.
  - It intentionally does not return flat `(r_next, c)` pairs.

- `collate_dimer_e3_graphs(samples, boxsize=5.0)`
  - DataLoader `collate_fn`.
  - Stacks sample dictionaries and calls `build_dimer_graph_batch`.
  - Emits the graph-batch dictionary consumed by `E3DimerCVAE`.

- `pack_e3_features(scalars, vectors)`
  - Packs e3nn node features as:
    `[flattened vector irreps, scalar irreps]`.
  - For irreps `"4x1o + 4x0e"`, the packed feature dimension is
    `4*3 + 4 = 16`.

- `build_dimer_graph_batch(...)`
  - Core graph constructor.
  - Builds one two-node graph per dimer sample:
    - node 0: bead 1
    - node 1: bead 2
    - edge `0 -> 1`: `dq`
    - edge `1 -> 0`: `-dq`
  - Uses minimal-image periodic convention for `dq`.
  - Decoder node vectors:
    - `v_i_n`
    - `v_i_nm1`
    - `r_i_n`
    - `r_i_nm1`
  - Decoder scalar features:
    - norms of the four decoder vectors
  - Encoder node vectors:
    - decoder vectors plus target `r_i_next`
  - Encoder scalar features:
    - norms of the five encoder vectors
  - Returns:
    - `h_dec_base`: `[B*2, dim("4x1o + 4x0e")]`
    - `h_enc`: `[B*2, dim("5x1o + 5x0e")]` or `None`
    - `edge_index`: `[2, 2*B]`
    - `edge_vec`: `[2*B, 3]`
    - `edge_radial`: `[2*B, 16]`
    - `bond_unit_node`: `[B*2, 3]`
    - `batch_index`: `[B*2]`
    - `r_next`: `[B*2, 3]` or `None`
    - `num_graphs`: `B`

- `append_z_to_decoder_features(h_dec_base, z_node)`
  - Appends invariant latent variables to scalar channels.
  - This is valid because `z` is a `0e` invariant.
  - Input:
    - `h_dec_base`: `[B*2, 16]`
    - `z_node`: `[B*2, zdim]`
  - Output irreps match `"4x1o + (4+zdim)x0e"`.

- `rotate_e3_features(x, irreps, R)`
  - Rotates packed e3nn features using `Irreps.D_from_matrix(R)`.

- `rotate_batch_vectors(batch, R, ...)`
  - Rotates the full graph batch for equivariance tests.
  - Rotates vector irreps, edge vectors, targets, and bond axes.
  - Leaves scalar/radial features unchanged.

### `messagepassing.py`

This file defines the reusable e3nn message-passing layer.

Key pieces:

- `RadialMLP`
  - MLP mapping invariant radial basis features to tensor-product weights.
  - Output dimension is `self.tp.weight_numel`.

- `E3MessageLayer`
  - Implements one equivariant graph convolution layer.
  - Constructor arguments:
    - `irreps_in`
    - `irreps_out`
    - `lmax=2`
    - `radial_dim=16`
    - `radial_hidden=64`
  - Internal components:
    - `self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)`
    - `o3.FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_out)`
    - radial MLP producing tensor-product weights
    - e3nn `o3.Linear` self-connection
    - `NormActivation`, which applies nonlinearities without coordinate-wise
      nonlinearities on vector/tensor irreps
  - Forward pass:
    1. Compute spherical harmonics from `edge_vec`.
    2. Compute radial weights from `edge_radial`.
    3. Tensor-product source node features with edge spherical harmonics.
    4. Sum messages into destination nodes with `index_add_`.
    5. Add equivariant self-connection.
    6. Apply norm activation.

This is the main operation that enforces equivariant mixing of scalar, vector,
and higher-order tensor channels.

### `encoder.py`

This file defines the invariant CVAE encoder.

Key pieces:

- `extract_0e(x, irreps)`
  - Extracts all even scalar (`0e`) channels from packed e3nn features.
  - These are true rotation/reflection invariants.

- `E3InvariantEncoder`
  - Input irreps: `"5x1o + 5x0e"`.
  - Hidden irreps default: `"32x0e + 16x1o + 8x2e"`.
  - Contains two `E3MessageLayer`s.
  - After message passing:
    1. Extracts only `0e` scalar channels.
    2. Pools over the two dimer nodes by mean.
    3. Applies an ordinary scalar MLP.
    4. Outputs `z_mu` and `z_logvar`.
  - Output shapes:
    - `z_mu`: `[B, zdim]`
    - `z_logvar`: `[B, zdim]`

The encoder posterior `q(z | c, r_next)` is invariant by construction because
only `0e` channels are pooled and passed into the MLP.

### `decoder.py`

This file defines the equivariant decoder.

Key class:

- `E3EquivariantDecoder`
  - Input irreps: `"4x1o + (4+zdim)x0e"`.
  - Hidden irreps default: `"32x0e + 16x1o + 8x2e"`.
  - Contains two `E3MessageLayer`s.
  - Output heads:
    - `vector_head = o3.Linear(hidden, "1x1o")`
      - Produces beadwise vector means `mu`.
      - Shape: `[B*2, 3]`.
      - Rotates equivariantly.
    - `sigma_head = o3.Linear(hidden, "2x0e")`
      - Produces two invariant scalar log-scales per bead.
      - Shape: `[B*2, 2]`.
      - Column 0: `log_sigma_parallel`.
      - Column 1: `log_sigma_perp`.

The decoder produces a 3D vector likelihood per bead, but its covariance is
axially symmetric around the dimer bond.

### `model.py`

This file defines the top-level CVAE module.

Key class:

- `E3DimerCVAE(zdim, hidden_irreps=...)`
  - Holds:
    - `self.encoder = E3InvariantEncoder(...)`
    - `self.decoder = E3EquivariantDecoder(...)`
  - `reparameterize(z_mu, z_logvar)`
    - Standard CVAE reparameterization:
      `z = z_mu + eps * exp(0.5*z_logvar)`.

Forward path:

```text
batch["h_enc"] -> encoder -> z_mu, z_logvar
z = reparameterize(z_mu, z_logvar)
z_node = repeat z onto both nodes
h_dec = append_z_to_decoder_features(batch["h_dec_base"], z_node)
h_dec -> decoder -> mu, log_sigma
```

Forward returns:

- `mu`: `[B*2, 3]`
- `log_sigma`: `[B*2, 2]`
- `log_sigma_para`: `[B*2, 1]`
- `log_sigma_perp`: `[B*2, 1]`
- `z_mu`: `[B, zdim]`
- `z_logvar`: `[B, zdim]`

Sampling path:

- `sample_torch(batch, Tr=1.0, Tz=1.0)`
  - Does not use the encoder.
  - Samples `z ~ N(0, I) * Tz`.
  - Decodes `mu` and axial log-sigmas.
  - Samples from the axial Gaussian with optional noise temperature `Tr`.
  - Returns `(r_next, mu, log_sigma)`.

The public rollout wrapper that accepts raw integrator state vectors is not
implemented yet. The current sampling API expects an already-built graph batch.

### `axial_covariance.py` and `axial_cov.py`

`axial_covariance.py` contains the axial Gaussian math. `axial_cov.py` is a
compatibility re-export module.

Key functions:

- `bond_unit_from_edge_vec(edge_vec, num_graphs)`
  - Reconstructs node-level unit bond axes from graph edge vectors.
  - Assumes the first `B` edges are `0 -> 1` with `dq`.
  - Returns `[B*2, 3]`.

- `axial_vector_nll(y, mu, bond_unit, log_sigma_para, log_sigma_perp)`
  - Per-node 3D Gaussian NLL with covariance:

```text
Sigma = sigma_para^2 ee^T + sigma_perp^2 (I - ee^T)
```

  - Decomposes residual into:
    - one parallel scalar component
    - two perpendicular dimensions
  - Returns one 3D-vector NLL per node: `[B*2]`.

- `sample_axial_gaussian(mu, bond_unit, log_sigma_para, log_sigma_perp, noise_scale=1.0)`
  - Samples parallel noise along the bond axis.
  - Samples a 3D normal and projects out the parallel component for the
    perpendicular noise.
  - Returns sampled vectors `[B*2, 3]`.

### `losses.py`

This file defines loss utilities for the E3 CVAE.

Key functions:

- `isotropic_vector_nll(y, mu, log_sigma)`
  - Fallback/simple 3D isotropic Gaussian vector NLL.
  - Currently not used by the default training path.

- `standard_gaussian_kl(z_mu, z_logvar)`
  - KL divergence from `q(z | c, y)` to the standard Gaussian prior.
  - Returns graph-level KL: `[B]`.

- `e3_cvae_axial_loss(outputs, batch, beta=1.0)`
  - Main training loss.
  - Uses `axial_vector_nll` for each bead/node.
  - Sums the two bead NLLs per graph:

```text
nll_graph = nll_node.reshape(B, 2).sum(dim=-1)
```

  - This is a 6D dimer observation likelihood:
    - bead 1: 3D vector likelihood
    - bead 2: 3D vector likelihood
  - Averages graph NLL and KL over the batch:

```text
loss = mean(nll_graph) + beta * mean(kl_graph)
```

### `normalization.py`

This file defines equivariance-preserving scaling.

Key class:

- `E3VectorNormalizer`
  - Fits scalar RMS scales for vector families:
    - velocities: `v1_n`, `v2_n`, `v1_nm1`, `v2_nm1`
    - auxiliary variables and targets:
      `r1_n`, `r2_n`, `r1_nm1`, `r2_nm1`, `r1_next`, `r2_next`
  - Does not subtract means.
  - Does not scale x/y/z independently.
  - Leaves `q1` and `q2` in physical units so the edge vector and radial basis
    preserve physical separation scale.
  - Saves/loads JSON state:
    - `velocity_scale`
    - `auxiliary_scale`

This is important because ordinary `StandardScaler` would destroy equivariance by
using coordinate-dependent centering and scaling.

### `training.py`

This file contains the graph-batch training loop.

Key functions:

- `move_graph_batch_to_device(batch, device)`
  - Moves tensor values in a graph-batch dictionary to CPU/GPU.

- `beta_schedule(epoch, beta_max, warmup_epochs)`
  - Linear warmup to `beta_max`.

- `train_e3_cvae(...)`
  - Optimizer: `AdamW`.
  - Scheduler: `CosineAnnealingLR`.
  - Loss: `e3_cvae_axial_loss`.
  - Supports:
    - KL warmup
    - gradient clipping
    - validation
    - early stopping
    - checkpoint saving
  - Expects DataLoader batches to already be graph dictionaries.

- `evaluate_e3_cvae(...)`
  - Computes validation loss and diagnostics:
    - total loss
    - NLL
    - KL
    - normalized RMSE
    - target RMS
    - decoder mean RMS
    - sampled output RMS
    - mean parallel/perpendicular log-sigmas
    - mean parallel/perpendicular sigmas
    - latent `z_mu` RMS
    - mean `z_logvar`

### `train_dimer.py`

This is the runnable training entry point for benchmark dimer data.

Main steps:

1. Parse CLI arguments.
2. Build a `CVAEConfig`-style config with:
   - `model_type="E3DimerCVAE"`
   - `conditioning="dqpipimririm"`
   - `scaler_type="e3_vector_rms"`
3. Save `config.yaml`.
4. Load benchmark trajectories using existing CVAE dataset utilities:
   - `load_datasets`
   - `extract_vars`
5. Construct structured E3 tensors with `construct_dqpipimririm_tensors`.
6. Split train/validation by trajectory.
7. Fit `E3VectorNormalizer` on the training split only.
8. Transform train/validation structured tensors.
9. Build `DimerE3Dataset` instances.
10. Build DataLoaders using `collate_dimer_e3_graphs`.
11. Build model using `build_model_from_config_e3`.
12. Train with `train_e3_cvae`.
13. Run one validation-batch equivariance diagnostic.
14. Save:
    - `config.yaml`
    - `e3_normalizer.json`
    - `checkpoint.pt`
    - `final.pt`
    - `history.json`
    - `diagnostics.json`

Example command:

```bash
MPLCONFIGDIR=/tmp /srv/data/jakut77/miniconda3/envs/pyta/bin/python \
  -m deepRD.noiseSampler.e3cvae.train_dimer \
  --data-dir /group/ag_cmb/scratch/maojrs/stochasticClosure/dimer/boxsize5/benchmark/ \
  --output-dir deepRD/noiseSampler/training/results/e3_dimer_dqpipimririm_axial \
  --n-trajectories 200 \
  --n-total 2500 \
  --epochs 50 \
  --batch-size 2048 \
  --zdim 3 \
  --device cuda
```

### `tests.py`

This file contains smoke/equivariance utilities.

Key functions:

- `make_random_dimer_batch(batch_size=4, device="cpu", dtype=torch.float32)`
  - Builds a random graph batch with all required fields.

- `smoke_forward_loss_backward_sample(model, batch=None, beta=1.0)`
  - Runs:
    - forward pass
    - axial ELBO
    - backward pass
    - sampling call
  - Checks key tensor shapes.

- `test_decoder_equivariance(model, batch, atol=1e-5, rtol=1e-5)`
  - Rotates all vector inputs.
  - Checks:
    - decoder `mu` rotates with the same rotation
    - decoder `log_sigma` remains invariant

- `test_encoder_invariance(model, batch, atol=1e-5, rtol=1e-5)`
  - Rotates all vector inputs.
  - Checks:
    - `z_mu` is invariant
    - `z_logvar` is invariant

### `__init__.py`

Exports the main user-facing pieces:

- `E3DimerCVAE`
- `E3VectorNormalizer`
- `DimerE3Dataset`
- graph construction and rotation helpers

### `deepRD/noiseSampler/cvae/config.py`

The shared CVAE config schema was extended with:

- optional `ModelSection.hidden_irreps`
- `build_model_from_config_e3(config)`

`build_model_from_config_e3` validates:

- `model_type in {"E3DimerCVAE", "CVAE_E3NN", "E3_CVAE"}`
- `system_type == "dimer"`
- `conditioning == "dqpipimririm"`

Then it constructs:

```python
E3DimerCVAE(
    zdim=config.model.latent_dim,
    hidden_irreps=config.model.hidden_irreps or "32x0e + 16x1o + 8x2e",
)
```

## End-to-End Data Flow

Training data flow:

```text
raw trajectory files
  -> load_datasets(...)
  -> extract_vars(...) giving q, v, r
  -> construct_dqpipimririm_tensors(q, v, r)
  -> split train/val by trajectory
  -> E3VectorNormalizer.fit(train)
  -> normalizer.transform(train), normalizer.transform(val)
  -> DimerE3Dataset
  -> DataLoader(collate_fn=collate_dimer_e3_graphs)
  -> graph batch dictionary
  -> E3DimerCVAE.forward
  -> e3_cvae_axial_loss
```

Graph batch flow:

```text
q1, q2 -> minimal_image_rel(q1, q2, boxsize) -> dq
dq -> edges [dq, -dq]
dq -> radial_embedding(|dq|)
dq -> bond_unit_node
v/r history -> node vector irreps
vector norms -> node scalar irreps
target r_next -> encoder-only vector/scalar irreps
```

Model flow:

```text
encoder input: 5x1o + 5x0e
  -> two equivariant message-passing layers
  -> extract 0e scalar channels
  -> pool over two nodes
  -> scalar MLP
  -> z_mu, z_logvar

z = z_mu + eps * exp(0.5*z_logvar)
z repeated over two nodes
decoder input: 4x1o + (4+zdim)x0e
  -> two equivariant message-passing layers
  -> vector head: mu, 1x1o
  -> scalar head: log_sigma_parallel/log_sigma_perp, 2x0e
```

Loss flow:

```text
target residual per bead:
  diff = r_i_next - mu_i

decompose along bond unit e:
  diff_parallel = (diff . e) e
  diff_perp = diff - diff_parallel

node NLL:
  one 1D Gaussian along e
  one 2D isotropic Gaussian in perpendicular plane

graph NLL:
  node NLL bead 1 + node NLL bead 2

ELBO:
  mean(graph NLL) + beta * mean(KL(q(z|c,y) || N(0,I)))
```

## Symmetry Properties

The intended symmetry behavior is:

- Translational invariance:
  - positions are used only through `minimal_image(q2 - q1)`.

- Rotation equivariance:
  - polar vectors are encoded as `1o` irreps.
  - edge directions enter through spherical harmonics.
  - tensor products enforce equivariant feature mixing.
  - decoder mean is a `1o` vector and rotates with the input.

- Scalar invariance:
  - vector norms, radial basis features, latent variables, and log-sigmas are
    `0e` scalar channels.
  - encoder posterior parameters are produced only from pooled `0e` channels.

- Reflection/O(3) behavior:
  - vector quantities are treated as polar vectors (`1o`).
  - hidden irreps include `0e`, `1o`, and `2e`.

## Normalization Policy

The E3 path intentionally does not reuse the old flat `StandardScaler`.

Reason:

- Centering x/y/z separately introduces coordinate-frame dependence.
- Scaling x/y/z separately breaks equivariance.

Instead:

- velocities are divided by one scalar RMS velocity scale
- auxiliary variables and targets are divided by one scalar RMS auxiliary scale
- positions remain in physical units

This keeps the model equivariant and keeps edge/radial distances physically
meaningful.

## Current Outputs and Diagnostics

Each training run writes:

- `config.yaml`
  - Full run config.

- `e3_normalizer.json`
  - Velocity and auxiliary scalar scales.

- `checkpoint.pt`
  - Best validation checkpoint when early stopping is active.

- `final.pt`
  - Final model state, history, config, normalizer, and equivariance diagnostics.

- `history.json`
  - Train/validation curves.

- `diagnostics.json`
  - History, normalizer, and final one-batch equivariance diagnostics.

Validation diagnostics include:

- `total`
- `nll`
- `kl`
- `rmse`
- `target_rms`
- `mu_rms`
- `sample_rms`
- `log_sigma_para_mean`
- `log_sigma_perp_mean`
- `sigma_para_mean`
- `sigma_perp_mean`
- `z_mu_rms`
- `z_logvar_mean`

## Current Limitations / Not Yet Implemented

- No Langevin integrator sampling wrapper yet.
  - `sample_torch` requires a graph batch.
  - A future wrapper should accept the raw dimer state vector from
    `langevinNoiseSamplerDimerGlobal`, apply the saved E3 normalizer, build the
    graph batch, sample normalized auxiliary vectors, and inverse-transform
    auxiliary outputs.

- No physical one-step diagnostic script is committed here yet.
  - Training writes scalar diagnostics but not plots/KS comparisons against the
    benchmark auxiliary variable distributions.

- No bead-swap equivariance/permutation test yet.
  - Rotation equivariance is tested.
  - Particle exchange symmetry may need explicit testing and perhaps architectural
    constraints depending on how bead labels are physically interpreted.

- The covariance model is axial but still bead-factorized.
  - It models each bead as a 3D axial Gaussian.
  - It does not model cross-covariance between bead 1 and bead 2.

- Hidden irreps and batch size are not tuned.
  - Default hidden irreps are `"32x0e + 16x1o + 8x2e"`.
  - For GPU training, larger batches such as 1024-4096 may be appropriate if
    memory allows.

- Package metadata is not fully updated.
  - Running from the repository works.
  - If installing `deepRD` as a package, `setup.py` may need to include nested
    packages and declare dependencies such as `torch` and `e3nn`.

## Runtime Environment

The known working environment is the `pyta` Conda environment:

```text
/srv/data/jakut77/miniconda3/envs/pyta/bin/python
```

This environment has:

```text
torch 2.5.1+cu121
e3nn 0.6.0
```

CUDA training requires starting from a GPU-visible shell where:

```bash
nvidia-smi
/srv/data/jakut77/miniconda3/envs/pyta/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

reports at least one available GPU.

