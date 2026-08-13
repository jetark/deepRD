import torch
from e3nn import o3
from torch.utils.data import Dataset
from deepRD.noiseSampler.cvae.transforms import minimal_image_rel

def radial_embedding(edge_vec, num_basis=16, r_cut=2.0):
    "Embedding for the edge vector."
    r = edge_vec.norm(dim=-1, keepdim=True)
    centers = torch.linspace(0.0, r_cut, num_basis, device=edge_vec.device)
    widths = (r_cut / num_basis)
    return torch.exp(-((r - centers) ** 2) / (widths ** 2))


def split_dimer_particles(x: torch.Tensor):
    """
    Split interleaved dimer trajectories into bead-1 and bead-2 tensors.

    x:       [n_traj, 2*T, 3], ordered bead1, bead2, bead1, bead2, ...
    returns: x1, x2 each [n_traj, T, 3]
    """
    return x[:, 0::2, :], x[:, 1::2, :]


def construct_dqpipimririm_tensors(q: torch.Tensor, v: torch.Tensor, r: torch.Tensor, lag2: bool = False):
    """
    Build structured global-frame dqpipimririm tensors from dimer trajectories.

    Without lag2: effective time index n=1..T-2, returns T-2 samples per traj.
    With lag2: effective time index n=2..T-2, returns T-3 samples per traj,
    adding v1_nm2/v2_nm2 and r1_nm2/r2_nm2 fields.

    q, v, r: interleaved dimer tensors [n_traj, 2*T, 3]
    """
    if q.shape != v.shape or q.shape != r.shape:
        raise ValueError(f"q, v, r must have matching shapes; got {q.shape}, {v.shape}, {r.shape}")
    if q.ndim != 3 or q.shape[-1] != 3:
        raise ValueError(f"Expected q, v, r as [n_traj, 2*T, 3], got {q.shape}")
    if q.shape[1] % 2 != 0:
        raise ValueError("Dimer trajectory axis must be even because beads are interleaved.")

    q1, q2 = split_dimer_particles(q)
    v1, v2 = split_dimer_particles(v)
    r1, r2 = split_dimer_particles(r)

    if lag2:
        # n=2..T-2; need n-2 so earliest index is 0
        return {
            "q1": q1[:, 2:-1, :],
            "q2": q2[:, 2:-1, :],
            "v1_n": v1[:, 2:-1, :],
            "v2_n": v2[:, 2:-1, :],
            "v1_nm1": v1[:, 1:-2, :],
            "v2_nm1": v2[:, 1:-2, :],
            "v1_nm2": v1[:, :-3, :],
            "v2_nm2": v2[:, :-3, :],
            "r1_n": r1[:, 2:-1, :],
            "r2_n": r2[:, 2:-1, :],
            "r1_nm1": r1[:, 1:-2, :],
            "r2_nm1": r2[:, 1:-2, :],
            "r1_nm2": r1[:, :-3, :],
            "r2_nm2": r2[:, :-3, :],
            "r1_next": r1[:, 3:, :],
            "r2_next": r2[:, 3:, :],
        }
    return {
        "q1": q1[:, 1:-1, :],
        "q2": q2[:, 1:-1, :],
        "v1_n": v1[:, 1:-1, :],
        "v2_n": v2[:, 1:-1, :],
        "v1_nm1": v1[:, :-2, :],
        "v2_nm1": v2[:, :-2, :],
        "r1_n": r1[:, 1:-1, :],
        "r2_n": r2[:, 1:-1, :],
        "r1_nm1": r1[:, :-2, :],
        "r2_nm1": r2[:, :-2, :],
        "r1_next": r1[:, 2:, :],
        "r2_next": r2[:, 2:, :],
    }


def flatten_structured_dqpipimririm(data: dict):
    """
    Flatten structured dqpipimririm tensors from [n_traj, T_eff, 3] to [N, 3].
    """
    return {key: value.reshape(-1, value.shape[-1]) for key, value in data.items()}


_BASE_KEYS = (
    "q1", "q2",
    "v1_n", "v2_n", "v1_nm1", "v2_nm1",
    "r1_n", "r2_n", "r1_nm1", "r2_nm1",
    "r1_next", "r2_next",
)
_LAG2_EXTRA_KEYS = ("v1_nm2", "v2_nm2", "r1_nm2", "r2_nm2")


class DimerE3Dataset(Dataset):
    """
    Dataset for global-frame dqpipimririm E3 graph training.

    Items are raw vector fields for one dimer sample. Use
    collate_dimer_e3_graphs as the DataLoader collate_fn to build graph batches.
    Pass lag2=True to include the two-step history fields v*_nm2 and r*_nm2.
    """

    def __init__(self, structured: dict, flatten=True, lag2: bool = False):
        if flatten:
            structured = flatten_structured_dqpipimririm(structured)
        self.lag2 = lag2
        self.keys = _BASE_KEYS + (_LAG2_EXTRA_KEYS if lag2 else ())
        missing = [key for key in self.keys if key not in structured]
        if missing:
            raise ValueError(f"Missing structured dqpipimririm fields: {missing}")

        n = structured[self.keys[0]].shape[0]
        for key in self.keys:
            if structured[key].shape[0] != n or structured[key].shape[-1] != 3:
                raise ValueError(f"Bad shape for {key}: {structured[key].shape}")
        self.data = structured

    def __len__(self):
        return self.data[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.keys}


def collate_dimer_e3_graphs(samples, boxsize=5.0):
    """
    Collate DimerE3Dataset items into one E3 graph batch.
    Automatically detects lag2 mode from the presence of v1_nm2 in the samples.
    """
    all_keys = samples[0].keys()
    batch = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in all_keys
    }
    lag2 = "v1_nm2" in batch
    return build_dimer_graph_batch(
        q1=batch["q1"],
        q2=batch["q2"],
        v1=batch["v1_n"],
        v2=batch["v2_n"],
        r1=batch["r1_n"],
        r2=batch["r2_n"],
        v1_prev=batch["v1_nm1"],
        v2_prev=batch["v2_nm1"],
        r1_prev=batch["r1_nm1"],
        r2_prev=batch["r2_nm1"],
        r1_next=batch["r1_next"],
        r2_next=batch["r2_next"],
        boxsize=boxsize,
        v1_prev2=batch.get("v1_nm2"),
        v2_prev2=batch.get("v2_nm2"),
        r1_prev2=batch.get("r1_nm2"),
        r2_prev2=batch.get("r2_nm2"),
    )

def pack_e3_features(scalars, vectors):
    """
    scalars: [N, n_scalar]
    vectors: [N, n_vector, 3]
    matching irreps: "{n_vector}x1o + {n_scalar}x0e"
    """
    return torch.cat([
        vectors.reshape(vectors.shape[0], -1),
        scalars,
    ], dim=-1)

def build_dimer_graph_batch(
    q1,
    q2,
    v1,
    v2,
    r1,
    r2,
    v1_prev,
    v2_prev,
    r1_prev,
    r2_prev,
    r1_next=None,
    r2_next=None,
    boxsize=5.0,
    v1_prev2=None,
    v2_prev2=None,
    r1_prev2=None,
    r2_prev2=None,
    add_dx_scalar=False,
    radial_num_basis=16,
    r_cut=2.0,
):
    """
    Shapes:
        q1, q2, v1, ...: [B, 3]
    Returns graph tensors for B two-node graphs.
    Pass v*_prev2 and r*_prev2 to enable lag-2 conditioning.

    Featurisation options (equivariance-preserving; default = legacy behaviour):
      add_dx_scalar    : append the bond length dx=|q1-q2| as a direct invariant
                         scalar node feature (both nodes share it). Gives the
                         decoder sharp, direct access to dx (cf. DAG2's dx input).
      radial_num_basis : number of Gaussian radial-embedding bases for dx (was 16).
      r_cut            : radial-embedding cutoff (was 2.0).
    """

    B = q1.shape[0]
    device = q1.device
    lag2 = v1_prev2 is not None

    x12 = minimal_image_rel(q1, q2, boxsize=boxsize)
    dx = x12.norm(dim=-1, keepdim=True)                       # [B,1]

    # Base: 4 per-node vectors (v_n, v_nm1, r_n, r_nm1); lag2 adds v_nm2 and r_nm2
    dec_vecs = [
        torch.stack([v1, v2], dim=1),
        torch.stack([v1_prev, v2_prev], dim=1),
        torch.stack([r1, r2], dim=1),
        torch.stack([r1_prev, r2_prev], dim=1),
    ]
    if lag2:
        dec_vecs += [
            torch.stack([v1_prev2, v2_prev2], dim=1),
            torch.stack([r1_prev2, r2_prev2], dim=1),
        ]

    # Node-wise vectors: [B, 2, num_vecs, 3] → flatten → [B*2, num_vecs, 3]
    vec_dec = torch.stack(dec_vecs, dim=2).reshape(B * 2, len(dec_vecs), 3)

    dx_node = dx.repeat_interleave(2, dim=0) if add_dx_scalar else None  # [2B,1]

    scal_dec = vec_dec.norm(dim=-1)
    if add_dx_scalar:
        scal_dec = torch.cat([scal_dec, dx_node], dim=-1)
    h_dec_base = pack_e3_features(scal_dec, vec_dec)

    if r1_next is not None:
        target = torch.stack([r1_next, r2_next], dim=1).reshape(B * 2, 3)
        vec_enc = torch.cat([vec_dec, target[:, None, :]], dim=1)
        scal_enc = vec_enc.norm(dim=-1)
        if add_dx_scalar:
            scal_enc = torch.cat([scal_enc, dx_node], dim=-1)
        h_enc = pack_e3_features(scal_enc, vec_enc)
    else:
        target = None
        h_enc = None

    # edges: for every graph, 0->1 and 1->0
    node_offset = 2 * torch.arange(B, device=device)

    src = torch.cat([node_offset + 0, node_offset + 1], dim=0)
    dst = torch.cat([node_offset + 1, node_offset + 0], dim=0)
    edge_index = torch.stack([src, dst], dim=0)

    edge_vec = torch.cat([x12, -x12], dim=0)

    edge_radial = radial_embedding(edge_vec, num_basis=radial_num_basis, r_cut=r_cut)
    bond_unit = x12 / x12.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bond_unit_node = bond_unit.repeat_interleave(2, dim=0)

    batch_index = torch.arange(B, device=device).repeat_interleave(2)

    return {
        "h_dec_base": h_dec_base,
        "h_enc": h_enc,
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "edge_radial": edge_radial,
        "bond_unit_node": bond_unit_node,
        "batch_index": batch_index,
        "r_next": target,
        "num_graphs": B,
    }


def append_z_to_decoder_features(h_dec_base: torch.Tensor, z_node: torch.Tensor) -> torch.Tensor:
    """
    Append invariant latent z to decoder node features.

    Assumes h_dec_base is packed as:
        [vector irreps..., scalar 0e features...]

    Since z is invariant, it is appended only to scalar channels.

    Args:
        h_dec_base: [B*2, base_dim]
        z_node:     [B*2, zdim]

    Returns:
        h_dec:      [B*2, base_dim + zdim]
    """
    if h_dec_base.ndim != 2:
        raise ValueError(f"h_dec_base must be 2D, got {h_dec_base.shape}")

    if z_node.ndim != 2:
        raise ValueError(f"z_node must be 2D, got {z_node.shape}")

    if h_dec_base.shape[0] != z_node.shape[0]:
        raise ValueError(
            f"Node count mismatch: h_dec_base has {h_dec_base.shape[0]}, "
            f"z_node has {z_node.shape[0]}"
        )

    return torch.cat([h_dec_base, z_node], dim=-1)


def rotate_e3_features(x: torch.Tensor, irreps, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate packed e3nn features according to their irreps.

    x:      [N, irreps.dim]
    R:      [3, 3]
    output: [N, irreps.dim]
    """
    irreps = o3.Irreps(irreps)
    D = irreps.D_from_matrix(R.cpu()).to(device=x.device, dtype=x.dtype)
    return x @ D.T


def rotate_vectors(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate ordinary 3D vectors stored as rows.

    x: [N, 3]
    R: [3, 3]
    """
    R = R.to(device=x.device, dtype=x.dtype)
    return x @ R.T


def rotate_batch_vectors(
    batch: dict,
    R: torch.Tensor,
    irreps_dec_base="4x1o + 4x0e",
    irreps_enc="5x1o + 5x0e",
) -> dict:
    """
    Rotate all vector-valued parts of an E3 dimer batch.

    Scalars/radial features are left unchanged.
    edge_index and batch_index are left unchanged.
    """
    out = dict(batch)

    if "h_dec_base" in batch and batch["h_dec_base"] is not None:
        out["h_dec_base"] = rotate_e3_features(
            batch["h_dec_base"],
            irreps_dec_base,
            R,
        )

    if "h_enc" in batch and batch["h_enc"] is not None:
        out["h_enc"] = rotate_e3_features(
            batch["h_enc"],
            irreps_enc,
            R,
        )

    if "edge_vec" in batch and batch["edge_vec"] is not None:
        out["edge_vec"] = rotate_vectors(batch["edge_vec"], R)

    if "r_next" in batch and batch["r_next"] is not None:
        out["r_next"] = rotate_vectors(batch["r_next"], R)

    if "bond_unit_node" in batch and batch["bond_unit_node"] is not None:
        out["bond_unit_node"] = rotate_vectors(batch["bond_unit_node"], R)

    # edge_radial is invariant, so do not rotate it
    return out
