import torch
import numpy as np

"""
Transforms for CVAE local frame construction and relative velocity computation.
"""

def compute_norms_separate(x, dim=-1):
    """
    Returns norms of separate component
    x: [..., 6]
    """
    x1_norm = torch.norm(x[..., :3], dim=-1)
    x2_norm = torch.norm(x[..., 3:6], dim=-1)

    return x1_norm, x2_norm

def dimer_rel_com(a1: torch.Tensor, a2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given per-particle vectors a1,a2 (...,3), return (rel, com).
    rel = a2 - a1
    com = 0.5*(a1 + a2)
    """
    rel = a2 - a1
    com = 0.5 * (a1 + a2)
    return rel, com

def dimer_from_rel_com(rel: torch.Tensor, com: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse mapping:
    a1 = com - 0.5*rel
    a2 = com + 0.5*rel
    """
    a1 = com - 0.5 * rel
    a2 = com + 0.5 * rel
    return a1, a2

def minimal_image_rel(q1, q2, boxsize=None, boundary_type='periodic'):
    """
    q1, q2: [..., 3] torch tensors
    returns q2 - q1 with minimal-image convention matching trajectoryTools.relativePosition
    """
    
    # ORIGINAL
    rel = q2 - q1  # [..., 3]
    
    # SWAPPED!!!! 
    #rel = q1 - q2  # [..., 3]

    if boundary_type == "periodic" and boxsize is not None:
        # box: tensor of shape [3]
        if torch.is_tensor(boxsize):
            box = boxsize.to(dtype=rel.dtype, device=rel.device)
        elif isinstance(boxsize, (list, tuple, np.ndarray)):
            box = torch.as_tensor(boxsize, dtype=rel.dtype, device=rel.device)
        else:  # scalar -> same in all dims
            box = torch.as_tensor(boxsize, dtype=rel.dtype, device=rel.device)

        if box.ndim == 0:
            box = box.expand(3)
        elif box.numel() == 3:
            box = box.reshape(3)
        else:
            raise ValueError(
                f"boxsize must be a scalar or 3-vector, got shape {tuple(box.shape)}"
            )

        # broadcast box over leading dims, minimal image per component
        rel = rel - box * torch.round(rel / box)

    return rel

def compute_axisRelVel(q1, q2, v1, v2, boxsize, boundary_type="periodic"):
    """
    Computes axis-relative velocity for a dimer in a fully vectorised way.

    Inputs:
      q1, q2: positions  [..., 3]
      v1, v2: velocities [..., 3]
      boxsize: scalar or 3-vector
      boundary_type: "periodic" or "none"

    Returns:
      axisRelVel: tensor [...]
    """

    # Minimal-image relative position
    rel_pos = minimal_image_rel(q1, q2, boxsize, boundary_type)   # [..., 3]

    # Relative velocity
    rel_vel = v2 - v1                                             # [..., 3]

    # Unit vector along dimer axis
    norm_rel = torch.norm(rel_pos, dim=-1, keepdim=True)          # [..., 1]
    unit_rel = rel_pos / (norm_rel + 1e-12)                       # avoid div-by-zero

    # Projection of relative velocity onto dimer axis
    axis_rel_vel = torch.sum(rel_vel * unit_rel, dim=-1, keepdim=False)  # [...]

    return axis_rel_vel

def compute_dx_dvx(q1, q2, v1, v2, boxsize, boundary_type="periodic", keepdim=False):
    """
    Computes bond length dx and axis-relative velocity dvx for a dimer in a
    fully vectorised way.

    Inputs:
      q1, q2: positions  [..., 3]
      v1, v2: velocities [..., 3]
      boxsize: scalar or 3-vector
      boundary_type: "periodic" or "none"
      keepdim: if True, return dx and dvx as [..., 1]; otherwise [...]

    Returns:
      dx, dvx: tensors with shape [...] or [..., 1] if keepdim=True
    """

    rel = minimal_image_rel(q1, q2, boxsize=boxsize, boundary_type=boundary_type)
    dx = torch.linalg.norm(rel, dim=-1, keepdim=keepdim)
    denom = dx if keepdim else dx.unsqueeze(-1)
    e = rel / denom.clamp_min(1e-12)
    dvx = torch.sum((v2 - v1) * e, dim=-1, keepdim=keepdim)
    
    return (dx, dvx)


def build_local_frame(
    q1: torch.Tensor,
    q2: torch.Tensor,
    boxsize: float = 5.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build bond-aligned local orthonormal frame for a dimer.

    Args
    ----
    q1, q2 : (..., 3)
        Particle positions in lab xyz.
    boxsize : float or (3,)
        Periodic box size(s) used for minimal image convention.
    eps : float
        Numerical epsilon.

    Returns
    -------
    R : (..., 3, 3)
        Rotation matrix whose columns are [e1, e2, e3] in lab coords.
        For any lab vector v_xyz: v_local = R^T @ v_xyz,  v_xyz = R @ v_local.
    r : (...,)
        Bond length ||d|| with minimal image convention.
    """
    # relative vector with minimal image
    d = minimal_image_rel(q1, q2, boxsize)                 # (..., 3)
    r = torch.linalg.norm(d, dim=-1).clamp_min(eps)     # (...,)
    e1 = d / r.unsqueeze(-1)                            # (..., 3)

    # Choose a reference axis not too aligned with e1 to build e2 stably
    # If |e1_x| < 0.9 => use x-axis else y-axis
    ex = torch.zeros_like(e1)
    ex[..., 0] = 1.0
    ey = torch.zeros_like(e1)
    ey[..., 1] = 1.0
    use_ex = (e1[..., 0].abs() < 0.9).unsqueeze(-1)     # (..., 1)
    a = torch.where(use_ex, ex, ey)                      # (..., 3)

    # Gram–Schmidt to make e2 orthogonal to e1
    a_proj = (a * e1).sum(dim=-1, keepdim=True) * e1
    u2 = a - a_proj
    u2_norm = torch.linalg.norm(u2, dim=-1, keepdim=True).clamp_min(eps)
    e2 = u2 / u2_norm                                    # (..., 3)

    # Right-handed e3
    e3 = torch.cross(e1, e2, dim=-1)                     # (..., 3)
    e3_norm = torch.linalg.norm(e3, dim=-1, keepdim=True).clamp_min(eps)
    e3 = e3 / e3_norm

    # Rotation matrix with columns [e1, e2, e3]
    R = torch.stack([e1, e2, e3], dim=-1)                # (..., 3, 3)
    return R, r

def to_local(R: torch.Tensor, v_xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert one or more concatenated 3D lab-frame vectors to local frame.

    Args
    ----
    R : (..., 3, 3)
        Rotation matrix with columns [e1, e2, e3] in lab coords.
    v_xyz : (..., 3*K)
        One or more concatenated 3D vectors in lab xyz coordinates.

    Returns
    -------
    v_local : (..., 3*K)
        Same shape as v_xyz, with each 3D block transformed as R^T @ v.
    """
    if v_xyz.shape[-1] % 3 != 0:
        raise ValueError(
            f"Last dimension of v_xyz must be a multiple of 3, got {v_xyz.shape[-1]}"
        )

    original_shape = v_xyz.shape
    k = original_shape[-1] // 3

    # (..., 3*K) -> (..., K, 3)
    v_blocks = v_xyz.reshape(*original_shape[:-1], k, 3)

    # Need R to broadcast over the K vector blocks.
    # R:        (..., 3, 3)
    # R_T:      (..., 3, 3)
    # R_T_exp:  (..., 1, 3, 3)
    # v_exp:    (..., K, 3, 1)
    v_local = (
        R.transpose(-2, -1).unsqueeze(-3)
        @ v_blocks.unsqueeze(-1)
    ).squeeze(-1)

    # (..., K, 3) -> (..., 3*K)
    return v_local.reshape(*original_shape)

def to_xyz(R: torch.Tensor, v_local: torch.Tensor) -> torch.Tensor:
    """
    Convert one or more concatenated 3D local-frame vectors to lab frame.

    Args
    ----
    R : (..., 3, 3)
        Rotation matrix with columns [e1, e2, e3] in lab coords.
    v_local : (..., 3*K)
        One or more concatenated 3D vectors in local coordinates.

    Returns
    -------
    v_xyz : (..., 3*K)
        Same shape as v_local, with each 3D block transformed as R @ v.
    """
    if v_local.shape[-1] % 3 != 0:
        raise ValueError(
            f"Last dimension of v_local must be a multiple of 3, got {v_local.shape[-1]}"
        )

    original_shape = v_local.shape
    k = original_shape[-1] // 3

    # (..., 3*K) -> (..., K, 3)
    v_blocks = v_local.reshape(*original_shape[:-1], k, 3)

    # R:       (..., 3, 3)
    # R_exp:   (..., 1, 3, 3)
    # v_exp:   (..., K, 3, 1)
    v_xyz = (
        R.unsqueeze(-3)
        @ v_blocks.unsqueeze(-1)
    ).squeeze(-1)

    # (..., K, 3) -> (..., 3*K)
    return v_xyz.reshape(*original_shape)

def unpack_state_vector(c_n_np, cond_type, device=None):
    if cond_type not in ("E3_base", "E3_dqipipimririm"):
        raise ValueError(f"Unsupported E3 state vector conditioning: {cond_type}")

    c_n_np = np.asarray(c_n_np, dtype=np.float32)
    if c_n_np.ndim == 1:
        c_n_np = c_n_np[None]
    if c_n_np.shape[-1] < 30:
        raise ValueError(
            f"E3 sampling expects a physical state vector with at least 30 values, got {c_n_np.shape[-1]}"
        )

    state = torch.as_tensor(c_n_np, dtype=torch.float32, device=device)

    q1 = state[:, 0:3]
    q2 = state[:, 3:6]
    v1 = state[:, 6:9]
    v2 = state[:, 9:12]
    v1p = state[:, 12:15]
    v2p = state[:, 15:18]
    r1 = state[:, 18:21]
    r2 = state[:, 21:24]
    r1p = state[:, 24:27]
    r2p = state[:, 27:30]

    return q1, q2, v1, v2, v1p, v2p, r1, r2, r1p, r2p
