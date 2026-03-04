import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.LayerNorm(h)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def reparam(mu, logvar, Tz=1.0):
    """
    Reparameterization for a diagonal Gaussian with parameters (mu, logvar).
    logvar = log(variance), so std = exp(0.5 * logvar).
    """
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(0.5 * logvar) * Tz

class DiagGaussianHead(nn.Module):
    """Outputs (mu, log_sigma) for R^3."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even: 2 * D"
        self.D = out_dim//2
        self.mlp = MLP(in_dim, out_dim, hidden=(128,128))
    def forward(self, x):
        out = self.mlp(x)
        mu, log_sigma = out[..., :self.D], out[..., self.D:]
        return mu, log_sigma

# ---------- CVAE ----------
class CVAE(nn.Module):
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri"):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=(128,128))
        self.prior   = MLP(self.cdim, out_dim=2*zdim, hidden=(128,128))
        self.decoder = DiagGaussianHead(zdim + self.cdim, 2*self.idim)
        
        # normalisers
        self.scaler_r = None
        self.scaler_c = None

    @staticmethod
    def assign_dims(system_type: str, cond_type: str) -> tuple[int, int]:
        # idim by system
        idim_map = {"bistable": 3, "dimer": 6}
        assert system_type in idim_map, f"Unknown system_type={system_type!r}"
        idim = idim_map[system_type]

        # cdim mapping by (system_type, cond_type)
        cdim_map = {
            "bistable": {
                "piri": 6,
                "piririm": 9,
                "pipimri": 9,
            },
            "dimer": {
                "pipimririm": 24,
                "local_dqipiri": 13,
                "local_abs_dqipiri": 13,
                "local_dqipiwiri": 14,
                "local_dqipipimririm": 25,
                "local_abs_dqipipimririm": 25
            },
        }

        assert system_type in cdim_map, f"Missing cdim map for system_type={system_type!r}"
        assert cond_type in cdim_map[system_type], (
            f"Unsupported cond_type={cond_type!r} for system_type={system_type!r}. "
            f"Supported: {tuple(cdim_map[system_type].keys())}"
        )

        cdim = cdim_map[system_type][cond_type]
        return idim, cdim
        
    def attach_normalizers(self, scaler_r, scaler_c):
        """Attach normalization scalers for automatic preprocessing."""
        self.scaler_r = scaler_r
        self.scaler_c = scaler_c

    def set_temps(self, Tr=None, Tz=None):
        """Set global sampling temperatures. Call with no args to unset."""
        if Tr is None and Tz is None:
            # remove attributes if they exist
            for name in ("Tr", "Tz"):
                if hasattr(self, name):
                    delattr(self, name)
        else:
            if Tr is not None:
                self.Tr = Tr
            if Tz is not None:
                self.Tz = Tz

    def encode(self, r_next, c):
        q = self.encoder(torch.cat([r_next, c], dim=-1))
        q_mu, q_logvar = q.split(self.zdim, dim=-1)
        return q_mu, q_logvar

    def prior_params(self, c):
        p = self.prior(c)
        p_mu, p_logvar = p.split(self.zdim, dim=-1)
        return p_mu, p_logvar

    def decode(self, z, c):
        mu, log_sigma = self.decoder(torch.cat([z, c], dim=-1))
        return mu, log_sigma

    def forward(self, r_next, c):
        p_mu, p_logv = self.prior_params(c)
        q_mu, q_logv = self.encode(r_next, c)
        z = reparam(q_mu, q_logv)
        dec_out = self.decode(z, c)
        return dec_out, (q_mu, q_logv), (p_mu, p_logv)   
    
    # ----- sampling ----- #
    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        p_mu, p_logv = self.prior_params(c)
        z = reparam(p_mu, p_logv, Tz=Tz)  # sample from p(z|c)
        
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return r

    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        """
        Public sampling entrypoint.
        If self.cond_type is local mode, c_n_np is a STATE vector:
            [q1,q2,v1,v2,r1,r2] (18,)
        Else, c_n_np is the model conditioning vector.
        """
        if self.scaler_c is None or self.scaler_r is None:
            raise ValueError("Call attach_normalizers(...) before sample().")

        if device is None:
            device = next(self.parameters()).device

        if hasattr(self, "Tr"): Tr = self.Tr
        if hasattr(self, "Tz"): Tz = self.Tz

        # route based on configured conditioning type
        if getattr(self, "cond_type", "").startswith("local_"):
            return self._sample_local_from_state(c_n_np, Tr=Tr, Tz=Tz, device=device)
        else:
            return self._sample_from_conditioning(c_n_np, Tr=Tr, Tz=Tz, device=device)


    def _sample_from_conditioning(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        """
        Your existing implementation, unchanged except factored out.
        Input is model conditioning vector in physical units.
        Output is model output in physical units.
        """
        # ---- Reshape and normalise ---- #
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False
        if c_n_np.ndim == 1:
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True

        c_norm = self.scaler_c.transform(c_n_np).astype(np.float32)
        c_t = torch.from_numpy(c_norm).to(device=device)

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=Tz)
        
        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        return r_next_phys.squeeze() if single_sample else r_next_phys


    def _sample_local_from_state(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        """
        state_np: (..., k)
        returns: (..., 6) xyz aux next as [r1_next(3), r2_next(3)]
        """

        rel = True
        if self.cond_type in ("local_abs_dqipiri", "local_abs_dqipipimririm"):
            rel=False

        # ---- Reshape and normalise ---- #
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False
        if c_n_np.ndim == 1:
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True

        if self.cond_type in ("local_dqipiri", "local_dqipiwiri", "local_abs_dqipiri"):
            # unpack
            q1 = c_n_np[:, 0:3]
            q2 = c_n_np[:, 3:6]
            v1 = c_n_np[:, 6:9]
            v2 = c_n_np[:, 9:12]
            r1 = c_n_np[:, 12:15]
            r2 = c_n_np[:, 15:18]

        elif self.cond_type in ("local_dqipipimririm", "local_abs_dqipipimririm"):
            # unpack
            q1      = c_n_np[:, 0:3]
            q2      = c_n_np[:, 3:6]
            q1_prev = c_n_np[:, 6:9]
            q2_prev = c_n_np[:, 9:12]
            v1      = c_n_np[:, 12:15]
            v2      = c_n_np[:, 15:18]
            v1_prev = c_n_np[:, 18:21]
            v2_prev = c_n_np[:, 21:24]
            r1      = c_n_np[:, 24:27]
            r2      = c_n_np[:, 27:30]
            r1_prev = c_n_np[:, 30:33]
            r2_prev = c_n_np[:, 33:36]
        else:
            raise ValueError("Invalid conditioning.")

        # torchify for geometry (batch)
        q1_t = torch.from_numpy(q1).to(device=device)
        q2_t = torch.from_numpy(q2).to(device=device)
        v1_t = torch.from_numpy(v1).to(device=device)
        v2_t = torch.from_numpy(v2).to(device=device)
        r1_t = torch.from_numpy(r1).to(device=device)
        r2_t = torch.from_numpy(r2).to(device=device)

        # build local frame
        # (store boxsize in self.parameters or self.boxsize)
        boxsize = 5.0
        R, bond_len = build_local_frame(q1_t, q2_t, boxsize=boxsize)  # R:(B,3,3), bond_len:(B,)

        # Converting to local frame
        v_rel_loc, v_com_loc = dimer_to_local(R, v1_t, v2_t, rel=rel)
        r_rel_loc, r_com_loc = dimer_to_local(R, r1_t, r2_t, rel=rel)
        
        if self.cond_type in ("local_dqipipimririm", "local_abs_dqipipimririm"):
            q1_prev_t = torch.from_numpy(q1_prev).to(device=device)
            q2_prev_t = torch.from_numpy(q2_prev).to(device=device)
            v1_prev_t = torch.from_numpy(v1_prev).to(device=device)
            v2_prev_t = torch.from_numpy(v2_prev).to(device=device)
            r1_prev_t = torch.from_numpy(r1_prev).to(device=device)
            r2_prev_t = torch.from_numpy(r2_prev).to(device=device)

            # build local frame
            R_prev, bond_len_prev = build_local_frame(q1_prev_t, q2_prev_t, boxsize=boxsize)  # R:(B,3,3), bond_len:(B,)
            # Converting to local frame          
            v_rel_prev_loc, v_com_prev_loc = dimer_to_local(R_prev, v1_prev_t, v2_prev_t, rel=rel)
            r_rel_prev_loc, r_com_prev_loc = dimer_to_local(R_prev, r1_prev_t, r2_prev_t, rel=rel)
    

        if self.cond_type in ("local_dqipiri", "local_abs_dqipiri"):
            # build model conditioning: [bond_len, v_rel_loc, r_rel_loc, r_com_loc] -> (B,13)
            c_model_t = torch.cat([bond_len.unsqueeze(-1), v_rel_loc, v_com_loc, r_rel_loc, r_com_loc], dim=-1)
        elif self.cond_type=="local_dqipiwiri":
            # Adding perpendicular velocity
            v_perp2 = (v_rel_loc[..., 1] ** 2 + v_rel_loc[..., 2] ** 2).unsqueeze(-1)
            # build model conditioning: [bond_len, v_rel_loc, r_rel_loc, r_com_loc] -> (B,13)
            c_model_t = torch.cat([bond_len.unsqueeze(-1), v_rel_loc, v_com_loc, v_perp2, r_rel_loc, r_com_loc], dim=-1)
        elif self.cond_type in ("local_dqipipimririm", "local_abs_dqipipimririm"):
            # build model conditioning: [bond_len, v_loc, v_prev_loc, r_loc, r_prev_loc] -> (B,25)
            c_model_t = torch.cat([bond_len.unsqueeze(-1), v_rel_loc, v_com_loc, v_rel_prev_loc, v_com_prev_loc, 
                                        r_rel_loc, r_com_loc, r_rel_prev_loc, r_com_prev_loc], dim=-1)

        c_model_np = c_model_t.detach().cpu().numpy().astype(np.float32)

        # sample in local frame (returns (B,6) = [r_rel_loc_next, r_com_loc_next])
        out_loc = self._sample_from_conditioning(c_model_np, Tr=Tr, Tz=Tz, device=device)  # np

        out_loc_t = torch.from_numpy(out_loc).to(device=device)
        r_rel_loc_next = out_loc_t[:, 0:3]
        r_com_loc_next = out_loc_t[:, 3:6]

        # Back to xyz frame
        r1_next, r2_next = dimer_to_xyz(R, r_rel_loc_next, r_com_loc_next, rel=rel)

        out_xyz = torch.cat([r1_next, r2_next], dim=-1).detach().cpu().numpy()
        return out_xyz.squeeze(0) if single_sample else out_xyz

### Helper functions for LOCAL transformation
def minimal_image_rel(q1, q2, boxsize=None, boundary_type='periodic'):
    """
    q1, q2: [..., 3] torch tensors
    returns q2 - q1 with minimal-image convention matching trajectoryTools.relativePosition
    """
    rel = q2 - q1  # [..., 3]

    if boundary_type == "periodic" and boxsize is not None:
        # box: tensor of shape [3]
        if isinstance(boxsize, (list, tuple, np.ndarray)):
            box = torch.tensor(boxsize, dtype=rel.dtype, device=rel.device)
        else:  # scalar -> same in all dims
            box = torch.full((3,), float(boxsize), dtype=rel.dtype, device=rel.device)

        # broadcast box over leading dims, minimal image per component
        rel = rel - box * torch.round(rel / box)

    return rel

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
    Convert vectors from lab xyz to local frame.

    R: (..., 3, 3) with columns [e1,e2,e3] in xyz
    v_xyz: (..., 3)
    returns v_local: (..., 3)
    """
    # v_local = R^T v_xyz
    return (R.transpose(-2, -1) @ v_xyz.unsqueeze(-1)).squeeze(-1)


def to_xyz(R: torch.Tensor, v_local: torch.Tensor) -> torch.Tensor:
    """
    Convert vectors from local frame to lab xyz.

    R: (..., 3, 3)
    v_local: (..., 3)
    returns v_xyz: (..., 3)
    """
    # v_xyz = R v_local
    return (R @ v_local.unsqueeze(-1)).squeeze(-1)

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

def dimer_to_local(R: torch.Tensor, a1_xyz: torch.Tensor, a2_xyz: torch.Tensor, rel=True):
    
    if rel==True:
        a1_xyz, a2_xyz = dimer_rel_com(a1_xyz, a2_xyz)
        
    a1_loc = to_local(R, a1_xyz)
    a2_loc = to_local(R, a2_xyz)

    return a1_loc, a2_loc

def dimer_to_xyz(R: torch.Tensor, a1_loc: torch.Tensor, a2_loc: torch.Tensor, rel=True):

    a1_xyz = to_xyz(R, a1_loc)
    a2_xyz = to_xyz(R, a2_loc)
    
    if rel==True:
        a1_xyz, a2_xyz = dimer_from_rel_com(a1_xyz, a2_xyz)
        
    return a1_xyz, a2_xyz