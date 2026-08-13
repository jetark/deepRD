import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepRD.tools.trajectoryTools import minimal_image_rel, build_local_frame, to_local, to_xyz, dimer_rel_com, dimer_from_rel_com
import deepRD.noiseSampler.cvae.models as models
import deepRD.noiseSampler.cvae.transforms as transforms
from deepRD.tools.modelsTools import MLP as MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- CVAE ----------
class CVAESampler(models.CVAE):
    """
    CVAE wrapper for compatibility with the Langevin Integrator sampling.
    """
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128), dim_maps=None):
        super().__init__(
            zdim=zdim,
            system_type=system_type, 
            cond_type=cond_type, 
            hidden=hidden,
            dim_maps=dim_maps)

    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None, return_stats=False):
        """
        Sample r_{n+1} in physical units given c_n_np in physical units as NumPy array. 
        Built in normalisation of input and denormalisation of output.

        Args:
            c_n_np (np.ndarray): shape (..., cdim)
            Tr (float): temperature scaling factor for stochasticity
            Tz (float): temperature scaling factor in latent space for stochasticity
            device (torch.device): GPU/CPU device to use (optional)
            
        Handles every conditioning type provided the corresponding scaler.

        Returns:
            np.ndarray: generated r_{n+1} in same physical scale as input
        """
        if self.scaler_c is None or self.scaler_r is None:
            raise ValueError("Call attach_normalizers(...) before sample().")

        if device is None:
            device = next(self.parameters()).device

        # overwrite with global temps if defined
        if hasattr(self, "Tr"):
            Tr = self.Tr
        if hasattr(self, "Tz"):
            Tz = self.Tz

        # ---- Reshape and normalise ---- #
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False

        if c_n_np.ndim == 1:
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True
        
        c_norm = self.scaler_c.transform(c_n_np).astype(np.float32)

        # --- Convert to torch tensor ---
        c_t = torch.from_numpy(c_norm).to(device=device)

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=Tz, return_stats=return_stats)

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys

# ---------- CVAE with Local Frame transformation ----------
class CVAE_LF(CVAESampler):
    
    @staticmethod
    def assign_dims(system_type: str, cond_type: str, idim_map=None, cdim_map=None) -> tuple[int, int]:
        # idim by system
        idim_map = {"dimer": 6}
        assert system_type in idim_map, f"Unknown system_type={system_type!r}"
        idim = idim_map[system_type]

        # cdim mapping by (system_type, cond_type)
        cdim_map = {
            "dimer": {
                "local_pipimririm": 24,
                "local_dqipipimririm": 25,
                "local_dqipiririm": 19,   # no v^{n-1} (E15)
                "local_dqidpipipimririm": 26,
                "local_dqidpipimmrimm": 38,
                "local_dqipiri": 13,
                "local_dqiririm": 13,   # velocity-free (E5): dx + r^n + r^{n-1}
                "local_piri": 12,
                "local_dqipi": 7,
            },
        }

        assert system_type in cdim_map, f"Missing cdim map for system_type={system_type!r}"
        assert cond_type in cdim_map[system_type], (
            f"Unsupported cond_type={cond_type!r} for system_type={system_type!r}. "
            f"Supported: {tuple(cdim_map[system_type].keys())}"
        )

        cdim = cdim_map[system_type][cond_type]
        return idim, cdim

    def prior_params(self, c=None, batch_shape=None, device=None, dtype=None):
        """
        Standard Gaussian prior:

            p(z) = N(0, I)

        Returns p_mu = 0, p_logvar = 0.

        c is accepted only for API compatibility with the learned-prior version.
        """
        if c is not None:
            shape = (*c.shape[:-1], self.zdim)
            device = c.device
            dtype = c.dtype
        else:
            assert batch_shape is not None, "Need either c or batch_shape."
            shape = (*batch_shape, self.zdim)

        p_mu = torch.zeros(shape, device=device, dtype=dtype)
        p_logvar = torch.zeros(shape, device=device, dtype=dtype)
        return p_mu, p_logvar

    # ----- sampling ----- #
    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0, return_stats=False):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        # samples z from p(z) ~ N(0,1)
        z = torch.randn(
            *c.shape[:-1],
            self.zdim,
            device=c.device,
            dtype=c.dtype,
        ) * Tz
        
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr

        if return_stats==True:
            return r, (mu, log_sigma)
        else:
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
        No local transformation.
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

        return r_next_phys

    def _sample_local_from_state(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        """
        state_np: (..., k)
        returns: (..., 6) xyz aux next as [r1_next(3), r2_next(3)]
        """

        # ---- Reshape and normalise ---- #
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False
        if c_n_np.ndim == 1:
            # Integrator sampling
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True
            
            # Assuming first 6 coordinates in label are positions
            q1, q2 = c_n_np[..., :3], c_n_np[..., 3:6]
            # cutting out positions from the label
            c_n_np = c_n_np[..., 6:]
        
        if single_sample == False:
            raise ValueError('Use function _sample_local_from_state in Integrator only. Use _sample_from_conditioning directly.')
        # torchify for geometry (batch)
        q1_t, q2_t = torch.from_numpy(q1).to(device=device), torch.from_numpy(q2).to(device=device)

        c_t = torch.from_numpy(c_n_np).to(device=device)
        # build local frame
        # (store boxsize in self.parameters or self.boxsize)
        boxsize = 5.0
        R, bond_len = build_local_frame(q1_t, q2_t, boxsize=boxsize)  # R:(B,3,3), bond_len:(B,)

        # Converting to local frame
        if self.cond_type=='local_pipimririm':
            # no dq in this cond_type's conditioning either; set it only to
            # satisfy the shared sanity check below (pre-existing bug fixed
            # here -- this branch previously left `dq` unbound, which was
            # never hit because no rollout had exercised this cond_type
            # before).
            dq = bond_len[:, None]
            c_loc_t = to_local(R, c_t)
        elif self.cond_type in ('local_dqipipimririm', 'local_dqipiririm'):
            # first coordinate "after"
            dq = c_t[..., 0:1]
            c_loc_t = torch.cat((dq, to_local(R, c_t[..., 1:])), dim=-1)
            # Optional but very useful sanity check

        elif self.cond_type in ('local_dqidpipipimririm', 'local_dqidpipimmrimm'):
            # first coordinate "after"
            dq = c_t[..., 0:1]
            dp = c_t[..., 1:2]
            c_loc_t = torch.cat((dq, dp, to_local(R, c_t[..., 2:])), dim=-1)

        elif self.cond_type == 'local_dqipiri':
            # no separate dq label is passed for this cond_type (the
            # integrator's raw return has no relDistance element) — dq IS
            # bond_len by construction, so the allclose check below passes
            # trivially (same tensor).
            dq = bond_len[:, None]
            c_loc_t = torch.cat((dq, to_local(R, c_t)), dim=-1)

        elif self.cond_type == 'local_dqiririm':
            # velocity-free (E5). Raw state carries no relDistance element, so
            # dq IS bond_len by construction and the check below is trivially
            # satisfied -- same convention as local_dqipiri.
            dq = bond_len[:, None]
            c_loc_t = torch.cat((dq, to_local(R, c_t)), dim=-1)

        elif self.cond_type == 'local_piri':
            # raw state is [q1,q2,v1,v2,r1,r2] (same as local_dqipiri) but dq
            # is dropped from the conditioning entirely; dq is only set here
            # to satisfy the shared sanity check below.
            dq = bond_len[:, None]
            c_loc_t = to_local(R, c_t)

        elif self.cond_type == 'local_dqipi':
            # raw state is [q1,q2,v1,v2] (no r) — the integrator branch for
            # this cond_type returns a leaner state since r isn't used.
            dq = bond_len[:, None]
            c_loc_t = torch.cat((dq, to_local(R, c_t)), dim=-1)

        if not torch.allclose(dq, bond_len[:, None], rtol=1e-4, atol=1e-5):
            raise ValueError(
                f"Provided dx={dq.detach().cpu().numpy()} does not match "
                f"bond_len from q1/q2={bond_len.detach().cpu().numpy()}"
            )

        c_loc_np = c_loc_t.detach().cpu().numpy().astype(np.float32)

        # sample in local frame
        out_loc = self._sample_from_conditioning(c_loc_np, Tr=Tr, Tz=Tz, device=device)  # np

        out_loc_t = torch.from_numpy(out_loc).to(device=device)
        out_xyz = to_xyz(R, out_loc_t).cpu().numpy()

        return out_xyz.squeeze(0) if single_sample else out_xyz

class CVAESampler_E3(models.CVAE_E3):
    def __init__(self, zdim=3, system_type="dimer", cond_type="E3_base", hidden=(256,256)):
        super().__init__(
            zdim=zdim,
            system_type=system_type, 
            cond_type=cond_type, 
            hidden=hidden)

    def _e3_conditioning_and_basis(self, c_n_np, device):
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        if c_n_np.ndim == 1:
            c_n_np = c_n_np[None]

        q1, q2, v1, v2, v1p, v2p, r1, r2, r1p, r2p = transforms.unpack_state_vector(
            c_n_np, self.cond_type, device=device
        )

        # --- Bond axis ---
        _, bond_len = build_local_frame(q1, q2, boxsize=5.0)
        d = minimal_image_rel(q1, q2, boxsize=5.0)
        dx = bond_len.unsqueeze(-1)        # (B,1)
        e  = d / dx.clamp_min(1e-12)      # (B,3) unit bond vector

        input_vectors = torch.stack([e, v1, v2, r1, r2, v1p, v2p, r1p, r2p], dim=-2) # (B, N_vec, 3)
        # Normalize rows so all basis vectors are unit-length.
        # This makes projection coefficients p_i = v̂_i · r all live on the same scale
        # and keeps V^T V well-conditioned regardless of physical magnitude differences
        # between bond axis (unit), velocities (~0.2), and noise vectors (~0.016).
        # The reconstruction r = (V^T V)^{-1} V^T p is identical before and after normalization.
        row_norms = torch.linalg.norm(input_vectors, dim=-1, keepdim=True).clamp_min(1e-12)
        input_vectors = input_vectors / row_norms

        def axial(x): return torch.sum(x * e, dim=-1, keepdim=True)
        def norm_(x): return torch.linalg.norm(x, dim=-1, keepdim=True)

        dvx = axial(v2 - v1)

        # building conditioning out of scalars only
        c_scalar_t = torch.cat([
            dx,   dvx,
            norm_(v1),  axial(v1),
            norm_(v2),  axial(v2),
            norm_(v1p), axial(v1p),
            norm_(v2p), axial(v2p),
            norm_(r1),  axial(r1),
            norm_(r2),  axial(r2),
            norm_(r1p), axial(r1p),
            norm_(r2p), axial(r2p),
        ], dim=-1)   # (B, 18)

        return c_scalar_t, input_vectors

    def _projected_vectors_to_xyz(self, p_phys, input_vectors):
        """Reparametrize projection coefficients into output"""
        p1_phys = p_phys[:, :self.N_vec]
        p2_phys = p_phys[:, self.N_vec:]

        # Reconstruct r via least squares: input_vectors @ r = p
        WtW = torch.bmm(input_vectors.transpose(-2, -1), input_vectors)  # (B, 3, 3)
        r1_next = torch.linalg.solve(
            WtW,
            torch.bmm(input_vectors.transpose(-2, -1), p1_phys.unsqueeze(-1))
        ).squeeze(-1)
        r2_next = torch.linalg.solve(
            WtW,
            torch.bmm(input_vectors.transpose(-2, -1), p2_phys.unsqueeze(-1))
        ).squeeze(-1)

        return torch.cat((r1_next, r2_next), dim=-1)

    def physical_to_model_space(self, r_next_np, c_n_np, device=None):
        """
        Converts physical vectors to the equivariant representation.
        Outputs stay unnormalized.
        """
        if device is None:
            device = next(self.parameters()).device

        r_next_np = np.asarray(r_next_np, dtype=np.float32)
        single = r_next_np.ndim == 1
        if single:
            r_next_np = r_next_np[None]
        if r_next_np.shape[-1] != 6:
            raise ValueError(f"E3 physical targets must have shape (..., 6), got {r_next_np.shape}")

        c_scalar_t, input_vectors = self._e3_conditioning_and_basis(c_n_np, device)
        r_next_t = torch.as_tensor(r_next_np, dtype=torch.float32, device=device)
        r1_next = r_next_t[:, :3]
        r2_next = r_next_t[:, 3:6]

        p1 = torch.bmm(input_vectors, r1_next.unsqueeze(-1)).squeeze(-1)
        p2 = torch.bmm(input_vectors, r2_next.unsqueeze(-1)).squeeze(-1)
        r_model_t = torch.cat([p1, p2], dim=-1)

        r_model = r_model_t.detach().cpu().numpy()
        c_model = c_scalar_t.detach().cpu().numpy()
        if single:
            return r_model.squeeze(0), c_model.squeeze(0)
        return r_model, c_model

    def projected_output_to_physical(self, output_norm, c_n_np, device=None):
        if self.scaler_r is None:
            raise ValueError("Call attach_normalizers() before converting E3 outputs.")
        if device is None:
            device = next(self.parameters()).device

        if torch.is_tensor(output_norm):
            output_norm_np = output_norm.detach().cpu().numpy()
        else:
            output_norm_np = np.asarray(output_norm, dtype=np.float32)

        single = output_norm_np.ndim == 1
        if single:
            output_norm_np = output_norm_np[None]

        _, input_vectors = self._e3_conditioning_and_basis(c_n_np, device)
        p_phys = torch.as_tensor(
            self.scaler_r.inverse_transform(output_norm_np).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        r_next = self._projected_vectors_to_xyz(p_phys, input_vectors)
        r_next_np = r_next.detach().cpu().numpy()

        return r_next_np.squeeze(0) if single else r_next_np

    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        if self.scaler_c is None or self.scaler_r is None:
            raise ValueError("Call attach_normalizers() before sample().")
        if device is None:
            device = next(self.parameters()).device
        if hasattr(self, "Tr"): Tr = self.Tr
        if hasattr(self, "Tz"): Tz = self.Tz

        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single = c_n_np.ndim == 1
        if single:
            c_n_np = c_n_np[None]  # (1, 30)

        c_scalar_t, input_vectors = self._e3_conditioning_and_basis(c_n_np, device)

        # --- Normalize, sample in local frame, denormalize ---
        c_np = c_scalar_t.cpu().numpy().astype(np.float32)
        c_norm = self.scaler_c.transform(c_np).astype(np.float32)
        c_norm_t = torch.from_numpy(c_norm).to(device)

        alpha = self.sample_torch(c_norm_t, Tr=Tr, Tz=Tz) # (1, idim)
        # Equivariant reconstruction
        p_phys = torch.from_numpy(
        self.scaler_r.inverse_transform(alpha.cpu().numpy()).astype(np.float32)
        ).to(device)  # (B, 18)

        r_next = self._projected_vectors_to_xyz(p_phys, input_vectors)
        r_next_np = r_next.detach().cpu().numpy()

        return r_next_np.squeeze(0) if single else r_next_np
    

# ---------- CVAE ----------
class CVAE_SP(CVAESampler):
    """
    Standard Gaussian prior CVAE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The base CVAE builds a learned-prior MLP (self.prior). This model uses a
        # fixed N(0, I) prior (prior_params overridden below), so that network is
        # never used in forward() or sampling. Drop it so it isn't carried as dead
        # parameters in checkpoints.
        if hasattr(self, "prior"):
            del self.prior

    def prior_params(self, c=None, batch_shape=None, device=None, dtype=None):
        """
        Standard Gaussian prior:

            p(z) = N(0, I)

        Returns p_mu = 0, p_logvar = 0.

        c is accepted only for API compatibility with the learned-prior version.
        """
        if c is not None:
            shape = (*c.shape[:-1], self.zdim)
            device = c.device
            dtype = c.dtype
        else:
            assert batch_shape is not None, "Need either c or batch_shape."
            shape = (*batch_shape, self.zdim)

        p_mu = torch.zeros(shape, device=device, dtype=dtype)
        p_logvar = torch.zeros(shape, device=device, dtype=dtype)
        return p_mu, p_logvar

    # ----- sampling ----- #
    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0, return_stats=False):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        # samples z from p(z) ~ N(0,1)
        z = torch.randn(
            *c.shape[:-1],
            self.zdim,
            device=c.device,
            dtype=c.dtype,
        ) * Tz
        
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr

        if return_stats==True:
            return r, (mu, log_sigma)
        else:
            return r




class CVAE_Inv(CVAESampler):
    """
    CVAE sampler with invariant scalar conditioning and local-frame output.
    Supports cond_type='inv_pipimririm'.
    
    The integrator passes a raw state vector of shape (30,):
        [q1(3), q2(3), v1(3), v2(3), v1_prev(3), v2_prev(3),
         r1(3), r2(3), r1_prev(3), r2_prev(3)]
    This class extracts invariant scalars from the state, runs the CVAE,
    and rotates the local-frame output back to xyz.
    """

    @staticmethod
    def assign_dims(system_type, cond_type):
        idim_map = {"dimer": 6}
        cdim_map = {"dimer": {"inv_pipimririm": 22}}
        assert system_type in idim_map
        assert cond_type in cdim_map[system_type]
        return idim_map[system_type], cdim_map[system_type][cond_type]

    def prior_params(self, c=None, batch_shape=None, device=None, dtype=None):
        if c is not None:
            shape = (*c.shape[:-1], self.zdim); device = c.device; dtype = c.dtype
        else:
            shape = (*batch_shape, self.zdim)
        return torch.zeros(shape, device=device, dtype=dtype), \
               torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0, return_stats=False):
        z = torch.randn(*c.shape[:-1], self.zdim, device=c.device, dtype=c.dtype) * Tz
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return (r, (mu, log_sigma)) if return_stats else r

    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        if self.scaler_c is None or self.scaler_r is None:
            raise ValueError("Call attach_normalizers() before sample().")
        if device is None:
            device = next(self.parameters()).device
        if hasattr(self, "Tr"): Tr = self.Tr
        if hasattr(self, "Tz"): Tz = self.Tz

        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single = c_n_np.ndim == 1
        if single:
            c_n_np = c_n_np[None]  # (1, 30)

        # --- Unpack state vector ---
        q1 = torch.from_numpy(c_n_np[:, 0:3]).to(device)
        q2 = torch.from_numpy(c_n_np[:, 3:6]).to(device)

        v1 = torch.from_numpy(c_n_np[:, 6:9]).to(device)
        v2 = torch.from_numpy(c_n_np[:, 9:12]).to(device)
        v1p= torch.from_numpy(c_n_np[:, 12:15]).to(device)
        v2p= torch.from_numpy(c_n_np[:, 15:18]).to(device)
        r1 = torch.from_numpy(c_n_np[:, 18:21]).to(device)
        r2 = torch.from_numpy(c_n_np[:, 21:24]).to(device)
        r1p= torch.from_numpy(c_n_np[:, 24:27]).to(device)
        r2p= torch.from_numpy(c_n_np[:, 27:30]).to(device)

        # --- Bond axis ---
        R, bond_len = build_local_frame(q1, q2, boxsize=5.0)
        d = minimal_image_rel(q1, q2, boxsize=5.0)
        dx = bond_len.unsqueeze(-1)        # (B,1)
        e  = d / dx.clamp_min(1e-12)      # (B,3) unit bond vector

        def axial(x): return torch.sum(x * e, dim=-1, keepdim=True)
        def norm_(x): return torch.linalg.norm(x, dim=-1, keepdim=True)

        dvx = axial(v2 - v1)

        c_t = torch.cat([
            dx,   dvx,
            norm_(v1),  axial(v1),
            norm_(v2),  axial(v2),
            norm_(v1p), axial(v1p),
            norm_(v2p), axial(v2p),
            norm_(r1),  axial(r1),
            norm_(r2),  axial(r2),
            norm_(r1p), axial(r1p),
            norm_(r2p), axial(r2p),
        ], dim=-1)   # (B, 18)

        # --- Normalize, sample in local frame, denormalize ---
        c_np = c_t.cpu().numpy().astype(np.float32)
        c_norm = self.scaler_c.transform(c_np).astype(np.float32)
        c_norm_t = torch.from_numpy(c_norm).to(device)

        r_loc_norm = self.sample_torch(c_norm_t, Tr=Tr, Tz=Tz)
        r_loc = self.scaler_r.inverse_transform(r_loc_norm.cpu().numpy())

        # --- Rotate local-frame output back to xyz ---
        r_loc_t = torch.from_numpy(r_loc.astype(np.float32)).to(device)
        r_xyz = to_xyz(R, r_loc_t).cpu().numpy()

        return r_xyz.squeeze(0) if single else r_xyz
    
# ---------- CVAE ----------
class CVAE_MDN(CVAE_LF):
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__(zdim, system_type, cond_type, hidden)
        
        assert system_type == "dimer"

        self.K = 3
        self.log_sig_min = -5.0
        self.log_sig_max = 1.5

        out_dim_dec = self.K + self.K * self.idim + self.K * self.idim
        
        # different decoder dimensionality compared to normal CVAE
        self.decoder = MLP(zdim+self.cdim, out_dim=out_dim_dec, hidden=hidden)

    def decode(self, z, c):
        """
        MDN decoder. Returns
        -------
        logit_pi : [B, K]
        mu       : [B, K, idim]
        log_sig  : [B, K, idim]
        """
        out = self.decoder(torch.cat([z, c], dim=-1))
        B = out.shape[0]

        i0 = self.K
        i1 = i0 + self.K * self.idim
        i2 = i1 + self.K * self.idim

        logit_pi = out[:, :i0]                                # [B, K]
        mu = out[:, i0:i1].view(B, self.K, self.idim)         # [B, K, idim]
        log_sig = out[:, i1:i2].view(B, self.K, self.idim)    # [B, K, idim]

        log_sig = torch.clamp(log_sig, self.log_sig_min, self.log_sig_max)

        return logit_pi, mu, log_sig
    
    # ----- sampling ----- #
    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0, return_component=False):
        """
        Sample r_next given normalized conditioning c.

        Returns
        -------
        r_samp : [B, idim]
        comp_idx : [B] if return_component=True
        """
        # samples z from p(z) ~ N(0,1)
        z = torch.randn(
            *c.shape[:-1],
            self.zdim,
            device=c.device,
            dtype=c.dtype,
        ) * Tz
        
        logit_pi, mu, log_sig = self.decode(z, c)   # [B,K], [B,K,D], [B,K,D]

        pi = nn.functional.softmax(logit_pi, dim=-1)          # [B,K]
        comp_idx = torch.multinomial(pi, num_samples=1).squeeze(-1)   # [B]

        batch_idx = torch.arange(c.shape[0], device=c.device)
        mu_sel = mu[batch_idx, comp_idx, :]               # [B,D]
        log_sig_sel = log_sig[batch_idx, comp_idx, :]     # [B,D]

        r_samp = mu_sel + torch.exp(log_sig_sel) * torch.randn_like(mu_sel) * Tr

        if return_component:
            return r_samp, comp_idx
        return r_samp