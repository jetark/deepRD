import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepRD.tools.trajectoryTools import minimal_image_rel, build_local_frame, to_local, to_xyz, dimer_rel_com, dimer_from_rel_com
import deepRD.noiseSampler.cvae.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- CVAE ----------
class CVAESampler(models.CVAE):
    """
    CVAE wrapper for compatibility with the Langevin Integrator sampling.
    """
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__(
            zdim=zdim,
            system_type=system_type, 
            cond_type=cond_type, 
            hidden=hidden)

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

        if self.cond_type=='e1pipimdqiririm':
            q1, q2 = torch.from_numpy(c_n_np[:, :3]), torch.from_numpy(c_n_np[:, 3:6])
            dRi = minimal_image_rel(q1, q2, boxsize=5.0, boundary_type='periodic')
            dx = torch.norm(dRi, dim=-1)
            e1 = dRi/dx
            c_n_np = np.concatenate((np.array(e1), c_n_np[:, 6:]), axis=-1)
        
        c_norm = self.scaler_c.transform(c_n_np).astype(np.float32)

        # --- Convert to torch tensor ---
        c_t = torch.from_numpy(c_norm).to(device=device)

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=Tz, return_stats=return_stats)

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if self.cond_type=="relcom_pipimririm":       
            rel, com = dimer_from_rel_com(r_next_phys[:, :3], r_next_phys[:, 3:])
            r_next_phys = np.concatenate((rel, com), axis=-1)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys

# ---------- CVAE ----------
class CVAE_SP(CVAESampler):
    
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


# ---------- CVAE with Local Frame transformation ----------
class CVAE_LF(CVAESampler):
    @staticmethod
    def assign_dims(system_type: str, cond_type: str) -> tuple[int, int]:
        # idim by system
        idim_map = {"dimer": 6}
        assert system_type in idim_map, f"Unknown system_type={system_type!r}"
        idim = idim_map[system_type]

        # cdim mapping by (system_type, cond_type)
        cdim_map = {
            "dimer": {
                "local_pipimririm": 24,
                "local_dqipipimririm": 25,
                "local_dqidpipipimririm": 26
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
            c_loc_t = to_local(R, c_t)
        elif self.cond_type=='local_dqipipimririm':
            # first coordinate "after"
            dq = c_t[..., 0:1]
            c_loc_t = torch.cat((dq, to_local(R, c_t[..., 1:])), dim=-1)
            # Optional but very useful sanity check

        elif self.cond_type=='local_dqidpipipimririm':
            # first coordinate "after"
            dq = c_t[..., 0:1]
            dp = c_t[..., 1:2]
            c_loc_t = torch.cat((dq, dp, to_local(R, c_t[..., 2:])), dim=-1)

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

