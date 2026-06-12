import torch
import torch.nn as nn

"""
Some other CVAE class with various modifications to test out ideas.
"""

class CVAE_FullGaussian(CVAE):

    def __init__(self, zdim=3, system_type="dimer", cond_type="piri"):
        super().__init__(zdim, system_type, cond_type)
        self.decoder = self.decoder = FullGaussianHead(zdim + self.cdim, self.idim)

    def decode(self, z, c):
        mu, L = self.decoder(torch.cat([z, c], dim=-1))
        return mu, L

    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0):
        p_mu, p_logv = self.prior_params(c)
        z = reparam(p_mu, p_logv, Tz=Tz)

        mu, L = self.decode(z, c)

        eps = torch.randn_like(mu)  # (..., D)
        r = mu + Tr * torch.einsum('...ij,...j->...i', L, eps)
        return r


# ---------- CVAE_AR ----------
class CVAE_AR(CVAE):

    def __init__(self, zdim=3, system_type="dimer", cond_type="piri"):
        super().__init__(zdim, system_type, cond_type)
        self.prior   = MLP(self.cdim, out_dim=2*self.zdim+1, hidden=(128,128))

        self.z_prev = None
        self.df = nn.Parameter(torch.tensor(6.0))

    @staticmethod
    def assign_dims(system_type: str, cond_type: str) -> tuple[int, int]:
        # cdim mapping by (system_type, cond_type)
        cdim_map = {
            "bistable": {
                "piri": 6,
                "piririm": 9,
                "pipimri": 9,
            },
            "dimer": {
                "pipimririm": 24,
                "pimmrimm": 36
            },
        }

        idim, cdim = CVAE.assign_dims(system_type, cond_type, cdim_map)
        return idim, cdim

    def prior_params(self, c):
        """ p(z|c) → μ_p, logσ²_p, ρ (0<rho<1) """
        p = self.prior(c)
        p_mu, p_logv, raw_rho = p.split(self.zdim, dim=-1)
        
        p_logv = torch.clamp(p_logv, -1.8, 1.5)

        rho_min, rho_max, temp = 0.05, 0.9, 1.0
        rho = sigmoid_box(raw_rho, rho_min, rho_max, temp)  # (rho_min, rho_max)
        return p_mu, p_logv, rho

    def forward(self, r_next, c, r_next_prev, c_prev, is_new_seq):      
        """
        Forward pass of the conditional VAE with autoregressive prior.

        Args:
            r_next (Tensor): Target variable at the current step.
            c (Tensor): Conditioning variables (e.g., velocity, auxiliary state) at the current step.
            
            r_next_prev (Tensor): Target variable from the previous step (for teacher forcing).
            c_prev (Tensor): Conditioning variables from the previous step.
            
            is_new_seq (Tensor): Binary mask (0/1) indicating sequence boundaries 
                                 (1 for the start of a new sequence).

        Returns:
            dec_out (Tensor): Decoder output (reconstruction of r_next).
            (q_mu, q_logv) (Tuple[Tensor, Tensor]): Mean and log-variance of the posterior q(z|r_next, c).
            (ar_mu, p_logv) (Tuple[Tensor, Tensor]): Mean and log-variance of the autoregressive prior p(z_t|z_{t-1}, c).
            rho_eff (Tensor): Effective autoregressive coefficient
        """
        
        # prior (μ_p, σ_p, ρ) at current step
        p_mu, p_logv, rho = self.prior_params(c)

        # current posterior
        q_mu, q_logv = self.encode(r_next, c)
        z_q = reparam(q_mu, q_logv)
        
        # previous latent from previous step (teacher forcing)
        prev_q_mu, prev_q_logv = self.encode(r_next_prev, c_prev)
        z_prev = prev_q_mu.detach()
        
        #zero-out AR link at sequence starts
        if is_new_seq.dim() == 1:
            is_new_seq = is_new_seq.unsqueeze(-1)
        rho_eff = rho * (1.0 - is_new_seq)  # 0 at start; rho elsewhere
        
        # AR(1) prior mean
        ar_mu = rho_eff*z_prev + (1.0-rho_eff)*p_mu
        
        dec_out = self.decode(z_q, c)
        return dec_out, (q_mu, q_logv), (ar_mu, p_logv), rho_eff

    def reset_latent_state(self, batch_size=1):
        """Call at the start of a new simulation trajectory"""
        self.z_prev = torch.zeros(batch_size, self.zdim, device='cpu')

    @torch.no_grad()
    def sample_torch(self, c_t, Tz=1.0, Tr=1.0):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        if self.z_prev is None:
            self.reset_latent_state(batch_size=c_t.shape[0])
            
        p_mu, p_logv, rho = self.prior_params(c_t)
        #z = reparam(p_mu, p_logv)  # sample from p(z|c)
        rho_eff = torch.clamp(rho, min=0.0, max=0.99)
        z = rho_eff * self.z_prev + (1 - rho_eff) * p_mu + torch.exp(0.5 * p_logv) * torch.randn_like(p_mu) * Tz
        self.z_prev = z.detach()
        
        mu, log_sigma = self.decode(z, c_t)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return r

class CVAE_post(CVAE_AR):

    """CVAE class with extra MLP at output."""

    def __init__(self, zdim=3, system_type="dimer", cond_type="piri"):
        super().__init__(zdim, system_type, cond_type)
        self.postout = DiagGaussianHead(2*self.idim+self.cdim, 2*self.idim, hidden=(64,64))

    def post(self, dec_out, c):
        """ processing decoder out with conditioning """
        mu_dec, log_sig_dec = dec_out  
        mu, log_sigma = self.postout(torch.cat([mu_dec, log_sig_dec, c], dim=-1))
        return mu, log_sigma

    def decode(self, z, c):
        """ p(x|z,c) → μ_r, logσ²_r """
        dec_out = self.decoder(torch.cat([z, c], dim=-1))
        #log_sigma = softplus_floor(raw_logsig, floor=-1.6)  # try -1.5 first (σ_min ≈ 0.223)
        mu, log_sigma = self.post(dec_out, c)
        return mu, log_sigma#, raw_logsig

class CVAE_x(CVAE_post):
    def __init__(self, zdim=3, system_type="dimer", cond_type="piri"):
        super().__init__(zdim, system_type, cond_type)
        self.cond_scale = nn.Parameter(torch.tensor(0.0))
        self.cond_encoder = MLP(self.cdim, out_dim=self.cdim, hidden=(64, 64))

    def encode_cond(self, c):
        dc = self.cond_encoder(c)
        return c + torch.tanh(self.cond_scale) * dc

    def encode(self, r_next, c):
        """ q(z|x,c) → μ_q, logσ²_q """

        c_feat = self.encode_cond(c)
        q = self.encoder(torch.cat([r_next, c_feat], dim=-1))
        q_mu, q_logvar = q.split(self.zdim, dim=-1)
        return q_mu, q_logvar

    def prior_params(self, c):
        """ p(z|c) → μ_p, logσ²_p, ρ (0<rho<1) """
        c_feat = self.encode_cond(c)
        p = self.prior(c_feat)
        p_mu, p_logv, raw_rho = p.split(self.zdim, dim=-1)
        
        p_logv = torch.clamp(p_logv, -1.8, 1.5)

        rho_min, rho_max, temp = 0.05, 0.9, 1.0
        rho = sigmoid_box(raw_rho, rho_min, rho_max, temp)  # (rho_min, rho_max)
        return p_mu, p_logv, rho

    def decode(self, z, c):
        """ p(x|z,c) → μ_r, logσ²_r """
        c_feat = self.encode_cond(c)
        dec_out = self.decoder(torch.cat([z, c_feat], dim=-1))
        #log_sigma = softplus_floor(raw_logsig, floor=-1.6)  # try -1.5 first (σ_min ≈ 0.223)
        mu, log_sigma = self.post(dec_out, c_feat)
        return mu, log_sigma


# ---------- CVAE ----------
class CVAE_LF_old(CVAE):
    def __init__(self, zdim=3, system_type="dimer", cond_type=None):
        super().__init__(zdim, system_type, cond_type)

    @staticmethod
    def assign_dims(system_type: str, cond_type: str) -> tuple[int, int]:
        # cdim mapping by (system_type, cond_type)
        cdim_map = {
            "dimer": {
                "local_dqipiri": 13,
                "local_abs_dqipiri": 13,
                "local_dqipiwiri": 14,
                "local_dqipipimririm": 25,
                "local_abs_dqipipimririm": 25
            },
        }
        
        assert cond_type in cdim_map[system_type], (
            f"Unsupported cond_type={cond_type!r} for system_type={system_type!r} for CVAE_LF. "
            f"Supported: {tuple(cdim_map[system_type].keys())}"
        )
        return super().assign_dims(system_type, cond_type, cdim_map)

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

        return self._sample_local_from_state(c_n_np, Tr=Tr, Tz=Tz, device=device)


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

class defaultSamplingModelDimer:
    '''
    Default sampler to be fed into noise sampler for testing cases in dimer system.
    '''
    def __init__(self, mean = [0,0,0], covariance = [[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]]):
        self.mean = mean
        self.covariance = covariance


    def sample(self, conditionedVariables):
        if isinstance(self.mean, list) and isinstance(self.covariance, list):
            r1 = np.random.multivariate_normal(self.mean, self.covariance)
            r2 = np.random.multivariate_normal(self.mean, self.covariance)
        else:
            r1 = np.random.normal(self.mean, self.covariance)
            r2 = np.random.normal(self.mean, self.covariance)
        
        return np.concatenate((r1,r2))

# ---------- CVAE ----------
class CVAE_DX(nn.Module):
    """
    Sampling cvae + small conditional kick along the bond axis.
    """
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)

        # CVAE networks for drawing r0
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=hidden)
        self.prior   = MLP(self.cdim, out_dim=2*zdim, hidden=hidden)
        self.decoder = DiagGaussianHead(zdim + self.cdim, 2*self.idim, hidden=hidden)
        
        # residual for small correction
        self.dx_net = DiagGaussianHead(self.cdim, 2, hidden=hidden)
        
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
            "dimer": {
                "dRipipimdqiririm": 28,
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

    def forward(self, r_next, c, training_mode=None):
        """
        Forward pass of the model. Mode = 'res' for modeling residual from CVAE, 'mean' for getting rmean from MLP.
        """
        
        assert training_mode is not None, 'Set training mode.'
        
        if training_mode=='cvae':

            p_mu, p_logv = self.prior_params(c)
            q_mu, q_logv = self.encode(r_next, c)
            z = reparam(q_mu, q_logv)
            dec_out = self.decode(z, c)
            return dec_out, (q_mu, q_logv), (p_mu, p_logv)   
        
        elif training_mode=='res':
            "training mode for residual correction, to be implemented later."
            return self.dx_net(c)
        
    
    # ----- sampling ----- #
    @torch.no_grad()
    def get_rrel_par(self, c_phys, c_norm):
        """
        q_pair_phys: (N, 6) physical positions [q1, q2]
        c_norm:      (N, cdim) normalized conditioning
        r0_norm:     (N, 6) normalized CVAE sample

        returns:
            dr_norm:  (N, 6) normalized correction to add to r0_norm
        """
        assert self.cond_type=='dRipipimdqiririm'
        
        rel = c_phys[..., :3]   # physical rel_pos
        e1 = rel / (torch.norm(rel, dim=-1, keepdim=True) + 1e-12)

        # scalar from MLP
        mu, log_sigma = self.dx_net(c_norm)                           # (N, 1)

        dr_rel = mu + torch.exp(log_sigma) * torch.randn_like(mu) # (N, 1)

        # relative kick along bond axis, COM unchanged
        dr_rel_phys = dr_rel * e1                         # (N, 3)
        dr1_phys = -0.5 * dr_rel_phys
        dr2_phys = +0.5 * dr_rel_phys
        dr_phys = torch.cat([dr1_phys, dr2_phys], dim=-1)   # (N, 6)

        return dr_phys.cpu().numpy()
    
    @torch.no_grad()
    def sample_cvae_torch(self, c, Tr=1.0, Tz=1.0):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        p_mu, p_logv = self.prior_params(c)
        z = reparam(p_mu, p_logv, Tz=Tz)  # sample from p(z|c)
        
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return r
    
    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, Trr=1.0, device=None, add_correction=True):
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

        if self.cond_type=='dRipipimdqiririm':
            q1, q2 = torch.from_numpy(c_n_np[:, :3]), torch.from_numpy(c_n_np[:, 3:6])
            dRi = minimal_image_rel(q1, q2, boxsize=5.0, boundary_type='periodic')
            c_n_np = np.concatenate((np.array(dRi), c_n_np[:, 6:]), axis=-1)
            
        c_norm = self.scaler_c.transform(c_n_np).astype(np.float32)

        # --- Convert to torch tensor ---
        c_t = torch.from_numpy(c_norm).to(device=device)
        
        # sampling r from cvae
        r0_next_norm_t = self.sample_cvae_torch(c_t, Tr=Tr, Tz=Tz)
        r_next_np = r0_next_norm_t.cpu().numpy()
        # --- De-normalize to physical scale ---
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if add_correction==True:
            beta_dx = -0.1
            # sampling correction
            rrel_next_phys = self.get_rrel_par(torch.from_numpy(c_n_np).to(device=device), c_t)
            r_next_phys = r_next_phys + beta_dx*rrel_next_phys

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys

class CVAE_RES(CVAE):
    """
    CVAE class that samples r_next = r_mean + dr, 
    where:
        - r_mean is predicted by an MLP conditioned on c
        - dr is a residual noise sampled from CVAE network.
    """

    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__(zdim, system_type, cond_type, hidden)

        # residual for small correction
        self.post = DiagGaussianHead(self.cdim + self.idim, 2*self.idim, hidden=hidden)
        self.scaler_dr = None

    def attach_normalizers(self, scaler_r, scaler_c, scaler_dr):
        """Attach normalization scalers for automatic preprocessing."""
        self.scaler_r = scaler_r
        self.scaler_c = scaler_c
        self.scaler_dr = scaler_dr
    
    def get_res(self, c):
        mu, log_sigma = self.post(c)
        return mu, log_sigma

    # ----- sampling ----- #
    @torch.no_grad()
    def sample_res_torch(self, c, Trr=1):
        """
        Sampling mean r|c.
        """
        mu, log_sigma = self.get_res(c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Trr
        return r
    
    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None, add_correction=True):
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
        
        # sampling r from cvae
        r0_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=1)
        # sampling correction
        dr_next_norm_t = self.sample_res_torch(torch.cat([c_t, r0_next_norm_t], dim=-1), Trr=Tz)
        res_next_np = self.scaler_dr.inverse_transform(dr_next_norm_t.cpu().numpy())

        # --- De-normalize to physical scale ---
        if add_correction==True:
            r_next_np = r0_next_norm_t.cpu().numpy() + res_next_np
        else:
            r_next_np = r0_next_norm_t.cpu().numpy()
            
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys

class CVAE_MDN_old(nn.Module):
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)
                
        self.K = 2
        self.log_sig_min = -5.0
        self.log_sig_max = 1.5

        out_dim_dec = self.K + self.K * self.idim + self.K * self.idim
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=hidden)
        self.prior   = MLP(self.cdim, out_dim=2*zdim, hidden=hidden)
        self.decoder = MLP(zdim+self.cdim, out_dim=out_dim_dec, hidden=hidden)
        
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
            "dimer": {
                "pipimririm": 24,
                "pipimdqiririm": 25,
                "pipimdqidpiririm": 26,
                "pipimdidimririm": 28,
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

    def forward(self, r_next, c):
        p_mu, p_logv = self.prior_params(c)
        q_mu, q_logv = self.encode(r_next, c)
        z = reparam(q_mu, q_logv)
        dec_out = self.decode(z, c)
        return dec_out, (q_mu, q_logv), (p_mu, p_logv)   
    
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
        p_mu, p_logv = self.prior_params(c)
        z = reparam(p_mu, p_logv, Tz=Tz)  # sample from p(z|c)
        
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
    
    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
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

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=Tz)

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys

class ConditionalGaussianNN(nn.Module):
    """
    Direct conditional density model:
        p(r_next | c) = N(mu(c), diag(sigma(c)^2))

    Input:
        c:      [B, cdim]
    Output:
        mu_r:   [B, rdim]
        log_sig_r: [B, rdim]
    """
    def __init__(self, cond_type, hidden=(128, 128),
                 log_sig_min=-5.0, log_sig_max=1.5):
        super().__init__()
        assert cond_type == 'pipimdqidpiririm', f"Unsupported cond_type={cond_type!r}"
        self.cond_type = cond_type
        self.cdim = 26
        self.rdim = 6
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max

        self.net = MLP(self.cdim, 2 * self.rdim, hidden=hidden)

        # normalisers
        self.scaler_r = None
        self.scaler_c = None

    def attach_normalizers(self, scaler_r, scaler_c):
        """Attach normalization scalers for automatic preprocessing."""
        self.scaler_r = scaler_r
        self.scaler_c = scaler_c

    def forward(self, c):
        out = self.net(c)
        mu_r, log_sig_r = torch.chunk(out, 2, dim=-1)
        log_sig_r = torch.clamp(log_sig_r, self.log_sig_min, self.log_sig_max)
        return mu_r, log_sig_r

    def set_temps(self, Tr=None, Tz=None):
        """Set global sampling temperatures. Call with no args to unset."""

        if Tr is None:
            if hasattr(self, "Tr"):
                delattr(self, "Tr")
        else:
            self.Tr = Tr

    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0):
        """
        Sample r_next given conditioning c.
        Ts can be used as a sampling temperature.
        """
        mu_r, log_sig_r = self.forward(c)
        eps = torch.randn_like(mu_r)
        r_samp = mu_r + torch.exp(log_sig_r) * eps * Tr
        return r_samp

    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, device=None):
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

        # ---- Reshape and normalise ---- #
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False
        if c_n_np.ndim == 1:
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True
            
        c_norm = self.scaler_c.transform(c_n_np).astype(np.float32)

        # --- Convert to torch tensor ---
        c_t = torch.from_numpy(c_norm).to(device=device)

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr)

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys

class ConditionalGaussianMixtureNN(nn.Module):
    """
    Conditional mixture density model:
        p(r_next | c) = sum_k pi_k(c) * N(mu_k(c), diag(sigma_k(c)^2))

    Input:
        c: [B, cdim]

    Output from forward():
        logit_pi:  [B, K]
        mu:        [B, K, rdim]
        log_sig:   [B, K, rdim]
    """
    def __init__(self, cond_type, n_components=3, hidden=(128, 128),
                 log_sig_min=-5.0, log_sig_max=1.5):
        super().__init__()

        assert cond_type == 'pipimdqidpiririm', f"Unsupported cond_type={cond_type!r}"
        self.cond_type = cond_type
        self.cdim = 26
        self.rdim = 6
        self.K = n_components
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max

        out_dim = self.K + self.K * self.rdim + self.K * self.rdim
        self.net = MLP(self.cdim, out_dim, hidden=hidden)

        # normalisers
        self.scaler_r = None
        self.scaler_c = None

    def attach_normalizers(self, scaler_r, scaler_c):
        self.scaler_r = scaler_r
        self.scaler_c = scaler_c

    def forward(self, c):
        """
        Returns
        -------
        logit_pi : [B, K]
        mu       : [B, K, rdim]
        log_sig  : [B, K, rdim]
        """
        out = self.net(c)
        B = out.shape[0]

        i0 = self.K
        i1 = i0 + self.K * self.rdim
        i2 = i1 + self.K * self.rdim

        logit_pi = out[:, :i0]                                # [B, K]
        mu = out[:, i0:i1].view(B, self.K, self.rdim)        # [B, K, rdim]
        log_sig = out[:, i1:i2].view(B, self.K, self.rdim)   # [B, K, rdim]

        log_sig = torch.clamp(log_sig, self.log_sig_min, self.log_sig_max)

        return logit_pi, mu, log_sig

    def set_temps(self, Tr=None, Tz=None):
        """Set global sampling temperatures. Call with no args to unset."""

        if Tr is None:
            if hasattr(self, "Tr"):
                delattr(self, "Tr")
        else:
            self.Tr = Tr

    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, return_component=False):
        """
        Sample r_next given normalized conditioning c.

        Returns
        -------
        r_samp : [B, rdim]
        comp_idx : [B] if return_component=True
        """
        logit_pi, mu, log_sig = self.forward(c)   # [B,K], [B,K,D], [B,K,D]

        pi = F.softmax(logit_pi, dim=-1)          # [B,K]
        comp_idx = torch.multinomial(pi, num_samples=1).squeeze(-1)   # [B]

        batch_idx = torch.arange(c.shape[0], device=c.device)
        mu_sel = mu[batch_idx, comp_idx, :]               # [B,D]
        log_sig_sel = log_sig[batch_idx, comp_idx, :]     # [B,D]

        eps = torch.randn_like(mu_sel)
        r_samp = mu_sel + torch.exp(log_sig_sel) * eps * Tr

        if return_component:
            return r_samp, comp_idx
        return r_samp

    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, device=None):
        """
        Sample r_{n+1} in physical units given c_n_np in physical units.
        """
        if self.scaler_c is None or self.scaler_r is None:
            raise ValueError("Call attach_normalizers(...) before sample().")

        if device is None:
            device = next(self.parameters()).device

        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False
        if c_n_np.ndim == 1:
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True

        c_norm = self.scaler_c.transform(c_n_np).astype(np.float32)
        c_t = torch.from_numpy(c_norm).to(device=device)

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr)
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()

        return r_next_phys

class defaultSamplingModelDimer:
    '''
    Default sampler to be fed into noise sampler for testing cases in dimer system.
    '''
    def __init__(self, mean = [0,0,0], covariance = [[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.00001]]):
        self.mean = mean
        self.covariance = covariance


    def sample(self, conditionedVariables):
        if isinstance(self.mean, list) and isinstance(self.covariance, list):
            r1 = np.random.multivariate_normal(self.mean, self.covariance)
            r2 = np.random.multivariate_normal(self.mean, self.covariance)
        else:
            r1 = np.random.normal(self.mean, self.covariance)
            r2 = np.random.normal(self.mean, self.covariance)
        
        return np.concatenate((r1,r2))

class DeterministicHead(nn.Module):
    """Outputs (mu, log_sigma) for R^3."""
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even: 2 * D"
        self.mlp = MLP(in_dim, out_dim, hidden=hidden)
    def forward(self, x):
        out = self.mlp(x)
        return out
    
# ---------- CVAE ----------
class CVAE_DET(nn.Module):
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=hidden)
        self.prior   = MLP(self.cdim, out_dim=2*zdim, hidden=hidden)
        self.decoder = DeterministicHead(zdim + self.cdim, self.idim, hidden=hidden)
        
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
                "piririmrimm": 12,
                "pipimririm": 12,
            },
            "dimer": {
                "pipimririm": 24,
                "pipimdqiririm": 25,
                "pipimdpiririm": 25,
                "pipimdqidpiririm": 26,
                "pipimdidimririm": 28,
                "pimmrimm": 36,
                "piMriM": 48,
                "pimmdqidpirimm": 38,
                "piMdqidpiriM": 50,
                "relcom_pipimririm": 24
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
        return self.decoder(torch.cat([z, c], dim=-1))

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
        
        dec_out = self.decode(z, c)
        return dec_out
    
    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
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

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=Tz)

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys
