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

# ---------- CVAE_AR ----------
class CVAE_AR(nn.Module):

    def __init__(self, zdim=3, system_type="dimer", cond_type="piri"):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*self.zdim, hidden=(128,128))
        self.prior   = MLP(self.cdim, out_dim=3*self.zdim, hidden=(128,128))
        self.decoder = DiagGaussianHead(self.zdim + self.cdim, 2*self.idim)
        
        # normalisers
        self.scaler_r = None
        self.scaler_c = None

        self.z_prev = None
        self.df = nn.Parameter(torch.tensor(6.0))

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
        """ q(z|x,c) → μ_q, logσ²_q """
        q = self.encoder(torch.cat([r_next, c], dim=-1))
        q_mu, q_logvar = q.split(self.zdim, dim=-1)
        return q_mu, q_logvar

    def prior_params(self, c):
        """ p(z|c) → μ_p, logσ²_p, ρ (0<rho<1) """
        p = self.prior(c)
        p_mu, p_logv, raw_rho = p.split(self.zdim, dim=-1)
        
        p_logv = torch.clamp(p_logv, -1.8, 1.5)

        rho_min, rho_max, temp = 0.05, 0.9, 1.0
        rho = sigmoid_box(raw_rho, rho_min, rho_max, temp)  # (rho_min, rho_max)
        return p_mu, p_logv, rho

    def decode(self, z, c):
        """ p(x|z,c) → μ_r, logσ²_r """
            
        mu, log_sigma = self.decoder(torch.cat([z, c], dim=-1))
        #log_sigma = softplus_floor(raw_logsig, floor=-1.6)  # try -1.5 first (σ_min ≈ 0.223)

        return mu, log_sigma#, raw_logsig

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
        return dec_out, (q_mu, q_logv), (ar_mu, p_logv)

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
        rho_eff = torch.clamp(alpha * rho, min=0.0, max=0.99)
        z = rho_eff * self.z_prev + (1 - rho_eff) * (p_mu + torch.exp(0.5 * p_logv) * torch.randn_like(p_mu) * Tz)
        self.z_prev = z.detach()
        
        mu, log_sigma, _ = self.decode(z, c_t)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return r
    
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