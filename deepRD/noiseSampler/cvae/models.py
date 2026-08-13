import torch
import torch.nn as nn
import torchvision
from deepRD.tools.modelsTools import MLP, DiagGaussianHead, reparam, FullGaussianHead

# ---------- CVAE ----------
class CVAE(nn.Module):
    """
    Basic CVAE class. 
    Learned prior p(z|c) and Gaussian encoder/decoder with diagonal covariances.
    """
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128), dim_maps=None):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
                
        if dim_maps is None:
            idim_map, cdim_map = None, None
        else:
            idim_map, cdim_map = dim_maps
            
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type,
                                                idim_map=idim_map, cdim_map=cdim_map)
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=hidden)
        self.prior   = MLP(self.cdim, out_dim=2*zdim, hidden=hidden)
        self.decoder = DiagGaussianHead(zdim + self.cdim, 2*self.idim, hidden=hidden)
        
        # normalisers
        self.scaler_r = None
        self.scaler_c = None

    @staticmethod
    def assign_dims(system_type: str, cond_type: str, idim_map=None, cdim_map=None) -> tuple[int, int]:
        # idim by system
        if idim_map ==None:
            idim_map = {"bistable": 3, "dimer": 6}
        assert system_type in idim_map, f"Unknown system_type={system_type!r}"
        idim = idim_map[system_type]

        if cdim_map==None:
            # Default cdim mapping by (system_type, cond_type)
            cdim_map = {
                "bistable": {
                    "piri": 6,
                    "piririm": 9,
                    "pipimri": 9,
                    "ririm": 6,             # velocity-free (E5): r^n, r^{n-1}
                    "piririmrimm": 12,      # v, r^n, r^{n-1}, r^{n-2}
                    "piririmrimmriM": 15    # + r^{n-3}
                },
                "dimer": {
                    "pidqiri": 13,
                    "dqidpiri": 8,
                    "dqidpiririm": 14,
                    "pipimririm": 24,
                    "e1pipimdqiririm": 28,
                    "pipimririmrimm": 30,
                    "pipimdqiririm": 25,
                    "pipimdpiririm": 25,
                    "pipimdqidpiririm": 26,
                    "pipimdidimririm": 28,
                    "pimmrimm": 36,
                    "piMriM": 48,
                    "pimmdqidpirimm": 38,
                    "piMdqidpiriM": 50,
                    "relcom_pipimdqiririm": 25,
                    "local_dqipipimririm": 25,
                    "local_dqipiririm": 19          # no v^{n-1} (E15)
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
        """ p(z|c) → μ_p, logσ²_p"""
        p = self.prior(c)
        p_mu, p_logvar = p.split(self.zdim, dim=-1)
        return p_mu, p_logvar

    def decode(self, z, c):
        """ p(x|z,c) → μ_r, logσ²_r """
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
    def sample_torch(self, c, Tr=1.0, Tz=1.0, return_stats=False):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        p_mu, p_logv = self.prior_params(c)
        
        z = reparam(p_mu, p_logv, Tz=Tz)  # sample from p(z|c)

        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        if return_stats==True:
            return r, (mu, log_sigma)
        else:
            return r

# ---------- CVAE ----------
class CVAE_LF(nn.Module):
    """
    CVAE with N(0,1) prior and Local Frame transformation.
    """
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri", hidden=(128,128)):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=hidden)
        self.decoder = DiagGaussianHead(zdim + self.cdim, 2*self.idim, hidden=hidden)
        
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
                "local_pipimririm": 24,
                "local_dqipipimririm": 25,
                "local_dqipiririm": 19,   # no v^{n-1} (E15)
                "local_dqidpipipimririm": 26,
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
        # samples z from p(z) ~ N(0,1)
        z = torch.randn(
            *c.shape[:-1],
            self.zdim,
            device=c.device,
            dtype=c.dtype,
        ) * Tz
        
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return r


# ---------- CVAE ----------
class CVAE_E3(CVAE):
    """
    SO(3) equivariant version of the CVAE. 
    Standard prior N(0, I) and Gaussian encoder/decoder with diagonal covariances.
    """
    def __init__(self, zdim=3, system_type="dimer", cond_type="E3_base", hidden=(256,256)):

        idim_map = {"dimer": 18}
        cdim_map = {
                "dimer": {
                    "E3_base": 18
                },
            }
        dim_maps = (idim_map, cdim_map)
        super().__init__(zdim, system_type, cond_type, hidden, dim_maps)

        self.N_vec = self.idim//2
    
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
    def sample_torch(self, c, Tr=1.0, Tz=1.0):
        """
        Samples the decoder output from torch tensor, no (de)normalisation.
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
        return r
    



### other classes ###
class CVAE_FullGaussian(CVAE_LF):

    def __init__(self, zdim=3, system_type="dimer", cond_type="piri", hidden=(128,128)):
        super().__init__(zdim, system_type, cond_type, hidden)
        self.decoder = FullGaussianHead(zdim + self.cdim, self.idim, hidden=hidden)

    def decode(self, z, c):
        mu, L = self.decoder(torch.cat([z, c], dim=-1))
        return mu, L

    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0, return_stats=False):
        # samples z from p(z) ~ N(0,1)
        z = torch.randn(
            *c.shape[:-1],
            self.zdim,
            device=c.device,
            dtype=c.dtype,
        ) * Tz

        mu, L = self.decode(z, c)

        eps = torch.randn_like(mu)  # (..., D)
        r = mu + Tr * torch.einsum('...ij,...j->...i', L, eps)

        if return_stats:
            return r, (mu, L)
        return r