import torch
import math

"""
Loss functions for model training
"""

def gaussian_nll_diag(x, mu, log_sig, per_sample=False):
    """
    x, mu, log_sig: [..., idim]
    Sum over dims; mean over batch if per_sample=False, else return per-sample NLL.
    """
    nll = 0.5 * ((x - mu)**2 * torch.exp(-2*log_sig) + 2*log_sig + torch.log(torch.tensor(2*math.pi, device=x.device))).sum(-1)
    return nll.mean() if not per_sample else nll

def kl_diag(q_mu, q_logv, p_mu, p_logv, per_sample=False, free_bits=0.0):
    """
    Computes KL divergence KL(q||p) for diagonal Gaussians q and p with parameters (mu, logvar).
    all [..., zdim], where zdim is the latent dimension.
    mean over batch if per_sample=False, else per-sample KL.
    """

    kl_per_dim = 0.5 * (
        torch.exp(q_logv - p_logv)
        + (q_mu - p_mu)**2 * torch.exp(-p_logv) 
        - 1 
        + p_logv 
        - q_logv)

    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    kl = kl_per_dim.sum(-1)

    return kl.mean() if not per_sample else kl

def gaussian_nll_diag_per_sample(x, mu, log_sig):
    return gaussian_nll_diag(x, mu, log_sig, per_sample=True)

def kl_diag_per_sample(q_mu, q_logv, p_mu, p_logv):
    return kl_diag(q_mu, q_logv, p_mu, p_logv, per_sample=True)

def elbo_loss(r_next, dec_out, q, p, beta=1.0, free_bits=0.0, per_sample=False):
    """
    Compute the ELBO loss for a CVAE with diagonal Gaussian encoder and decoder.
    Set per_sample=True to return per-sample losses instead of mean loss.
    The free_bits parameter allows for a minimum KL divergence contribution.
    """

    q_mu, q_logv = q
    p_mu, p_logv = p

    mu, log_sig = dec_out
    nll = gaussian_nll_diag(r_next, mu, log_sig, per_sample=per_sample)
    kl  = kl_diag(q_mu, q_logv, p_mu, p_logv, per_sample=per_sample, free_bits=free_bits)

    return nll + beta * kl, nll, kl

def reweight_losses(losses, w):
    """
    Reweight per-sample losses by w, normalised by sum of w.
    """

    weighted_losses = []
    for loss in losses:
        loss = (w*loss).sum()/ w.sum()
        weighted_losses.append(loss)

    return weighted_losses