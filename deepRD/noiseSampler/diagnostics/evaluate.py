import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch

def correlation_fft(a, b, trunc):
    """Calculates correlation via FFT."""
    a = np.asarray(a)
    b = np.asarray(b)
    
    len_a = len(a)
    a = a - np.mean(a)
    b = b - np.mean(b)
    
    a = np.concatenate([a, np.zeros(len_a)])
    b = np.concatenate([b, np.zeros(len_a)])
    a_fft = np.fft.fft(a)
    b_fft = np.fft.fft(b)
    corr = np.fft.ifft(a_fft * np.conj(b_fft))
    corr = corr[:trunc].real
    corr /= np.linspace(len_a, len_a - trunc + 1, trunc)
    return corr

def compute_acf(tensor, lagtimesteps, mTrajs):
    """
    Computes normalized ACF for a single tensor.
    tensor: shape (nTrajs, nTimesteps, k)
     - nTrajs: number of trajectories
     - nTimesteps: number of time steps per trajectory

    - lagtimesteps: number of lag time steps to compute ACF for
    - mTrajs: subsample trajectories for ACF computation
    
    """
    nTrajs = tensor.shape[0]
    ACF = np.zeros(lagtimesteps)

    for trajInd in np.random.choice(nTrajs, mTrajs, replace=False):
        # Sum over the last dimension (k)

        ACF += np.sum([correlation_fft(tensor[trajInd, :, d], tensor[trajInd, :, d], lagtimesteps) 
                              for d in range(tensor.shape[-1])], axis=0)

    ACF /= ACF[0]
    return ACF

def get_binned_stats(x, r_scalar, nbins=30, rmin=None, rmax=None, min_count=100):
    """
    Compute conditional mean/variance of a vector as a function of bond length dx.

    Args
    ----
    x        : (n_traj, n_time, 3) 3-D vector
    r_scalar  : (n_traj, n_time) scalar value for binning (e.g bond length)
    nbins     : number of r-bins
    rmin/rmax : optional bin range
    min_count : minimum samples per bin (else NaN)

    Returns
    -------
    centers   : (nbins,)
    mean_x    : (nbins,)
    var_x     : (nbins,)
    counts     : (nbins,) number of samples in each bin
    """
    x = x.reshape(-1, 3).detach().cpu()
    rr = r_scalar.reshape(-1).detach().cpu()

    if rmin is None: rmin = rr.min().item()
    if rmax is None: rmax = rr.max().item()

    edges = torch.linspace(rmin, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_x = torch.full((nbins,), float("nan"))
    var_x  = torch.full((nbins,), float("nan"))
    counts  = torch.zeros(nbins, dtype=torch.long)

    for i in range(nbins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        c = int(m.sum().item())
        counts[i] = c
        if c < min_count:
            continue

        mean_x[i] = x[m].mean(dim=0).mean()
        var_x[i]  = x[m].var(dim=0, unbiased=True).mean()

    return centers, mean_x, var_x, counts

def get_binned_stats_local(x_loc, r_scalar, nbins=30, rmin=None, rmax=None, min_count=100):
    """
    Compute conditional mean/variance of parallel component and perp-average
    as a function of bond length r.
    Assuming input vector is in local frame.

    Args
    ----
    x_loc   : (n_traj, n_time, 3) local components
    r_scalar: (n_traj, n_time) bond length
    nbins   : number of r-bins
    rmin/rmax: optional bin range
    min_count: minimum samples per bin (else NaN)

    Returns
    -------
    centers : (nbins,)
    mean_par, mean_perp : (nbins,)
    var_par,  var_perp  : (nbins,)
    counts  : (nbins,) number of samples in each bin
    """
    x = x_loc.reshape(-1, 3).detach().cpu()
    rr = r_scalar.reshape(-1).detach().cpu()

    if rmin is None: rmin = rr.min().item()
    if rmax is None: rmax = rr.max().item()

    edges = torch.linspace(rmin, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_par  = torch.full((nbins,), float("nan"))
    mean_perp = torch.full((nbins,), float("nan"))
    var_par   = torch.full((nbins,), float("nan"))
    var_perp  = torch.full((nbins,), float("nan"))
    counts    = torch.zeros(nbins, dtype=torch.long)

    for i in range(nbins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        c = int(m.sum().item())
        counts[i] = c
        if c < min_count:
            continue

        # parallel component
        xpar = x[m, 0]
        mean_par[i] = xpar.mean()
        var_par[i]  = xpar.var(unbiased=True)

        # perp average: treat the two perpendicular components as one pooled set
        # (better than averaging two separate means/vars when means might differ slightly)
        xperp = torch.cat([x[m, 1], x[m, 2]], dim=0)
        mean_perp[i] = xperp.mean()
        var_perp[i]  = xperp.var(unbiased=True)

    return centers, mean_par, mean_perp, var_par, var_perp, counts