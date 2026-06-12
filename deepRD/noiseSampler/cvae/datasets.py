import numpy as np
import torch
import deepRD.tools.trajectoryTools as trajectoryTools
import deepRD.tools.analysisTools as analysisTools
from deepRD.noiseSampler.cvae.transforms import minimal_image_rel, compute_axisRelVel, compute_dx_dvx, to_local, build_local_frame
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


"""
General functions to load and extract/manipulate data from simulations
"""

def load_fpt_file(path):
    with open(path, "r") as f:
        return np.array([float(line.strip()) for line in f if line.strip()], dtype=float)

def load_model_fpt(paths, base_dir=None):
        """
        Load multiple FPT files and return a list of numpy arrays.

        Args:
            paths (list of str): filenames or full paths to FPT files.
            base_dir (str, optional): directory to prepend to filenames in paths when they are not absolute.

        Returns:
            list of numpy.ndarray: loaded FPT arrays.
        """
        out = []
        for p in paths:
                fp = p if os.path.isabs(p) or base_dir is None else os.path.join(base_dir, p)
                out.append(load_fpt_file(fp))
        return out
        

def load_datasets(datasetDirectory, n_datasets, n_total, trajtype='bench'):
    """
    Load datasets from simulation files.

    Args:
    - localDirectory (str): Directory containing simulation files.
    - n_datasets (int): Total number of datasets to load.
    - n_Total (int): Total number of available simulation files.
    - type (str): 'bench' for full datasets, 'reduced' for reduced datasets.

    Returns:
    - dataset (torch.Tensor): Combined dataset tensor.
    - parameters (dict): Loaded parameters from the parameters file.
    """

    fileName = "simMoriZwanzigReduced_" if type=='reduced' else "simMoriZwanzig_"
    
    # Sample simulation files randomly
    fnums = np.sort(np.random.choice(n_total, n_datasets, replace=False))
    dataset = None

    for i, f_num in enumerate(fnums):
        try:
            ds = torch.Tensor(trajectoryTools.loadTrajectory(datasetDirectory + fileName, f_num)).unsqueeze(0)
        except FileNotFoundError:
            print(f'File {f_num} not available.')
            continue

        if dataset is None:
            dataset = ds
        else:
            dataset = torch.cat((dataset, ds), dim=0)
        print(f'File no. {i+1}/{n_datasets} loaded', end='\r')

    # Load parameters from parameters file
    parameters = analysisTools.readParameters(datasetDirectory + "parameters")
    
    return dataset, parameters

def extract_vars(dataset):
    """
    Extract q, v, r from the dataset tensor.
    """
    q = dataset[..., 1:4]  # position (not used now, but may be later)
    v = dataset[..., 4:7]   # velocity
    r = dataset[..., 8:11]  # auxiliary var

    return q, v, r

def normalize_RC(r, c, scaler_r, scaler_c):
    """
    Normalize r and c using provided pre-fit scalers.
    """
    r_next_norm = torch.tensor(scaler_r.transform(r_next), dtype=torch.float32)
    c_norm = torch.tensor(scaler_c.transform(c), dtype=torch.float32)

    return r_next_norm, c_norm

"""
Functions related to the construction of datasets for training specifically.
"""

class RCDataset(Dataset):
    """
    Simple dataset serving (r_next, c) pairs.

    r_next_all : torch.Tensor [N, idim]
    c_all      : torch.Tensor [N, cdim]
    """
    def __init__(self, r_next_all, c_all):
        super().__init__()
        assert isinstance(r_next_all, torch.Tensor)
        assert isinstance(c_all, torch.Tensor)
        assert r_next_all.shape[0] == c_all.shape[0]

        self.r_next = r_next_all
        self.c      = c_all

    def __len__(self):
        return self.r_next.shape[0]

    def __getitem__(self, idx):
        return self.r_next[idx], self.c[idx]

class RVSeqDataset(Dataset):
    """
    Dataset of sliding windows over trajectories of (r_next, c).

    Args
    ----
    r_next_seq : torch.Tensor, shape [N_traj, T, idim]
        r_{n+step} (or whatever you've precomputed per time step)
    c_seq      : torch.Tensor, shape [N_traj, T, cdim]
        conditioning vectors c_n
    L          : int >= 1
        window length (L=1 gives single-step windows)
    step       : int >= 1
        stride between window starts (in time indices)

    Returns
    -------
    __getitem__(idx) -> (r_win, c_win):
        r_win : [L, idim]
        c_win : [L, cdim]
    """
    def __init__(self,
                 r_next_seq: torch.Tensor,
                 c_seq: torch.Tensor,
                 w_seq: torch.Tensor,
                 L: int,
                 step: int = 1):
        super().__init__()

        assert isinstance(r_next_seq, torch.Tensor)
        assert isinstance(c_seq, torch.Tensor)
        assert r_next_seq.ndim == 3, "r_next_seq must be [N_traj, T, idim]"
        assert c_seq.ndim == 3, "c_seq must be [N_traj, T, cdim]"
        assert r_next_seq.shape[:2] == c_seq.shape[:2], \
            "r_next_seq and c_seq must match in [N_traj, T]"
        assert L >= 1, "L must be >= 1"
        assert step >= 1, "step must be >= 1"

        self.r_next_seq = r_next_seq       # [N_traj, T, 3]
        self.c_seq      = c_seq            # [N_traj, T, cdim]
        self.L = L
        self.step = step

        if w_seq is not None:
            self.w_seq      = w_seq
        else:
            self.w_seq = torch.ones_like(c_seq[..., :1])

        self.N_traj, self.T, _ = r_next_seq.shape
        if self.T < L:
            raise ValueError(
                f"Window length L={L} longer than trajectory length T={self.T}."
            )

        # windows start at t = 0, step, 2*step, ..., with t+L-1 < T
        self.windows_per_traj = (self.T - L) // step + 1
        self.total_windows = self.N_traj * self.windows_per_traj

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        traj_idx = idx // self.windows_per_traj
        win_idx  = idx %  self.windows_per_traj

        t_start = win_idx * self.step
        t_end   = t_start + self.L

        r_win = self.r_next_seq[traj_idx, t_start:t_end, :]   # [L, idim]
        c_win = self.c_seq[traj_idx,  t_start:t_end, :]       # [L, cdim]
        w_win = self.w_seq[traj_idx, t_start:t_end, :]
        return r_win, c_win, w_win


"""
Functions to build certain conditioning variable sets and scalers
"""

def build_conditioning_and_scalers(
    q: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    system_type: str = None,
    cond_type: str = None,
    step: int = 1,
    parameters: dict = None,
    scale: float = 1.0,
):
    """
    Build (r_next_all, c_all) and fit scalers.

    Args
    ----
    q, v, r : [n_traj, T, 3]
        Fine trajectories. If step > 1, `r` is ignored and r_cg is computed.
    cond_type : {"piri", "piririm", "pipimri"}
    step : int
        - step == 1:    use original r (fine auxiliary variable)
        - step  > 1:    compute r_cg from (q, v), and work on that grid
    dt, Gamma, mass, minimaDist, kconstants, scale:
        Required for step > 1 (to compute r_cg).

    Returns
    -------
    r_next_all : [N, 3]
    c_all      : [N, cdim]
    scaler_r   : StandardScaler for r_next_all
    scaler_c   : StandardScaler for c_all
    """
    assert q.shape == v.shape == r.shape
    assert q.ndim == 3 and q.shape[-1] == 3
    assert step >= 1
    
    system_type = system_type.lower()
    if system_type not in {"bistable", "dimer"}:
        raise ValueError(f"Unknown system_type='{system_type}', expected 'bistable' or 'dimer'.")

    if system_type =="dimer":
        
        if step != 1:
            raise ValueError(
                "Coarse-grained trajectory not implemented for dimer. Use step=1"
            )
        
        n_traj, T2, _ = q.shape
        if T2 % 2 != 0:
            raise ValueError("For 'dimer', time dimension must be even (alternating p1/p2 samples).")
        T = T2 // 2

        r_next, c = construct_rc_dimer(q, v, r, cond_type)

    elif system_type =="bistable":

        # ---- decide which r to use: fine vs coarse ----
        if step == 1:
            assert r is not None, "r must be provided for step=1"
            q_eff, v_eff, r_eff = q, v, r
        else:
            # compute r_cg from fine trajectories, use that as 'r_eff'

            q_eff, v_eff, r_eff = compute_r_cg_from_fine(
                q, v,
                k=step,
                parameters=parameters
            )
            # Note: q_eff, v_eff, r_eff have shape [n_traj, T-step, 3].
            # We now treat them as our "new trajectories" with one logical step per coarse interval.
            r_next, c = construct_rc_bistable(q_eff, v_eff, r_eff, cond_type)

    #plot_binned_mean_var(r_next[..., 3:6]-r_next[..., :3], dx, "r_rel_local", nbins=40)
    #plot_binned_mean_var(r_next[..., :3]+r_next[..., 3:6], dx, "r_com_local", nbins=40)
            
    # ---------- FLATTEN + SCALERS (COMMON) ----------
    print(r_next.shape, c.shape)
    r_next_all = r_next.reshape(-1, r_next.shape[-1])
    c_all      = c.reshape(-1, c.shape[-1])

    scaler_c = StandardScaler().fit(c_all.cpu().numpy())
    scaler_r = StandardScaler().fit(r_next_all.cpu().numpy())

    return r_next_all, c_all, scaler_r, scaler_c

def split_particles(x):
    # x1: particle 1, x2: particle 2
    return x[:, 0::2, :], x[:, 1::2, :]

def construct_rc_dimer(q, v, r, cond_type):
    """
    Construct r_next and c for the dimer system, based on the specified conditioning type.
    """

    q1, q2 = split_particles(q)
    v1, v2 = split_particles(v)
    r1, r2 = split_particles(r)

    if cond_type == "pipimdqidpiririm":
        r1_next = r1[:, 2:, :]
        r2_next = r2[:, 2:, :]
        r_next  = torch.cat([r1_next, r2_next], dim=-1)  # [..., 6]

        v1_n = v1[:, 1:-1, :]
        v2_n = v2[:, 1:-1, :]
        r1_n = r1[:, 1:-1, :]
        r2_n = r2[:, 1:-1, :]
        
        v1_prev = v1[:, :-2, :]
        v2_prev = v2[:, :-2, :]
        r1_prev = r1[:, :-2, :]
        r2_prev = r2[:, :-2, :]
        
        q1_n = q1[:, 1:-1, :]
        q2_n = q2[:, 1:-1, :]
        
        delta_x, delta_vx = compute_dx_dvx(
            q1_n, q2_n, v1_n, v2_n, boundary_type='periodic', boxsize=5.0, keepdim=True
        )
        
        c = torch.cat([delta_x, delta_vx, v1_n, v2_n, v1_prev, v2_prev, r1_n, r2_n, r1_prev, r2_prev], dim=-1)

    elif cond_type == "local_pipimririm":
        r1_next = r1[:, 2:, :]
        r2_next = r2[:, 2:, :]
        r_next  = torch.cat([r1_next, r2_next], dim=-1)  # [..., 6]

        v1_n = v1[:, 1:-1, :]
        v2_n = v2[:, 1:-1, :]
        r1_n = r1[:, 1:-1, :]
        r2_n = r2[:, 1:-1, :]
        
        v1_prev = v1[:, :-2, :]
        v2_prev = v2[:, :-2, :]
        r1_prev = r1[:, :-2, :]
        r2_prev = r2[:, :-2, :]

        c = torch.cat([v1_n, v2_n, v1_prev, v2_prev, r1_n, r2_n, r1_prev, r2_prev], dim=-1)
        
        q1_n = q1[:, 1:-1, :]
        q2_n = q2[:, 1:-1, :]
        R_n, dx = build_local_frame(q1_n, q2_n)
        
        r_next = to_local(R_n, r_next)
        c = to_local(R_n, c)
        
    elif cond_type == "local_dqipipimririm":
        r1_next = r1[:, 2:, :]
        r2_next = r2[:, 2:, :]
        r_next  = torch.cat([r1_next, r2_next], dim=-1)  # [..., 6]

        v1_n = v1[:, 1:-1, :]
        v2_n = v2[:, 1:-1, :]
        r1_n = r1[:, 1:-1, :]
        r2_n = r2[:, 1:-1, :]
        
        v1_prev = v1[:, :-2, :]
        v2_prev = v2[:, :-2, :]
        r1_prev = r1[:, :-2, :]
        r2_prev = r2[:, :-2, :]
        
        q1_n = q1[:, 1:-1, :]
        q2_n = q2[:, 1:-1, :]
        
        delta_x, _ = compute_dx_dvx(
            q1_n, q2_n, v1_n, v2_n, boundary_type='periodic', boxsize=5.0, keepdim=True
        )
        
        
        c = torch.cat([delta_x, v1_n, v2_n, v1_prev, v2_prev, r1_n, r2_n, r1_prev, r2_prev], dim=-1)
        
        q1_n = q1[:, 1:-1, :]
        q2_n = q2[:, 1:-1, :]
        R_n, dx = build_local_frame(q1_n, q2_n)
        
        r_next = to_local(R_n, r_next)
        c = torch.cat((delta_x, to_local(R_n, c[..., 1:])), dim=-1)
        
    elif cond_type == "local_dqidpipipimririm":
        r1_next = r1[:, 2:, :]
        r2_next = r2[:, 2:, :]
        r_next  = torch.cat([r1_next, r2_next], dim=-1)  # [..., 6]

        v1_n = v1[:, 1:-1, :]
        v2_n = v2[:, 1:-1, :]
        r1_n = r1[:, 1:-1, :]
        r2_n = r2[:, 1:-1, :]
        
        v1_prev = v1[:, :-2, :]
        v2_prev = v2[:, :-2, :]
        r1_prev = r1[:, :-2, :]
        r2_prev = r2[:, :-2, :]
        
        q1_n = q1[:, 1:-1, :]
        q2_n = q2[:, 1:-1, :]
        
        delta_x, delta_vx = compute_dx_dvx(
            q1_n, q2_n, v1_n, v2_n, boundary_type='periodic', boxsize=5.0, keepdim=True
        )
        
        c = torch.cat([delta_x, delta_vx, v1_n, v2_n, v1_prev, v2_prev, r1_n, r2_n, r1_prev, r2_prev], dim=-1)

        R_n, dx = build_local_frame(q1_n, q2_n)
        
        r_next = to_local(R_n, r_next)
        c = torch.cat((delta_x, delta_vx, to_local(R_n, c[..., 2:])), dim=-1)
    else:
        raise ValueError(
            f"Unknown conditioning type: {cond_type} for 'dimer'. "
        )

    return r_next, c

def construct_rc_bistable(q_eff, v_eff, r_eff, cond_type):
    # ---- build one-step pairs on the effective grid ----
    if cond_type == "piri":
        # n = 0 .. T_eff-2
        r_next = r_eff[:, 1:, :]       # r_{n+1}
        v_n    = v_eff[:, :-1, :]      # v_n
        r_n    = r_eff[:, :-1, :]      # r_n
        c = torch.cat([v_n, r_n], dim=-1)

    elif cond_type == "piririm":
        # n = 1 .. T_eff-2
        r_next = r_eff[:, 2:, :]       # r_{n+1}
        v_n    = v_eff[:, 1:-1, :]     # v_n
        r_n    = r_eff[:, 1:-1, :]     # r_n
        r_prev = r_eff[:, :-2, :]      # r_{n-1}
        c = torch.cat([v_n, r_n, r_prev], dim=-1)

    elif cond_type == "piririmrimm":
        # v_n, r_n, r_{n-1}, r_{n-2} -> predict r_{n+1}
        # valid n indices: 2 .. T_eff-2  (so that n+1 exists and n-2 exists)

        r_next = r_eff[:, 3:, :]        # r_{n+1}
        v_n    = v_eff[:, 2:-1, :]      # v_n
        r_n    = r_eff[:, 2:-1, :]      # r_n
        r_prev1 = r_eff[:, 1:-2, :]     # r_{n-1}
        r_prev2 = r_eff[:, :-3, :]      # r_{n-2}

        c = torch.cat([v_n, r_n, r_prev1, r_prev2], dim=-1)

    elif cond_type == "pipimri":
        # n = 1 .. T_eff-2
        r_next = r_eff[:, 2:, :]       # r_{n+1}
        v_n    = v_eff[:, 1:-1, :]     # v_n
        r_n    = r_eff[:, 1:-1, :]     # r_n
        v_prev = v_eff[:, :-2, :]      # v_{n-1}
        c = torch.cat([v_n, v_prev, r_n], dim=-1)

    elif cond_type == "pipimririm":
        # n = 1 .. T_eff-2
        r_next = r_eff[:, 2:, :]       # r_{n+1}
        v_n    = v_eff[:, 1:-1, :]     # v_n
        r_n    = r_eff[:, 1:-1, :]     # r_n
        v_prev = v_eff[:, :-2, :]      # v_{n-1}
        r_prev = r_eff[:, :-2, :]
        
        c = torch.cat([v_n, v_prev, r_n, r_prev], dim=-1)
    else:
        raise ValueError(
            f"Unknown conditioning type: {cond_type}. "
            "Expected one of {'piri', 'piririm', 'piririmrimm', 'pipimri', 'pipimririm'}."
        )

    return r_next, c

def make_train_val_ds(r_next_norm, c_norm, weights, n_timesteps, n_datasets, L, systemType, conditionedOn):
    """
    Creates training and validation datasets from normalized data.

    Parameters:
    r_next_norm (Tensor): Normalized next state samples.
    c_norm (Tensor): Normalized conditioning data.
    weights (Tensor): Weights for the samples.
    n_datasets (int): Number of datasets to create.
    L (int): Length of sequences for the dataset.
    systemType (str): Type of the system (currently only "dimer" is supported).
    conditionedOn (str): Conditioning type for the dataset.

    Returns:
    tuple: Training and validation datasets.
    """
    # N_flat total samples

    N = len(r_next_norm)
    assert N % n_datasets == 0, "r_next_norm must be divisible by n_datasets"

    # --- hyper-params ---
    stride = 1          # set to 2, 3, 5, ... to subsample in time
            
    if systemType == "dimer":
        # full time length per trajectory in the *original* dataset divided by 2
        T_full = n_timesteps
        
        # effective number of r_{n+1} steps per trajectory (before stride)
        if conditionedOn in ('piri'):
            T_eff = T_full - 1    # r_{n+1} exists for n = 0..T_full-2
        elif conditionedOn in ("local_pipimririm", "local_dqipipimririm", "local_dqidpipipimririm"):
            T_eff = T_full - 2         # you lose an extra step for r_{n-1}
        else:
            raise ValueError(f"Unknown conditioning: {conditionedOn}")
    else:
        raise ValueError('only dimer system')

    # reshape flat arrays → [n_datasets, T_eff, ...]
    r_next_traj = r_next_norm.view(n_datasets, T_eff, -1)
    c_traj      = c_norm.view(n_datasets, T_eff, -1)
    w_traj      = weights.view(n_datasets, T_eff, -1)

    # Sliding-window dataset: sequences of length L
    split_traj = int(0.8 * n_datasets)

    train_ds = RVSeqDataset(
        r_next_traj[:split_traj],   # [N_train, T_eff_strided, 3]
        c_traj[:split_traj],        # [N_train, T_eff_strided, cdim]
        w_traj[:split_traj],
        L=L,
        step=stride
    )
    val_ds = RVSeqDataset(
        r_next_traj[split_traj:],
        c_traj[split_traj:],
        w_traj[split_traj:],
        L=L,
        step=stride
    )
    
    return train_ds, val_ds



# Not used for now (will clean up later)

def apply_periodic(q, boxsize):
    """
    Apply periodic boundary conditions to positions q.

    q       : [..., 3] (torch.Tensor)
    boxsize : float or array-like of length 3
              box spans [-L/2, L/2] in each dimension
    """
    if not torch.is_tensor(q):
        q = torch.as_tensor(q, dtype=torch.float32)

    if not torch.is_tensor(boxsize):
        boxsize = torch.as_tensor(boxsize, dtype=q.dtype, device=q.device)

    if boxsize.ndim == 0:
        boxsize = boxsize.expand(3)

    # Map to [0, L), then shift back to [-L/2, L/2)
    q_shifted = q + boxsize / 2.0
    q_wrapped = torch.remainder(q_shifted, boxsize)
    q_periodic = q_wrapped - boxsize / 2.0
    return q_periodic


def bistable_force(q, minimaDist, kconstants, scale=1.0):
    """
    Vectorised bistable force for positions q.

    q         : [..., 3]  (any leading batch dims, torch.Tensor)
    minimaDist: float
    kconstants: (3,) array-like or tensor [kx, ky, kz]
    scale     : float

    Returns:
        force: [..., 3]
    """
    if not torch.is_tensor(q):
        q = torch.as_tensor(q, dtype=torch.float32)

    kx, ky, kz = kconstants
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]

    force = torch.zeros_like(q)
    force[..., 0] = - kx * 4 * x * (x**2 - minimaDist**2) / (minimaDist**4)
    force[..., 1] = - ky * 2 * y
    force[..., 2] = - kz * 2 * z

    return scale * force


def aboba_deterministic_step(q_n, v_n, dt_eff, Gamma, mass,
                             minimaDist, kconstants, boxsize, scale=1.0):
    """
    Deterministic ABOBA step (no noise) for one time step dt_eff.

    q_n, v_n : [..., 3]
    dt_eff   : float (effective dt = step * dt)
    Gamma    : friction scalar
    mass     : mass scalar
    """
    # A: first half-step in position
    q_half = q_n + v_n * (dt_eff / 2.0)
    #q_half = apply_periodic(q_half, boxsize)

    # Force at x^{n+1/2}
    F_half = bistable_force(q_half, minimaDist, kconstants, scale=scale)

    # BOB step without noise:
    # expterm = exp(-Gamma * dt / mass)
    # frictionForceTerm = v_n * expterm + (1 + expterm) * F * dt/(2m)
    expterm = torch.exp(torch.tensor(-Gamma * dt_eff / mass,
                                     dtype=q_n.dtype, device=q_n.device))

    v_det = v_n * expterm + (1.0 + expterm) * F_half * (dt_eff / (2.0 * mass))

    # A: second half-step in position
    q_next = q_half + v_det * (dt_eff / 2.0)
    q_next = apply_periodic(q_next, boxsize)

    return q_next, v_det

def compute_r_cg_from_fine(q, v, k, parameters,
                           boxsize=5.0, scale=1.0):
    """
    Compute coarse-grained interaction noise r_cg from fine benchmark trajectories,
    using the same ABOBA scheme as langevinNoiseSampler but *without* noise.

    q, v : [n_traj, T, 3]  (fine dt)
    dt   : fine timestep
    k : integer k (dt_eff = k * dt)

    Gamma, mass       : friction and mass
    minimaDist        : bistable parameter
    kconstants        : (3,) for bistable (kx, ky, kz)
    boxsize           : scalar or (3,) – periodic box, interval [-L/2, L/2]
    scale             : scale factor for the potential

    Returns:
        q_next, v_next, r_next  : [n_traj, T-step, 3]

            v_next = v_det(q_n, v_n; dt_eff) + r_next
    """
    assert q.shape == v.shape
    assert q.ndim == 3 and q.shape[-1] == 3, "q, v must be [n_traj, T, 3]"
    assert k >= 1
    
    dt = parameters['dt']
    Gamma = parameters['Gamma']
    mass = parameters['mass']

    n_traj, T, _ = q.shape
    
    T_cg = T - k
    if T_cg <= 0:
        raise ValueError(f"Not enough timesteps T={T} for step={k}.")

    dt_eff = k * dt

    # fine start states for every jump
    q_start    = q[:, :T_cg, :]    # [n_traj, T-step, 3]
    v_start    = v[:, :T_cg, :]    # [n_traj, T-step, 3]

    # --- fine end velocities for each jump ---
    v_end = v[:, k:, :]                # v_{n+k}


    # deterministic ABOBA step with dt_eff
    _, v_det_end = aboba_deterministic_step(
        q_start, v_start,
        dt_eff=dt_eff,
        Gamma=Gamma,
        mass=mass,
        minimaDist=minimaDist,
        kconstants=kconstants,
        boxsize=boxsize,
        scale=scale,
    )

    # noise that produces the END velocity
    r_end = v_end - v_det_end          # r_{n+k} in end-aligned convention

    # --- now restrict to the coarse grid: n = 0, k, 2k, ... ---
    # end indices: k, 2k, 3k, ...
    q_eff = q[:, k::k, :]
    v_eff = v[:, k::k, :]
    
    # r_end[n] corresponds to jump from n -> n+k, i.e. lands on (n+k)
    # so pick start indices 0, k, 2k, ...
    r_eff = r_end[:, 0::k, :]

    # lengths match by construction
    assert q_eff.shape == v_eff.shape == r_eff.shape
    print('Trajectories tensor shape:', tuple(q_eff.shape))

    return q_eff, v_eff, r_eff
