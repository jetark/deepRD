import h5py
import numpy as np
import torch

# Functions to load trajectories and manipulate them

def loadTrajectory(fnamebase, fnumber, fastload = False):
    '''
    Reads data from discrete trajectory and returns a simple np.array of
    integers representing the discrete trajectory. Assumes h5 file.
    :param fnamebase, base of the filename
    :param fnumber, filenumber
    :param fastload if true loads the H5 data, if false it converts the data to numpy.
    this however makes the loading very slow.
    :return: array of arrays representing the trajectory
    '''
    filename = fnamebase + str(fnumber).zfill(4) + '.h5'
    f = h5py.File(filename, 'r')

    # Get the data
    a_group_key = list(f.keys())[0]
    if fastload:
        data = f[a_group_key]
    else:
        data = np.array(f[a_group_key])
        #data = f[a_group_key][:] # equivalent

    return data


def writeTrajectory(traj, fnamebase, fnumber):
    '''
    Write trajectory into h5 file into the filename fnamebase+fnumber
    '''
    filename = fnamebase + str(fnumber).zfill(4) + '.h5'
    f = h5py.File(filename, 'w')
    f.create_dataset('trajectory', data=traj)
    f.close()

def loadDiscreteTrajectory(fnamebase, fnumber, fnamesuffix = '_discrete', filetype = 'h5'):
    '''
    Reads data from discrete trajectory and returns a simple np.array of
    integers representing the discrete trajectory. The file can be in the
    h5 or xyz format.
    :param fnamebase, base of the filename
    :param fnumber, filenumber
    :param fnamesuffix, suffix added at end of filename before the extension
    :param filetype, string indicating which format, h5 or xyz, is the file
    :return: array with integers representing the discrete trajectory
    '''
    if filetype == 'h5':
        filename = fnamebase + str(fnumber).zfill(4) + fnamesuffix + '.h5'
        f = h5py.File(filename, 'r')

        # Get the data
        a_group_key = list(f.keys())[0]
        array = f.get(a_group_key)
        nparray = np.array(array).transpose()[0]

        return nparray

    if filetype == 'xyz':
        filename = fnamebase + str(fnumber).zfill(4) + fnamesuffix + '.xyz'
        file = open(filename, "r")

        # Read file and save to array
        filelines = file.readlines()
        array = np.zeros(len(filelines), dtype = int)
        for i, line in enumerate(filelines):
            array[i] = int(float(line))
        return array



def listIndexSplit(inputList, *args):
    '''
    Function that splits inputList into smaller list by slicing in the indexes given by *args.
    :param inputList:
    :param args: int indexes where list should be splitted (Note to convert a
    list "mylist" into *args just do: *mylist)
    :return: list of sliced lists
    If extra arguments were passed prepend the 0th index and append the final
    # index of the passed list, in order toa v check for oid checking the start
    # and end of args in the loop. Also, add one in args for correct indexing.
    '''
    if args:
        args = (0,) + tuple(data+1 for data in args) + (len(inputList)+1,)
    # Slice list and return list of lists.
    slicedLists = []
    for start, end in zip(args, args[1:]):
        slicedLists.append(inputList[start:end-1])
    if slicedLists == []:
        slicedLists.append(inputList)
    return slicedLists



def splitDiscreteTrajs(discreteTrajs, unboundStateIndex = 0):
    '''
    Splits trajectories into smaller trajectories by cutting out
    all the states unboundStateindex (0)
    :param discreteTrajs: list of discrete trajectories
    :param unboundStateIndex: index of the unbound state used to
    decide where to cut the trajectories, normally we choose it to be
    zero.
    :return: List of sliced trajectories
    '''
    slicedDtrajs = []
    trajnum = 0
    for dtraj in discreteTrajs:
        # Slice trajectory using zeros as reference point
        indexZeros = np.where(dtraj==unboundStateIndex)
        slicedlist = listIndexSplit(dtraj, *indexZeros[0])
        # Remove the empty arrays
        for array in slicedlist:
            if array.size > 1:
                slicedDtrajs.append(array)
        trajnum += 1
        print("Slicing trajectory ", trajnum, " of ", len(discreteTrajs), " done.", end="\r")  
    return slicedDtrajs



def stitchTrajs(slicedDtrajs, minlength = 1000):
    '''
    Joins splitted trajectories into long trajectories of at least minlength. The trajectories that cannot
    be joined are left as they were.
    :param slicedDtrajs: list of discrete trajectories. Each discrete trajectory is a numpy array
    :param minlength: minimum length of stitched trajectory if any stititching is possible
    :return: list of stitched trajectories
    '''
    myslicedDtrajs = slicedDtrajs.copy()
    stitchedTrajs = []
    # Stitch trajectories until original sliced trajectory is empty
    while len(myslicedDtrajs) > 0:
        traj = myslicedDtrajs[0]
        del myslicedDtrajs[0]
        percentageDone = int(100.0*(1-len(myslicedDtrajs)/len(slicedDtrajs.copy())))
        print("Stitching trajectories: ", percentageDone, "% done   ", end="\r")
        # Try to keep all resulting trajectories over a certain length min length
        while traj.size <= minlength:
            foundTrajs = False
            for i in reversed(range(len(myslicedDtrajs))):
                # If end point and start point match, join trajectories
                if traj[-1] == myslicedDtrajs[i][0]:
                    if traj[-1] < 0:
                    	print("Error in stitching   ")
                    if myslicedDtrajs[i][0] < 0:
                    	print("Error in stitching   ")
                    foundTrajs = True
                    traj = np.concatenate([traj, myslicedDtrajs[i]])
                    del myslicedDtrajs[i]
            # If no possible trajectory to join is found, save trajectory and continue.
            if foundTrajs == False:
                break;
        stitchedTrajs.append(traj)   
    return stitchedTrajs

def convert2trajectory(timeArray, variableArrayList):
    '''
    Given a list of arrays, where each array corresponds to the trajectory of that variable, output
    a trajectory of the concatenated arrays as a single trajectory with more components. Note the variableArrayList
    is e.g. a list of array of positions, velocities, etc... each of this arrays stores in the each entry another
    array with the values (position/velocity) of all the particles in the simulation.
    '''
    traj = []
    trajLength = len(timeArray)
    varLength = len(variableArrayList[0][0])
    for i in range(trajLength):
        for j in range(varLength):
            time = np.array([timeArray[i]])
            trajElement = [time]
            for variableArray in variableArrayList:
                trajElement.append(variableArray[i][j])
            traj.append(np.concatenate((trajElement)))
    return np.array(traj)


def extractVariableFromTrajectory(trajs, variableIndex):
    '''
    Extracts the variable with index variableIndex from trajectories into an array. If variableIndex is a list with
    two indexes, extracts the variables corresponding to that range of indexes.
    '''
    variableArray = []
    if np.isscalar(variableIndex):
        for traj in trajs:
            for i in range(len(traj)):
                variableArray.append(traj[i][variableIndex])
    else:
        for traj in trajs:
            for i in range(len(traj)):
                variableArray.append(traj[i][variableIndex[0]:variableIndex[1]])
    return np.array(variableArray)


def calculateMean(trajs, varOrIndex = 'position'):
    '''
    Calculates mean of trajectories, varOrIndex can be 'position' or 'velocity', assuming indexing in each element
    of a trajectory be (t,position,velocity), or it can be an index range, e.g. [1,4] for position
    '''
    if varOrIndex == 'position':
        indexl = 1
        indexr = 4
    elif varOrIndex == 'velocity':
        indexl = 4
        indexr = 7
    elif varOrIndex == 'raux':
        indexl = 8
        indexr = 11
    elif varOrIndex == 'raux2':
        indexl = 11
        indexr = 14
    elif varOrIndex == 'rauxReduced':
        indexl = 7
        indexr = 10
    else:
        indexl = varOrIndex[0]
        indexr = varOrIndex[1]
    dimension = indexr - indexl
    mean = np.zeros(dimension)
    totalSamples = 0
    for traj in trajs:
        for i in range(len(traj)):
            mean += traj[i][indexl:indexr]
        totalSamples += len(traj)
    mean = mean/totalSamples
    return mean

def calculateVariance(trajs, varOrIndex = 'position', mean = None):
    '''
    Calculates variance of trajectories, var can be 'position' or 'velocity', assuming indexing in each element
    of a trajectory be (t,position,velocity), or it can be an index range, e.g. [1,4] for position.
    If mean is not given, it calls calculate mean.
    '''
    if mean.any() == None:
        mean = calculateMean(trajs, varOrIndex)
    if varOrIndex == 'position':
        indexl = 1
        indexr = 4
    elif varOrIndex == 'velocity':
        indexl = 4
        indexr = 7
    elif varOrIndex == 'raux':
        indexl = 8
        indexr = 11
    elif varOrIndex == 'raux2':
        indexl = 11
        indexr = 14
    elif varOrIndex == 'rauxReduced':
        indexl = 7
        indexr = 10
    else:
        indexl = varOrIndex[0]
        indexr = varOrIndex[1]
    dimension = indexr - indexl
    variance = np.zeros(dimension)
    totalSamples = 0
    for traj in trajs:
        for i in range(len(traj)):
            devFromMean = traj[i][indexl:indexr] - mean
            variance += devFromMean*devFromMean
        totalSamples += len(traj)
    variance = variance/totalSamples
    return variance

def calculateStdDev(trajs, varOrIndex = 'position', mean = None):
    variance = calculateVariance(trajs, varOrIndex, mean)
    stddev = np.sqrt(variance)
    return stddev

def calculateAutoCorrelation(trajs, lagtimesteps, stride = 1, var = 'position', mean = None, variance = None):
    '''
    Calculates autocorrelation of trajectories, for a given stride. Variable var can be 'position' or 'velocity',
    assuming indexing in each element of a trajectory be (t,position,velocity). If mean and variance are not given,
    it calls calulate mean and calculate variance.
    '''
    if var == 'position':
        index = 1
    elif var == 'velocity':
        index = 4
    elif var == 'raux':
        index = 8
    elif var == 'raux2':
        index = 11
    elif var == 'rauxReduced':
        index = 7
    if mean.any() == None:
        mean = calculateMean(trajs, var)
    if variance.any() == None:
        variance = calculateVariance(trajs, var, mean)
    totalSamples = 0
    AC = 0.0
    for traj in trajs:
        for i in range(len(traj)-lagtimesteps*stride):
            devFromMean = traj[i][index:index+3] - mean
            devFromMean2 = traj[i + lagtimesteps*stride][index:index+3] - mean
            AC += np.dot(devFromMean, devFromMean2)
        totalSamples += len(traj) - lagtimesteps*stride
    AC = AC/totalSamples
    AC = AC/variance
    return AC

def autoCorrelationFromTimeSeries(timeSeriesData, lagtime=1):
    return np.corrcoef(np.array([timeSeriesData[:-lagtime], timeSeriesData[lagtime:]]))[0,1]

def calculateAutoCorrelationFunction(trajs, lagtimesteps, stride = 1, var = 'position'):
    '''
    Calculates autocorrelation function of trajectories, for a given lagtimesteps (length of time interval in
    timesteps) and stride. Variable var can be 'position' or 'velocity', assuming indexing in each element of a
    trajectory be (t,position,velocity).
    '''
    ACF = []
    mean = calculateMean(trajs, var)
    # Calculate one dimensional variance
    if var == 'position':
        index = 1
    elif var == 'velocity':
        index = 4
    elif var == 'raux':
        index = 8
    elif var == 'raux2':
        index = 11
    elif var == 'rauxReduced':
        index = 7
    variance = 0
    totalSamples = 0
    for traj in trajs:
        for i in range(len(traj)):
            devFromMean = traj[i][index:index+3] - mean
            variance += np.dot(devFromMean,devFromMean)
        totalSamples += len(traj)
    variance = variance/totalSamples
    for lagtime in range(lagtimesteps):
        ACF.append(calculateAutoCorrelation(trajs, lagtime, stride, var, mean, variance))
        print('Computing ACF for', var, ': ', 100*(lagtime+1)/lagtimesteps , '% complete   ', end="\r")
    ACF = np.array(ACF)
    return ACF

def autoCorrelationFunctionFromTimeSeries(timeSeriesData, lagtimesteps, stride = 1):
    '''
    Calculates autocorrelation function of time series data, for a given lagtimesteps (length of time interval in
    timesteps) and stride.
    '''
    ACF = []
    for lagtime in range(lagtimesteps):
        ACF.append(autoCorrelationFromTimeSeries(timeSeriesData, lagtime*stride + 1))
        print('Computing ACF: ', 100*(lagtime+1)/lagtimesteps , '% complete   ', end="\r")
    ACF = np.array(ACF)
    return ACF


def extractParticleTrajectories(trajs, particleIndex, numParticles):
    output_trajs = []
    for traj in trajs:
        particleTraj = traj[particleIndex::numParticles] # Extracts the elements corresponding to the desired particle
        output_trajs.append(particleTraj)
    return np.array(output_trajs)


def relativePosition(pos1, pos2, boundaryType, boxsize):
    if not isinstance(boxsize, (list, tuple, np.ndarray)):
        boxsize = [boxsize]*len(pos1)
    p1periodic = 1.0 * pos1
    if (boundaryType == "periodic" and boxsize != None):
        for i in range(3):
            if (pos2[i] - pos1[i]) > 0.5 * boxsize[i]:
                p1periodic[i] += boxsize[i]
            if (pos2[i] - pos1[i]) < -0.5 * boxsize[i]:
                p1periodic[i] -= boxsize[i]
    return pos2 - p1periodic

def rotateVec(unitvec, vec):
    '''
    Rotates vec to match new frame of reference where unitvec corresponds to the
    direction of the x axis. If unitvec is not unitary it automatically normalizes it.
    '''
    unitvecNorm = np.linalg.norm(unitvec)
    if unitvecNorm != 1:
        unitvec = unitvec/unitvecNorm
    theta = np.arccos(unitvec[2])
    phi = np.arctan2(unitvec[1],unitvec[0])
    vec = rotateZaxis(vec, -1 * phi)
    vec = rotateYaxis(vec, -1 * (theta - np.pi / 2.0))
    return vec

def rotateVecInverse(unitvec, vec):
    '''
    Inverse rotation than that of rotate2vec
    '''
    unitvecNorm = np.linalg.norm(unitvec)
    if unitvecNorm != 1:
        unitvec = unitvec/unitvecNorm
    theta = np.arccos(unitvec[2])
    phi = np.arctan2(unitvec[1],unitvec[0])
    vec = rotateYaxis(vec, theta-np.pi/2.0)
    vec = rotateZaxis(vec, phi)
    return vec

def rotateXaxis(vec, theta):
    rotMatrix = np.array([[1, 0, 0],
                          [np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0]])
    return rotMatrix.dot(vec)

def rotateYaxis(vec, theta):
    rotMatrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    return rotMatrix.dot(vec)

def rotateZaxis(vec, theta):
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0,0,1]])
    return rotMatrix.dot(vec)

### Helper functions for LOCAL FRAME transformation
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


