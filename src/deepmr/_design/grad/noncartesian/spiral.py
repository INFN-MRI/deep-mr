"""2D Spiral trajectory design routine."""

__all__ = ["spiral"]

import copy

import numpy as np
from scipy import interpolate

from .. import utils

gamma_bar = 42.575 * 1e6 # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar # rad / T / s

def spiral(fov, shape, accel=1, nintl=1, **kwargs):
    r"""
    Design a dual or constant density spiral.

    Args:
        fov (float): field of view in [mm].
        shape (tuple of ints): matrix size (plane, echoes=1, frames=1).
        accel (int): in-plane acceleration. Ranges from 1 (fully sampled) to nintl.
        nintl (int): number of interleaves to fully sample a plane. For dual density,
            inner spiral is single shot.
    
    Kwargs:
        tilt_type (str): tilt of the shots.
        tilt (bool): if True, keep rotating the spiral through echo train. If false, keep same arm for each echo (defaults to False).
        acs_shape (int): matrix size for inner spiral (defaults to None).
        acs_nintl (int): number of interleaves to fully sample inner spiral. Defaults to 1.
        trans_dur (float): duration (in units of kr / kmax) of transition region beteween inner and outer spiral.
        variant (str): type of spiral. Allowed values are
                - 'center-out': starts at the center of k-space and ends at the edge (default).
                - 'reverse': starts at the edge of k-space and ends at the center.
                - 'in-out': starts at the edge of k-space and ends on the opposite side (two 180° rotated arms back-to-back).
        gdt (float): trajectory sampling rate in [us].
        gmax (float): maximum gradient amplitude in [mT / m].
        smax (float): maximum slew rate in [T / m / s].
        rew_derate (float): derate slew rate during rewinder and z phase-encode blip by this factor, to reduce PNS. Default: 0.1.
        fid (tuple of ints): number of fid points before and after readout (defaults to (0, 0)).

    Returns:
        (dict): structure containing info for reconstruction (coordinates, dcf, matrix, timing...).
        (dict): structure containing info for acquisition (gradient waveforms...).

    Notes:
        The following values are accepted for the tilt name, with :math:`N` the number of
        partitions:

        - "uniform": uniform tilt: 2:math:`\pi / N`
        - "inverted": inverted tilt :math:`\pi/N + \pi`
        - "golden": golden angle tilt :math:`\pi(\sqrt{5}-1)/2`. For 3D, refers to through plane axis (in-plane is uniform).
        - "tiny-golden": tiny golden angle tilt 2:math:`\pi(15 -\sqrt{5})`. For 3D, refers to through plane axis (in-plane is uniform).
        - "tgas": tiny golden angle tilt with shuffling along through-plane axis (3D only)`

    """
    # parse defaults
    gmax, smax, gdt, rew_derate, fid, acs_shape, moco_shape = utils.get_noncart_defaults(kwargs)
    
    # build base interleaf and rotation angles
    kr, phi, shape, acs_shape, variant = _make_spiral_interleave(fov, shape, accel, nintl, kwargs)
    
    # get nframes and nechoes for clarity
    mtx, nechoes, nframes = shape[0], shape[1], shape[2]
    
    # optionally, enforce system constraint (and design base interleaf waveform)
    kr, grad, adc, _ = utils.make_arbitrary_gradient(kr, gmax, smax, gdt, rew_derate, balance=True)
    
    # post-process variant
    kr, grad, adc, echo_idx = _postprocess_spiral(variant, kr, grad, adc, fid)
    
    # compute density compensation factor
    dcf = utils.voronoi(kr.T, nintl)
    
    # compute timing
    if grad is None:
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, kr, grad)
    else:
        if variant == "center-out":
            gtmp = np.concatenate((grad["read"], grad["rew"]), axis=-1)
        elif variant == "reverse":
            gtmp = np.concatenate((grad["pre"], grad["read"]), axis=-1)
        elif variant == "in-out":
            gtmp = np.concatenate((grad["pre"], grad["read"], grad["rew"]), axis=-1)
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, kr, gtmp)
    
    # compute loop indexes (kt, kecho)
    kt, phi = utils.broadcast_tilt(np.arange(nframes), phi, loop_order="new-first")
    kecho, phi = utils.broadcast_tilt(np.arange(nechoes), phi, loop_order="old-first")
            
    # prepare acs
    acs = utils.extract_acs(kr, dcf, shape, acs_shape)
    
    # prepare moco
    moco = utils.extract_acs(kr, dcf, shape, moco_shape)
    
    # prepare grad struct
    if grad is None:
        grad = {"read": None, "rew": None, "pre": None, "rot": None}
    else:
        grad["rot"] = phi
    
    # prepare compressed structure
    R = utils.angleaxis2rotmat(phi, [0, 0, 1])
    compressed = {"kr": utils.scale_traj(kr), "kecho": kecho, "kt": kt, "rot": R, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": copy.deepcopy(acs), "moco": copy.deepcopy(moco)}
    
    # expand
    kr = utils.projection(kr, R).astype(np.float32).transpose(1, 2, 0)
    acs["kr"] = utils.projection(acs["kr"], R).astype(np.float32).transpose(1, 2, 0)
    moco["kr"] = utils.projection(moco["kr"], R).astype(np.float32).transpose(1, 2, 0)
  
    # prepare trajectory structure
    traj = {"kr": utils.scale_traj(kr), "kecho": kecho, "kt": kt, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": acs, "moco": moco, "compressed": compressed}
    if nechoes == 1:
        traj.pop("kecho", None)
        traj["compressed"].pop("kecho", None)
    if nframes == 1:
        traj.pop("kt", None)
        traj["compressed"].pop("kt", None)
    
    # return traj, grad, prot 
    return traj, grad
      
#%% local utils
def _make_spiral_interleave(fov, shape, accel, nintl, kwargs):
    if "tilt_type" in kwargs:
        tilt_type = kwargs["tilt_type"]
    else:
        tilt_type = "uniform"
    if "tilt" in kwargs:
        tilt = kwargs["tilt"]
    else:
        tilt = False      
    if "variant" in kwargs:
        variant = kwargs["variant"]
    else:
        variant = "center-out"
    if "moco_shape" in kwargs:
        moco_shape = kwargs["moco_shape"]
    else:
        moco_shape = None
    if "acs_shape" in kwargs:
        acs_shape = kwargs["acs_shape"]
    else:
        acs_shape = None
    if "acs_nintl" in kwargs:
        acs_nintl = kwargs["acs_nintl"]
    else:
        acs_nintl = 1
    if "trans_dur" in kwargs:
        trans_dur = kwargs["trans_dur"]
    else:
        trans_dur = None
    
    # check variant
    message = (
        f"Error! Unrecognized spiral variant = {variant} - valid types are"
        " 'center-out', 'reverse' and 'in-out'"
    )
    assert variant in ["center-out", "reverse", "in-out"], message

    # handle scalar input
    if np.isscalar(nintl):
        nintl = [nintl]
        
    # shape
    tmp = [None, 1, 1] # (mtx, nechoes, nframes)
    if np.isscalar(shape):
        shape = [shape]
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp
    
    # get resolution
    mtx = [None, None, shape[0]]
    if acs_shape is not None:
        mtx[1] = acs_shape
    if moco_shape is not None:
        mtx[0] = moco_shape
    
    # triple density
    nintl = [1, acs_nintl] + nintl
        
    # clean nintl and mtx
    nintl = [nintl[n] for n in range(len(mtx)) if mtx[n] is not None]
    mtx = [m for m in mtx if m is not None]
    
    # get nframes and nechoes
    nechoes, nframes = shape[1], shape[2]

    # transform to array
    nintl = np.array(nintl)
    mtx = np.array(mtx)

    # cast dimensions
    fov *= 1e-3  # mm -> m
    
    # get resolution
    res = fov / mtx

    # calculate Nyquist sampling
    pr, ptheta = utils.radial_fov(fov, np.min(res))

    # handle in-out case
    if variant == "in-out":
        nintl = nintl * 2
    
    # convert variable number of interleaves to variable fov
    R = nintl / nintl.max()
    nintl = nintl.max()
    fov = fov / R
    
    # check acceleration
    assert accel > 0 and accel <= nintl, f"Acceleration = {accel} must be between positive and <= {nintl}."

    # unpack Nyquist params
    nr, _ = pr
    ntheta, _ = ptheta
    nsamp = nr * ntheta // nintl
    
    # calculate maximum k space radius
    kmax = np.pi / np.min(res)

    # calculate k space radius
    kr = kmax * np.linspace(0.0, 1.0, nsamp)
    dkr = np.diff(kr)[0]
    
    # # process radius
    r = res[1:] / res[0]
    r = np.flip(r)
                
    # build fractional radius
    if len(r) == 0:
        radius = np.asarray([0.0, 1.0])
    else:
        # default transition width
        if trans_dur is None:
            trans_dur = r[0]
            
        # check on transition width
        assert r.max() + trans_dur < 1.0, "Error! Transition band must finish before kmax is reached."
        
        # calculate nodes position
        r = np.repeat(r ,2)
        r[1::2] += trans_dur
        radius = np.concatenate(([0.0], r, [1.0]))
        
    # get radius-dependent fov for each node
    fov = np.repeat(fov ,2)
                     
    # fov interpolation
    interp = interpolate.PchipInterpolator(radius * kmax, fov)
    fov = interp(kr)

    # calculate angular function
    dtheta = dkr * fov / nintl
    theta = np.cumsum(dtheta)

    # build k
    base_k = kr * np.exp(1j * theta)

    # convert complex trajectory to stack
    base_k = utils.traj_complex_to_array(base_k)
    
    # generate angles    
    if variant == "in-out":
        nintl = nintl // 2      
    angles = utils.make_tilt(tilt_type, nintl, accel, nframes, nechoes, tilt)

    return base_k, angles, shape, acs_shape, variant

def _postprocess_spiral(variant, k, grad, adc, fid):
    
    # trajectory
    if variant == "reverse":
        k = np.flip(k, axis=-1)
    elif variant == "in-out":
        k = np.concatenate((-np.flip(k), k), axis=-1)
    
    # gradient
    if grad is not None:
        if variant == "reverse":
            grad["pre"] = np.flip(grad["rew"], axis=-1)
            grad["rew"] = None
            grad["read"] = np.flip(grad["read"], axis=-1)
        elif variant == "in-out":
            grad["pre"] = np.flip(grad["rew"], axis=-1)
            grad["read"] = np.concatenate((-np.flip(grad["read"]), grad["read"]), axis=-1)
        
    # adc
    if adc is not None:
        if variant == "reverse":
            adc = np.flip(adc, axis=-1)
        elif variant == "in-out":
            adc = np.concatenate((np.flip(adc), adc), axis=-1)
    
    # echo index
    if grad is not None:
        if variant == "reverse":
            echo_idx = grad["read"].shape[-1] - 1
        elif variant == "in-out":
            echo_idx = int(grad["read"].shape[-1] // 2)
        else:
            echo_idx = 0
    else:
        if variant == "reverse":
            echo_idx = k.shape[-1] - 1
        elif variant == "in-out":
            echo_idx = int(k.shape[-1] // 2)
        else:
            echo_idx = 0
    
    # add dummies
    if grad is not None:
        grad["read"] = np.pad(grad["read"], ((0, 0), tuple(fid)))
    
    if adc is not None:
        adc = np.pad(adc, tuple(fid))
    
    # update echo_idx
    if grad is not None:
        echo_idx += fid[0]
        
    return k, grad, adc, echo_idx
        
    
        
        
