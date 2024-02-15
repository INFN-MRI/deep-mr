"""2D Rosette trajectory design routine."""

__all__ = ["rosette"]

import copy

import numpy as np

from .. import utils

gamma_bar = 42.575 * 1e6 # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar # rad / T / s

def rosette(fov, shape, esp, accel=1, npetals=None, bending_factor=1, osf=1.0, **kwargs):
    r"""
    Design a rosette trajectory.

    Args:
        fov (float): field of view in [mm].
        shape (tuple of ints): matrix size (plane, echoes=1, frames=1).
        esp (float): echo spacing in [ms]. In real world, must account for hardware and safety limitations.
        accel (int): in-plane acceleration. Ranges from 1 (fully sampled) to nintl.
        npetals (int): number of petals. By default, satisfy Nyquist criterion.
        bending_factor (float): 0 for radial-like trajectory, increase for maximum coverage per shot. In real world, must account for hardware and safety limitations.
        osf (float): radial oversampling factor.
    
    Kwargs:
        tilt_type (str): tilt of the shots.
        tilt (bool): if True, keep rotating the petals through echo train. If false, keep same spoke for each echo (defaults to False).
        acs_shape (int): matrix size for autocalibration (defaults to None).
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
    gmax, smax, gdt, rew_derate, fid, acs_shape, _ = utils.get_noncart_defaults(kwargs)
    
    # build base interleaf and rotation angles
    kr, phi, shape, acs_shape, npetals = _make_rosette_interleave(fov, shape, accel, esp, npetals, bending_factor, osf, kwargs)
    
    # get nframes and nechoes for clarity
    mtx, nechoes, nframes = shape[0], shape[1], shape[2]
    
    # optionally, enforce system constraint (and design base interleaf waveform)
    kr, grad, adc, echo_idx = _make_rosette_gradient(kr, gmax, smax, gdt, rew_derate, fid)
            
    # compute density compensation factor
    dcf = utils.voronoi(kr.T, npetals)
    
    # compute timing
    if grad is not None:
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, kr, grad["read"])
    else:
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, kr, None)

    # compute loop indexes (kt, kecho)
    kt, phi = utils.broadcast_tilt(np.arange(nframes), phi, loop_order="new-first")
    kecho, phi = utils.broadcast_tilt(np.arange(nechoes), phi, loop_order="old-first")
            
    # prepare acs
    acs = utils.extract_acs(kr, dcf, shape, acs_shape)
    
    # prepare grad struct
    if grad is None:
        grad = {"read": None, "rew": None, "pre": None, "rot": None}
    else:
        grad["rot"] = phi
    
    # prepare compressed structure
    R = utils.angleaxis2rotmat(phi, [0, 0, 1])
    compressed = {"kr": utils.scale_traj(kr), "kecho": kecho, "kt": kt, "rot": R, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": copy.deepcopy(acs)}
    
    # expand
    kr = utils.projection(kr, R).astype(np.float32).transpose(1, 2, 0)
    acs["kr"] = utils.projection(acs["kr"], R).astype(np.float32).transpose(1, 2, 0)
    
    # prepare trajectory structure
    traj = {"kr": utils.scale_traj(kr), "kecho": kecho, "kt": kt, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": acs, "compressed": compressed}
    if nechoes == 1:
        traj.pop("kecho", None)
        traj["compressed"].pop("kecho", None)
    if nframes == 1:
        traj.pop("kt", None)
        traj["compressed"].pop("kt", None)

    # prepare protocol header
    # prot = {"kr": utils.scale_traj(kr), "phi": phi, "kecho": kecho, "kt": kt, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": acs_shape}

    # plot reports
    # TODO
    
    # return traj, grad, prot 
    return traj, grad
      
#%% local utils
def _make_rosette_interleave(fov, shape, accel, esp, npetals, bending_factor, osf, kwargs):
    if "tilt_type" in kwargs:
        tilt_type = kwargs["tilt_type"]
    else:
        tilt_type = "uniform"
    if "acs_shape" in kwargs:
        acs_shape = kwargs["acs_shape"]
    else:
        acs_shape = None
            
    # shape
    tmp = [None, 1, 1] # (mtx, nechoes, nframes)
    if np.isscalar(shape):
        shape = [shape]
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp
        
    # get nframes and nechoes
    mtx, nechoes, nframes = shape
    
    # transform to array
    mtx = np.array(mtx)

    # cast dimensions
    fov *= 1e-3  # mm -> m
    
    # get resolution
    res = fov / mtx

    # calculate Nyquist sampling
    pr, ptheta = utils.radial_fov(fov, res)
    
    # unpack Nyquist params
    nr, _ = pr
    ntheta, _ = ptheta

    # calculate frequencies
    w1 = np.pi * 1 / esp
    w2 = bending_factor * w1

    # calculate maximum k space radius
    kmax = np.pi / res

    # calculate trajectory
    t = np.linspace(0, esp, int(osf * nr))
    base_k = kmax * np.sin(w1 * t) * np.exp(1j * w2 * t)

    # convert complex trajectory to stack
    base_k = utils.traj_complex_to_array(base_k)

    # get "Nyquist" number of petals
    if bending_factor <= 1:
        npetals_design = int(ntheta / (1 + 3 * bending_factor**2) ** 0.5)
    else:
        npetals_design = int(ntheta / (3 + bending_factor**2) ** 0.5)
        
    # set default spokes if not provided
    if npetals is None:
        npetals = npetals_design
        
    # generate angles      
    angles = utils.make_tilt(tilt_type, npetals, accel, nframes, nechoes)

    return base_k, angles, shape, acs_shape, npetals

def _make_rosette_gradient(kr, gmax, smax, gdt, rew_derate, fid):
    # build gradient
    kr, grad, adc, _ = utils.make_arbitrary_gradient(kr, gmax, smax, gdt, rew_derate, balance=False)
                
    return kr, grad, adc, np.asarray([0])
        
    
        
        
