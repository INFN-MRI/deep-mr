"""3D Stack-Of-Spirals trajectory design routine."""

__all__ = ["spiral_stack"]

import numpy as np

from .spiral import spiral

from .. import utils

def spiral_stack(fov, shape, accel=1, nintl=1, **kwargs):
    r"""
    Design a dual or constant density stack-of-spirals.

    Args:
        fov (tuple of floats): field of view (FOVr, FOVz) in [mm]. If scalar, assume isotropic FOV (FOVz = FOVr).
        shape (tuple of ints): matrix size (plane, nz, echoes=1, frames=1).
        accel (tuple of ints): acceleration (Rplane, Rz, Pf). Ranges from (1, 1, 1) (fully sampled) to (nintl, nz, 0.75).
        nintl (int): number of interleaves to fully sample a plane. For dual density,
            inner spiral is single shot.
    
    Kwargs:
        ordering (str): acquire partitions sequentially ("sequentially") or not ("interleaved") when nframes > 1.
            Default to "interleaved".
        tilt_type (str): tilt of the shots.
        tilt (tuple of bools): if True, keep rotating the spiral through echo train. If false, keep same arm for each echo (defaults to False).
        acs_shape (tuple of ints): matrix size for calibration regions (ACSplane, ACSz). Defaults to (None, None).
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
    # parsing
    fov, shape, accel, kwargs, ordering = utils.config_stack(fov, shape, accel, kwargs)
          
    # prepare phase encoding plan
    act_traj, gpre, plan = utils.prep_1d_phase_plan(fov[1], shape[1], accel[1], ordering, **kwargs[1])
    grew = -np.flip(gpre)
    
    # options
    nslices = act_traj["kz"].shape[0] # actual number of z encodes after accelerations
    
    # rotated vs non-rotated sos
    if kwargs[1]["tilt"]: # rotate during z encodings (spiral caipirinha)
        shape[0][-1] *= nslices
        
    # get in-plane trajectory
    traj, grad = spiral(fov[0], shape[0], accel[0], nintl, **kwargs[0])
    
    # after this I have
    # rot = (nframes * nintl / R,) if tilt = (0, 0)
    # rot = (nechoes * nframes * nintl / R,) if tilt = (1, 0)
    # rot = (nslices * nframes * nintl / R,) if tilt = (0, 1)
    # rot = (nechoes * nslices * nframes * nintl / R,) if tilt = (1, 1)
    # and
    # z = (nslices,) -> I need to expand z accordingly
    
    # put together
    traj, plan = utils.make_stack(ordering, traj, act_traj["kz"], plan, kwargs[1])
    if grad["pre"] is not None and grad["rew"] is not None: # in-out
        grad["pre"], delta_te = utils.compose_gradients("before", gx=grad["pre"][0], gy=grad["pre"][1], gz=gpre, **kwargs[0])
        grad["rew"], _ = utils.compose_gradients("after", gx=grad["rew"][0], gy=grad["rew"][1], gz=grew)
    elif grad["pre"] is not None: # reverse
        grad["pre"], delta_te = utils.compose_gradients("before", gx=grad["pre"][0], gy=grad["pre"][1], gz=gpre, **kwargs[0])
        grad["rew"], _ = utils.compose_gradients("before", gz=grew)
    elif grad["rew"] is not None: # center-out
        grad["pre"], delta_te = utils.compose_gradients("before", gz=gpre, **kwargs[0])    
        grad["rew"], _ = utils.compose_gradients("after", gx=grad["rew"][0], gy=grad["rew"][1], gz=grew)
    grad["amp"] = plan
    
    # add te
    traj["te"] += delta_te
    
    return traj, grad
    

        
    
    

