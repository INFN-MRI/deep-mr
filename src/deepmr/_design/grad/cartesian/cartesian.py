"""Cartesian trajectory design routine."""

__all__ = ["cartesian2D", "cartesian3D"]

import warnings

import numpy as np

from .. import utils

gamma_bar = 42.575 * 1e6 # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar # rad / T / us -> rad / T / s

def cartesian2D(fov, shape, accel=1, osf=1.0, **kwargs):
    r"""
    Design a 2D (+t) cartesian encoding scheme.

    Args:
        fov (tuple of floats): field of view (FOVx, FOVy) in [mm]. If scalar, assume isotropic FOV (FOVx = FOVy).
        shape (tuple of ints): matrix size (x, y, echoes=1, frames=1).
        accel (tuple of ints): acceleration (Ry, Pf). Ranges from (1, 1) (fully sampled) to (ny, 0.75).
        osf (float): readout oversampling factor (defaults to 1.0)
.    
    Kwargs:
        ordering (str): acquire phase sequentially ("sequentially") or not ("interleaved") when nframes > 1.
            Default to "interleaved".
        echo_type (str): can be "bipolar" or "flyback".
        acs_shape (int): matrix size for calibration regions ACSy. Defaults to None.
        gdt (float): gradient sampling rate in [us].
        gmax (float): maximum gradient amplitude in [mT / m].
        smax (float): maximum slew rate in [T / m / s].
        rew_derate (float): derate slew rate during rewinder and z phase-encode blip by this factor, to reduce PNS. Default: 0.1.
        fid (tuple of ints): number of fid points before and after readout (defaults to (0, 0)).

    Returns:
        (dict): structure containing info for reconstruction (coordinates, dcf, matrix, timing...).
        (dict): structure containing info for acquisition (gradient waveforms...).
        (dict): structure containing info for saving-/loading- (base trajectory, angles...).

    """
    # parsing
    fov, shape, accel, kwargs, ordering = utils.config_cartesian_2D(fov, shape, accel, kwargs)
          
    # prepare phase encoding plan
    act_traj, gpre, plan = utils.prep_2d_phase_plan(fov[1], shape[1], accel, ordering, **kwargs[1])
    
    # get in-plane trajectory
    traj, grad = _make_cartesian_readout(fov[0], shape[0], osf, kwargs[0])
            
    # put together
    traj["mask"] = np.atleast_2d(act_traj["mask"])[:, None, :, None]
    traj["k"] = act_traj["k"]
    traj["kt"] = act_traj["kt"]
    traj, plan = utils.make_multiecho(traj, plan, nechoes=shape[0][1])
    grad["pre"], delta_te = utils.compose_gradients("before", gx=grad["pre"], gy=gpre, **kwargs[0])
    grad["pre"] = grad["pre"][:2, :]
    grad["rew"] = -np.flip(grad["pre"], axis=-1)
    grad["amp"] = plan
    
    # add te
    traj["te"] += delta_te
    
    return traj, grad


def cartesian3D(fov, shape, accel=1, osf=1.0, accel_type="PI", **kwargs):
    r"""
    Design a 3D (+t) cartesian encoding scheme.

    Args:
        fov (tuple of floats): field of view (FOVx, FOVy, FOVz) in [mm]. If scalar, assume isotropic FOV (FOVx = FOVy = FOVz).
        shape (tuple of ints): matrix size (x, y, z, echoes=1, frames=1).
        accel (tuple of ints): acceleration (Ry, Rz, Pf). Ranges from (1, 1, 1) (fully sampled) to (ny, nz, 0.75).
        osf (float): readout oversampling factor (defaults to 1.0)
        accel_type (str): "PI" (regular undersampling) or "CS" (poisson-disk). For nframes > 1, force "CS".
    
    Kwargs:
        ordering (str): acquire partitions sequentially ("sequentially") or not ("interleaved") - only for accel_type="PI".
            Default to "interleaved" (ignored for accel_type="CS").
        echo_type (str): can be "bipolar" or "flyback".
        acs_shape (tuple of ints): matrix size for calibration regions (ACSy, ACSz). Defaults to (None, None).
        gdt (float): trajectory sampling rate in [us].
        gmax (float): maximum gradient amplitude in [mT / m].
        smax (float): maximum slew rate in [T / m / s].
        rew_derate (float): derate slew rate during rewinder and z phase-encode blip by this factor, to reduce PNS. Default: 0.1.
        fid (tuple of ints): number of fid points before and after readout (defaults to (0, 0)).

    Returns:
        (dict): structure containing info for reconstruction (coordinates, dcf, matrix, timing...).
        (dict): structure containing info for acquisition (gradient waveforms...).
        (dict): structure containing info for saving-/loading- (base trajectory, angles...).

    """
    # parsing
    fov, shape, accel, kwargs, ordering = utils.config_cartesian_3D(fov, shape, accel, kwargs)
          
    # prepare phase encoding plan
    act_traj, gpre, plan = utils.prep_3d_phase_plan(fov[1], shape[1], accel, accel_type, ordering, **kwargs[1])
    
    # get in-plane trajectory
    traj, grad = _make_cartesian_readout(fov[0], shape[0], osf, kwargs[0])
                
    # put together
    traj["mask"] = np.atleast_3d(act_traj["mask"])[..., None]
    traj["k"] = act_traj["k"]
    traj["kt"] = act_traj["kt"]
    traj, plan = utils.make_multiecho(traj, plan, nechoes=shape[0][1])
    grad["pre"], delta_te = utils.compose_gradients("before", gx=grad["pre"], gy=gpre[0], gz=gpre[1], **kwargs[0])
    grad["rew"] = -np.flip(grad["pre"], axis=-1)
    grad["amp"] = plan
    
    # add te
    traj["te"] += delta_te
    
    return traj, grad
    

#%%  local utils
def _make_cartesian_readout(fov, shape, osf, kwargs):
    
    # parse parameters
    gmax, smax, gdt, rew_derate, fid, _, flyback, _ = utils.get_cartesian_defaults(kwargs)
        
    # get nframes and nechoes for clarity
    mtx, nechoes = shape
    
    # ignore flyback for nechoes=1
    if nechoes == 1 and flyback:
        warnings.warn("Flyback was requested, but nechoes=1. Ignoring.")
        flyback = False

    # cast dimensions
    fov *= 1e-3  # mm -> m
    
    # get resolution
    res = fov / mtx
    
    # calculate maximum k space radius
    kmax = np.pi / np.min(res)

    # calculate Nyquist sampling
    px, _ = utils.squared_fov(fov, res)

    # unpack Nyquist params
    nx, _ = px
    nsamp = int(osf * nx)
            
    # calculate trajectory
    k = np.linspace(0.0, 2.0, nsamp) - 1

    # rescale
    k *= kmax
    
    # calculate gradients
    grad, adc, echo_idx = utils.make_cartesian_gradient(k, gmax, smax, gdt, rew_derate, fid, flyback)
            
    # compute timing
    if grad is None:
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, k, grad)
    else:
        gtmp = np.concatenate((grad["read"], grad["flybck"]))
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, k, gtmp)
    
    # prepare trajectory structure
    traj = {"t": t, "te": te, "adc": adc}
    
    # prepare structure
    if grad is None:
        grad = {"read": None, "pre": None, "flybck": None}
          
    # return traj, grad 
    return traj, grad
    
        
    
    


