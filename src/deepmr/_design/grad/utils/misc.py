"""Miscellaneous utils."""

__all__ = ["traj_complex_to_array", "traj_array_to_complex", "scale_traj", "pad", "flatten_echoes", "extract_acs"]

import numpy as np

def traj_complex_to_array(k, axis=0):
    """
    Convert complex convention trajectory to (x, y) trajectory.

    Args:
        k (complex array): Nt vector
    """
    return np.stack((k.real, k.imag), axis=axis)


def traj_array_to_complex(k):
    """
    Convert [2, Nt] convention traj to complex convention.

    Args:
        k (complex array): Nt vector
    """
    kout = k[0] + 1j * k[1]
    return kout

def scale_traj(coord, axis=-1):
    """
    Normalize the trajectory to be used by NUFFT operators.

    Args:
        coord (array): The trajectory to normalize, it might be of shape (..., dim).

    Returns:
        (array): The normalized trajectory of shape (..., dim) in -0.5, 0.5.
    
    """
    cabs = (coord**2).sum(axis=axis)**0.5
    cmax = cabs.max()
    return coord / cmax / 2.0
    
def pad(input, length, side="after"):
    """
    Numpy pad wrapper.
    """
    if length <= input.shape[-1]:
        return input
    elif side == "after":
        padsize = length - input.shape[-1]
        npad = [(0, 0)] * input.ndim
        npad[-1] = (0, padsize)
    elif side == "before":
        padsize = length - input.shape[-1]
        npad = [(0, 0)] * input.ndim
        npad[-1] = (padsize, 0)
    return np.pad(input, npad)
    
def flatten_echoes(nechoes, adc, grad):
    """
    Flatten echoes along readout axis.
    """
    if grad is not None:
        nint = int(grad.shape[0] // nechoes)
        npts = grad.shape[-2]
        ndim = grad.shape[-1]
        grad = grad.reshape(nint, nechoes, npts, ndim)
        grad = grad.reshape(nint, -1, ndim)
        grad = np.ascontiguousarray(grad)
        
    if adc is not None:
        adc = np.apply_along_axis(np.tile, 0, adc, nechoes)
        
    return grad, adc

def extract_acs(coord, dcf, shape, acs_shape):
    """
    Get acs coordinates, dcf, indexes and matrix shape.
    """
    if acs_shape is None:
        acs_shape = int(shape[0] // 10)
        
    # compute threshold
    threshold = acs_shape / shape[0]
    
    # get k-space radius
    kr = scale_traj(coord, axis=0)
    kabs = (kr**2).sum(axis=0)**0.5

    # get indexes
    idx = kabs <= 0.5 * threshold
    
    return {"kr": kr[:, idx], "mtx": [acs_shape, acs_shape], "dcf": dcf[idx], "adc": idx}

    
    
