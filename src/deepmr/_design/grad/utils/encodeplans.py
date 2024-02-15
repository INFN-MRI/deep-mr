"""Phase encoding planning routines."""

__all__ = ["prep_1d_phase_plan", "prep_2d_phase_plan", "prep_3d_phase_plan"]

import warnings
import numpy as np

from .. import utils

from .mask import *
from .misc import pad

gamma_bar = 42.575 * 1e6 # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar # rad / T / us -> rad / T / s


def prep_1d_phase_plan(fov, shape, accel, ordering, **kwargs):
    """
    """
    # parse parameters
    gmax, smax, gdt, rew_derate, fid, acs_shape, _, _ = utils.get_cartesian_defaults(kwargs)

    # parse shape and acceleration
    nz = shape
    Rz, pf = accel
    
    # design phase encoding gradients with maximum amplitude
    dz = fov * 1e-3 / nz # m
    kzmax = np.pi / dz
    
    # actual design
    if gmax is not None:
        gpre, _ = utils.make_trapezoid(-kzmax, 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
    else:
        gpre = None
    
    # design scaling
    if gmax is not None:
        if nz % 2 == 0:
            scaling = np.linspace(-nz // 2, nz // 2 - 1, nz)
        else:
            scaling = np.linspace(-nz // 2 + 1, nz // 2, nz)
        scaling = scaling / nz
        
    # simple k-space acquisition
    mask = utils.parallel_imaging(2, nz, int(Rz), acs_shape)
    if pf != 1.0:
        mask *= utils.partial_fourier(nz, pf)
        
    # define scaling
    kz = np.argwhere(mask)[:, 0]
        
    if gmax is not None:
        scaling = scaling[kz]
    else:
        scaling = None
                            
    return {"mask": mask, "kz": kz}, gpre, scaling


def prep_2d_phase_plan(fov, shape, accel, ordering, **kwargs):
    """
    """
    # parse parameters
    gmax, smax, gdt, rew_derate, fid, acs_shape, _, shift = utils.get_cartesian_defaults(kwargs)

    # parse shape and acceleration
    ny, nframes = shape
    Ry, pf = accel
    
    # design phase encoding gradients with maximum amplitude
    dy = fov * 1e-3 / ny # m
    kymax = np.pi / dy
    
    # actual design
    if gmax is not None:
        gpre, _ = utils.make_trapezoid(-kymax, 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
    else:
        gpre = None
    
    # design scaling
    if gmax is not None:
        if ny % 2 == 0:
            scaling = np.linspace(-ny // 2, ny // 2 - 1, ny)
        else:
            scaling = np.linspace(-ny // 2 + 1, ny // 2, ny)
        scaling = scaling / ny
        
    # simple k-space acquisition
    if nframes == 1:
        mask = utils.parallel_imaging(2, ny, int(Ry), acs_shape)
        if pf != 1.0:
            mask *= utils.partial_fourier(ny, pf)
            
        # define scaling
        kt, ky = None, np.argwhere(mask)[:, 0]
            
    else: # k-t acquisition
        mask = utils.parallel_imaging(3, [nframes, ny], [1, int(Ry)], acs_shape, shift, crop_corner=False)     
        if pf != 1.0:
            mask *= utils.partial_fourier(ny, pf)
        
        # define scaling
        if ordering == "interleaved":
            idx = _get_indexes(mask, order="F")
            kt, ky = idx[0], idx[1]
        elif ordering == "sequential":
            idx = _get_indexes(mask, order="C")
            kt, ky = idx[0], idx[1]
        else:
            raise RuntimeError(f"ordering = {ordering} not recognized - set either 'sequential' or 'interleaved'")
    
    # squeeze
    ky = ky.squeeze()
    
    if gmax is not None:
        scaling = scaling[ky]
    else:
        scaling = None
        
    return {"mask": mask, "k": ky[:, None], "kt": kt}, gpre, scaling


def prep_3d_phase_plan(fov, shape, accel, accel_type, ordering, **kwargs):
    """
    """
    # parse parameters
    gmax, smax, gdt, rew_derate, fid, acs_shape, _, shift = utils.get_cartesian_defaults(kwargs)

    # parse shape and acceleration
    ny, nz, nframes = shape
    Ry, Rz, pf = accel
    
    # check for acceleration type
    assert accel_type in ["PI", "CS"], f"Acceleration type = {accel_type} not recognized. Choose either 'PI' or 'CS'."
    if nframes > 1 and accel_type == "PI":
        warnings.warn("Acceleration type was set to 'PI', but nframes is > 1. We only support 'CS' for 3D kt acquisition for now. Switching to 'CS'")
        accel_type = "CS"
        
    # design phase encoding gradients with maximum amplitude
    dy = fov[0] * 1e-3 / ny # m
    kymax = np.pi / dy
    dz = fov[1] * 1e-3 / nz # m
    kzmax = np.pi / dz
    
    # actual design
    if gmax is not None:
        gprey, _ = utils.make_trapezoid(-kymax, 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
        gprez, _ = utils.make_trapezoid(-kzmax, 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
        
        # get length of rewinder
        gpre_length = max(gprey.shape[-1], gprez.shape[-1])
        
        # pad rewinder
        gprey = pad(gprey, gpre_length, side="before")
        gprez = pad(gprez, gpre_length, side="before")
                    
        # build rewinder and append to gradient
        gpre = np.stack((gprey, gprez), axis=0)
    else:
        gpre = [None, None]
    
    # design scaling
    if gmax is not None:
        if ny % 2 == 0:
            yscaling = np.linspace(-ny // 2, ny // 2 - 1, ny)
        else:
            yscaling = np.linspace(-ny // 2 + 1, ny // 2, ny)
        yscaling = yscaling / ny
        if nz % 2 == 0:
            zscaling = np.linspace(-nz // 2, nz // 2 - 1, nz)
        else:
            zscaling = np.linspace(-nz // 2 + 1, nz // 2, nz)
        zscaling = zscaling / nz
    
    # prepare mask
    if accel_type == "PI":
        mask = utils.parallel_imaging(3, [ny, nz], [int(Ry), int(Rz)], acs_shape, shift, crop_corner=True)
        mask = mask.transpose() # [nz, ny]
    else:
        if Ry*Rz > 1:
            mask = []
            for f in range(nframes):
                tmp, _ = utils.poisson_disk([ny, nz, 1], Ry*Rz, acs_shape, crop_corner=True, seed=f)
                mask.append(tmp)
            mask = np.stack(mask, axis=0)
        else:
            mask = np.ones([nframes, nz, ny], dtype=bool)
        
    # getting indexes
    if accel_type == "PI":
        # define scaling
        if ordering == "interleaved":
            if pf != 1.0:
                mask *= utils.partial_fourier(ny, pf)
            idx = _get_indexes(mask, order="F")
            kt, kz, ky = None, idx[0], idx[1]
        elif ordering == "sequential":
            if pf != 1.0:
                mask *= utils.partial_fourier(nz, pf)[:, None]
            idx = _get_indexes(mask, order="C")
            kt, kz, ky = None, idx[0], idx[1]
        else:
            raise RuntimeError(f"ordering = {ordering} not recognized - set either 'sequential' or 'interleaved'")
        # adapt
        mask = mask[None, ...]
    else: 
        if pf != 1.0:
            mask *= utils.partial_fourier(nz, pf)[:, None]

        idx = _get_indexes(mask, order="F")
        kt, kz, ky = idx[0], idx[1], idx[2]
        
    # assemble trajectory
    ky = ky.squeeze()
    kz = kz.squeeze()
    k = np.stack((ky, kz), axis=-1)
   
    if gmax is not None:
        scaling = np.stack((yscaling[ky], zscaling[kz]), axis=0)
    else:
        scaling = None
                            
    return  {"mask": mask, "k": k, "kt": kt}, gpre, scaling

def _get_indexes(mask, order):
    shape = mask.shape
    idx = np.argwhere(mask.flatten(order))
    return np.unravel_index(idx, shape, order)
