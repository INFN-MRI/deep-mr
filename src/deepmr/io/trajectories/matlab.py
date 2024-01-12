"""I/O Routines for MATLAB trajectories."""

__all__ = ["read_matfile_traj"]

import numpy as np

from ..generic import matlab
from ..utils.header import Header


def read_matfile_traj(filename, dcfname=None, schedulename=None):
    """
    Read MR trajectory written as a matfile.

    Parameters
    ----------
    filename : str
        Path of the file on disk.

    Returns
    -------
    dict
        Deserialized matfile.
        
    """
    # load dcf
    matfile, filename = matlab.read_matfile(filename, True)
    
    # get k space trajectory
    k = _get_trajectory(matfile)
    ndim = k.shape[-1]
    
    # get indexes
    if "ind" in matfile:
        ind = matfile["ind"][0]
    elif "inds" in matfile:
        ind = matfile["inds"].squeeze()
    else:
        raise RuntimeError("ADC indexes not found!")
    
    # get dcf
    if "dcf" in matfile:
        dcf = matfile["dcf"].reshape(k.shape[:-1])
    else:
        try:
            if dcfname is None:
                dcfname = ".".join([filename.split(".")[0] + "_dcf", "mat"])
            dcf = matlab.read_matfile(dcfname)
            dcf = dcf["dcf"].reshape(k.shape[:-1])
        except Exception:
            dcf = None
            
    # get sampling time
    if "t" in matfile:
        dt = np.diff(matfile["t"][0])[0] * 1e3 # ms
    elif "ts" in matfile:
        dt = np.diff(matfile["ts"].squeeze())[0] * 1e3 # ms
    else:
        raise RuntimeError("Sampling time not found!")
    
    # get matrix
    if "mtx" in matfile:
        shape = [int(matfile["mtx"].squeeze())] * ndim
    elif "npix" in matfile:
        shape = [int(matfile["npix"].squeeze())] * ndim
    else:
        raise RuntimeError("Matrix size not found!")
    
    # get resolution
    if "fov" in matfile:
        fov = [float(matfile["fov"].squeeze()) * 1e3] * ndim # mm
    else:
        raise RuntimeError("Field of View not found!")
    resolution = (np.asarray(fov) / np.asarray(shape)).tolist() 
    
    # if ndim != 3:
    #     shape = [1] + shape
    #     resolution = [1.0] + resolution
        
    # initialize header
    head = Header(shape, resolution, dt=dt)
    head.adc = ind
   
    # get schedule file
    head = _get_schedule(head, matfile, schedulename)
    
    # get ordering
    ncontrasts = len(head.FA)
    nviews = int(k.shape[0] / ncontrasts)  
    
    k = k.reshape(nviews, ncontrasts, -1, ndim).swapaxes(0, 1)
    k = np.ascontiguousarray(k).astype(np.float32)
    if dcf is not None:
        dcf = dcf.reshape(nviews, ncontrasts, -1).swapaxes(0, 1)
        dcf = np.ascontiguousarray(dcf).astype(np.float32)
    
    return k, dcf, head

    
# %% subroutines
def _get_trajectory(matfile):
        
    if "k" in matfile:
        k = matfile["k"]
        # get shape
        if "ind" in matfile:
            npts = matfile["ind"].shape[-1]
        elif "inds" in matfile:
            npts = matfile["inds"].shape[-1]
        else:
            raise RuntimeError("ADC indexes not found!")
        nviews = int(k.shape[1] / npts)
        
        # reshape
        k = k.reshape(nviews, npts, -1)
    elif "ks" in matfile and "phi" in matfile:
        ks = matfile["ks"]
        phi = matfile["phi"].T
        
        # get shape
        npts, nviews = ks.shape[1], phi.shape[1]
        
        # rotate in plane
        sinphi = np.sin(np.deg2rad(phi))
        cosphi = np.cos(np.deg2rad(phi))
        
        if "theta" in matfile:
            theta = matfile["theta"].T
            sintheta = np.sin(np.deg2rad(theta))
            costheta = np.cos(np.deg2rad(theta))
            k = np.stack(
                [ks[..., 0] * costheta * cosphi + ks[..., 1] * sinphi - ks[..., 2] * sintheta * cosphi,
                 -ks[..., 0] * costheta * sinphi + ks[..., 1] * cosphi - ks[..., 2] * sintheta * sinphi,
                 ks[..., 0] * sintheta + ks[..., 2] * costheta],
                axis=-1)
        else:
            k = np.stack(
                [ks[..., 0] * cosphi + ks[..., 1] * sinphi,
                 -ks[..., 0] * sinphi + ks[..., 1] * cosphi]
                )
            if "kz" in matfile:
                kz = np.repeat(matfile["theta"].T, npts, -1)
                nz = kz.shape[0]
                k = np.concatenate(
                    [np.apply_along_axis(np.tile, 0, k, nz),
                     np.apply_along_axis(np.tile, 0, kz, nviews)], 
                     axis=-1)
    else:
        raise RuntimeError("K-space trajectory not found!")
        
    return k


def _get_schedule(head, matfile, schedulename):
    if "method" in matfile:
        schedule = matfile["method"]
    else:
        try:
            if schedulename is None:
                schedulename = ".".join(schedulename.split(".")[-1].extend(["_method", "mat"]))
            schedule = matlab.read_matfile(schedulename)
            schedule = schedule["method"]
        except Exception:
            schedule = None
    if schedule is not None:
        if "VariableFlip" in schedule.dtype.fields:
            FA = schedule["VariableFlip"][0][0].squeeze()
        else:
            FA = 0.0
        if "VariablePhase" in schedule.dtype.fields:
            phase = schedule["VariableFlip"][0][0].squeeze()
            FA = FA * np.exp(1j * np.deg2rad(phase))
        else:
            FA = 0.0
        if "VariableTE" in schedule.dtype.fields:
            TE = schedule["VariableTE"][0][0].squeeze()
        elif "TE" in schedule.dtype.fields:
            TE = float(schedule["TE"][0][0].squeeze())
        if "VariableTR" in schedule.dtype.fields:
            TR = schedule["VariableTR"][0][0].squeeze()
        else:
            TR = 0.0
        if "InversionTime" in schedule.dtype.fields:
            TI = schedule["InversionTime"][0][0].squeeze()
        else:
            TI = 0.0
    else:
        FA = 0.0
        TE = 0.0
        TR = 0.0
        TI = 0.0
        
    # broadcast contrasts
    FA, TI, TE, TR = np.broadcast_arrays(FA, TI, TE, TR)
    
    # update header
    head.TI = TI
    head.TE = TE
    head.TR = TR
    head.FA = FA
    
    return head
        
        
