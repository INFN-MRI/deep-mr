"""I/O routines for MATLAB acquisition header."""

__all__ = ["read_matlab_acqhead"]

import numpy as np

from ..generic import matlab
from ..types.header import Header

def read_matlab_acqhead(filename, dcfname=None, schedulename=None):
    """
    Read acquistion header from matlab file.

    Parameters
    ----------
    filename : str
        Path to the file on disk.

    Returns
    -------
    head : deepmr.Header
        Deserialized acqusition header.
        
    """
    # load dcf
    matfile, filename = matlab.read_matfile(filename, True)
    
    # get k space trajectory
    k, reshape = _get_trajectory(matfile)
    ndim = k.shape[-1]
    
    # get indexes
    if "adc" in matfile:
        ind = matfile["ind"]
    else:
        if "ind" in matfile:
            ind = matfile["ind"][0].astype(bool)
        elif "inds" in matfile:
            ind = matfile["inds"].squeeze().astype(bool)
        else:
            raise RuntimeError("ADC indexes not found!")
        ind = np.argwhere(ind)[[0, -1]].squeeze()
    
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
        t = np.atleast_2d(matfile["t"][0])
    elif "ts" in matfile:
        t = matfile["ts"].squeeze()
    else:
        raise RuntimeError("Sampling time not found!")
    
    # s to ms
    if np.round(t).max() == 0:
        t *= 1e3
        
    # remove offset from sampling time
    t -= t[0]
    
    # get matrix
    if "mtx" in matfile:
        shape = matfile["mtx"].squeeze()[::-1]
    elif "npix" in matfile:
        shape = matfile["npix"].squeeze()[::-1]
    elif "shape" in matfile:
        shape = matfile["shape"]
    else:
        raise RuntimeError("Matrix size not found!")
        
    # expand scalar
    if len(shape) == 1:
        shape = [int(shape)] * ndim
    
    # get resolution
    if "resolution" in matfile:
        resolution = matfile["resolution"]
    else:
        if "fov" in matfile:
            fov = [float(matfile["fov"].squeeze()) * 1e3] * ndim # mm
        else:
            raise RuntimeError("Field of View not found!")
        resolution = (np.asarray(fov) / np.asarray(shape)).tolist() 
    
    # get spacing
    if "spacing" in matfile:
        spacing = matfile["spacing"]
    else:
        spacing = resolution[0]
            
    # initialize header
    head = Header(shape, resolution, spacing, t=t)
    head.adc = ind
   
    # get schedule file
    head = _get_schedule(head, matfile, schedulename)
    
    # get ordering
    if reshape:
        ncontrasts = len(head.FA)
        nviews = int(k.shape[0] / ncontrasts)  
        
        k = k.reshape(nviews, ncontrasts, -1, ndim).swapaxes(0, 1)
        k = np.ascontiguousarray(k).astype(np.float32)
        if dcf is not None:
            dcf = dcf.reshape(nviews, ncontrasts, -1).swapaxes(0, 1)
            dcf = np.ascontiguousarray(dcf).astype(np.float32)
        
    # append
    head.traj = k
    head.dcf = dcf
    
    return head
 
# %% subroutines
def _get_trajectory(matfile):
    
    reshape = True
    if "traj" in matfile:
        k = matfile["traj"]
        reshape = False
    elif "k" in matfile:
        k = matfile["k"]
        # get shape
        if "t" in matfile:
            npts = matfile["t"].shape[-1]
        elif "ts" in matfile:
            npts = matfile["ts"].shape[-1]
        else:
            raise RuntimeError("Time not found!")
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
        
    return k, reshape

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
            FA = schedule["VariableFlip"][0][0].squeeze().astype(np.float32)
        else:
            FA = 0.0
        if "VariablePhase" in schedule.dtype.fields:
            phase = schedule["VariableFlip"][0][0].squeeze().astype(np.float32)
            FA = FA * np.exp(1j * np.deg2rad(phase))
        else:
            FA = 0.0
        if "VariableTE" in schedule.dtype.fields:
            TE = schedule["VariableTE"][0][0].squeeze().astype(np.float32)
        elif "TE" in schedule.dtype.fields:
            TE = schedule["TE"][0][0].squeeze().astype(np.float32)
        if "VariableTR" in schedule.dtype.fields:
            TR = schedule["VariableTR"][0][0].squeeze().astype(np.float32)
        else:
            TR = 0.0
        if "InversionTime" in schedule.dtype.fields:
            TI = schedule["InversionTime"][0][0].squeeze().astype(np.float32)
        else:
            TI = 0.0
    else:
        if "FA" in matfile:
            FA = matfile["FA"]
        else:
            FA = 0.0
        if "TE" in matfile:
            TE = matfile["TE"]
        else:
            TE = 0.0
        if "TR" in matfile:
            TR = matfile["TR"]
        else:
            TR = 0.0
        if "TI" in matfile:
            TI = matfile["TI"]
        else:
            TI = 0.0
        
        # rf phase
        if "user" in matfile and "rf_phase" in matfile["user"]:
            phase = np.frombuffer(matfile["user"]["rf_phase"], dtype=np.float32)
            FA = FA * np.exp(1j * np.deg2rad(phase))
            
    # broadcast contrasts
    FA, TI, TE, TR = np.broadcast_arrays(FA, TI, TE, TR)
    
    # update header
    head.TI = TI
    head.TE = TE
    head.TR = TR
    head.FA = FA
    
    return head
        
        
