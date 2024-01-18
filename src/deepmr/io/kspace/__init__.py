"""KSpace IO routines."""

import math
import numpy as np
import numba as nb

from . import gehc as _gehc
from . import mrd as _mrd
# from . import siemens as _siemens

__all__ = ["read_rawdata"]

def read_rawdata(filepath, acqheader=None, device=None):
    """
    Read kspace data from GEHC file.

    Parameters
    ----------
    filepath : str
        Path to PFile or ScanArchive file.
    acqheader : Header, deepmr.optional
        Acquisition header loaded from trajectory.
        If not provided, assume Cartesian acquisition and infer from data.
        The default is None.
    
    Returns
    -------
    data : np.ndarray
        Complex k-space data of shape (ncoils, ncontrasts, nslices, nview, npts).
    head : deepmr.Header
        Metadata for image reconstruction.
    """
    done = False
    
    # convert header to numpy
    
    # mrd
    try:
        data, head = _mrd.read_mrd_rawdata(filepath)
        done = True
    except Exception:
        pass
    
    # gehc
    if not(done):
        try:
            data, head = _gehc.read_gehc_rawdata(filepath, acqheader)
            done = True
        except Exception:
            pass
        
    # siemens
    # if not(done):
    #     try:
    #         head = _siemens.read_siemens_rawdata(filepath, acqheader)
    #         done = True
    #     except Exception:
    #         pass

    # check if we loaded data
    if not(done):
        raise RuntimeError("File not recognized!")
        
    # transpose
    data = data.transpose(2, 0, 1, 3, 4)
    
    # select actual readout
    data = _select_readout(data, head)    

    # center fov
    data = _fov_centering(data, head)
    
    # check individual axis for cartesian

    # separation of slices in fourier space for hybrid trajectories
    
    # estimate mask
    
    # cast to data and header to torch
    
    return data, head

# %% sub routines
def _select_readout(data, head):
    if head.adc[-1] == data.shape[-1]:
        data = data[..., head.adc[0]:]
    else:
        data = data[..., head.adc[0]:head.adc[1]+1]
    return data
    
def _fov_centering(data, head):
    
    if head.traj is not None:
        
        # ndimensions
        ndim = head.traj.shape[-1]
        
        # shift (mm)
        dr = np.asarray(head.shift)[:ndim]
        
        # convert in units of voxels
        dr /= head.resolution[::-1]
        
        # apply
        data *= np.exp(1j * 2 * math.pi * (head.traj * dr).sum(axis=-1))
        
    return data

def _estimate_acq_type(data, head):
    
    if head.traj is not None:
        ndim = head.traj.shape[-1]
        traj = head.traj * head.shape
        iscart = [np.allclose(traj[..., n].astype(float), traj[..., n].astype(int)) for n in range(ndim)]
        if np.all(iscart):
            acq_type = "cartesian"
        elif np.any(iscart):
            acq_type = "hybrid"
        else:
            acq_type = "noncartesian"
    else:
        acq_type = "cartesian"
        iscart = None
    
    return acq_type, iscart

def _decouple_hybrid(data, head, iscart):
    assert np.sum(iscart) == 1, f"Only one Cartesian axis is allowed for hybrid trajectores, found {np.sum(iscart)}"
    
    # find cartesian axis
    cartax = int(np.argwhere(iscart).squeeze())
    
    # get trajectory and dcf
    traj = head.traj    
    dcf = head.dcf

    # if trajectory does not account for partition already, assume
    # it is flattened in the contrast dimension (all contrasts, then next slice)
    if len(traj.shape) < 5:
        nz = len(np.unique(traj[..., cartax]))
        traj = traj.reshape(nz, int(traj.shape[0] / nz), traj.shape[2], traj.shape[3], -1)
        if dcf is not None:
            dcf = dcf.reshape(nz, int(dcf.shape[0] / nz), dcf.shape[2], -1)
        data = data.reshape(data.shape[0], data.shape[1], nz, int(data.shape[2] / nz), data.shape[3], -1)
    else:
        nz = traj.shape[0]
    
    # split cartesian and noncartesians axes
    cart = traj[:, 0, 0, 0, cartax]
    noncart = np.delete(traj, cartax, axis=-1)
    
    # check if it is separable
    isseparable = np.asarray([np.allclose(noncart[0], noncart[n]) for n in range(nz)])
    isseparable = np.all(isseparable)
    
    # if it is separable, sort
    if isseparable:
        order = np.argsort(cart[:, 0, 0, 0])
        
        # sort
        data = data[order, ...]
        
        # actual separation
        data = np.fft.fftshift(np.fft.fft(np.fft.fftshift(data, axes=0), axis=0), axes=0)
        head.traj = noncart[0]
        if dcf is not None:
            head.dcf = dcf[0]
    else: # if it is not separable, stack all along view dimension
        data = data.transpose(1, 2, 0, 3, -1)
        traj = traj.swapaxes(0, 1)
        dcf = dcf.swapaxes(0, 1)
        
        data = data.reshape(data.shape[0], data.shape[1], -1, data.shape[-1]) 
        traj = traj.reshape(traj.shape[0], -1, traj.shape[-2], traj.shape[-1]) 
        dcf = dcf.reshape(dcf.shape[0], -1, dcf.shape[-1])
        
        head.traj = traj
        head.dcf = dcf
        
    return data, head

def _decouple_cartesian(data, head):
    pass
    # # estimate mask
    # if head.traj is not None:
    #     coord = (head.traj * head.shape).astype(int)
        
    #     # 3D sample y-k plane (trajectory should be (ncontrasts, nviews=ny*nz, nsample, ndim))
    #     if coord.shape[-1] == 3:
    #         coord = coord[..., :-1] # (ncontrasts, ny*nz, 2)
            
    #         # initialize mask
    #         mask = np.zeros([coord.shape[0]] + list(head.shape[:-1]), dtype=int)
            
    # else:
        
            
        
    
        
    
    
    
    
