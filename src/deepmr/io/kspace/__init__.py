"""KSpace IO routines."""

import math
import numpy as np

import torch

from . import gehc as _gehc
from . import mrd as _mrd
# from . import siemens as _siemens

__all__ = ["read_rawdata"]

def read_rawdata(filepath, acqheader=None, device="cpu"):
    """
    Read kspace data from file.

    Parameters
    ----------
    filepath : str
        Path to kspace file.
    acqheader : Header, deepmr.optional
        Acquisition header loaded from trajectory.
        If not provided, assume Cartesian acquisition and infer from data.
        The default is None.
    
    Returns
    -------
    data : np.ndarray
        Complex k-space data of shape (nslices, ncoils, ncontrasts, nviews, nsamples).
    head : deepmr.Header
        Metadata for image reconstruction.
    """
    done = False
    
    # convert header to numpy
    head.numpy()

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
    data = data.transpose(2, 0, 1, 3, 4) # (slice, coil, contrast, view, sample)
    
    # select actual readout
    data = _select_readout(data, head)    

    # center fov
    data = _fov_centering(data, head)
    
    # remove oversampling for Cartesian
    if "mode" in head.user:
        if head.user["mode"][2:] == "cart":
            data, head = _remove_oversampling(data, head)
    
    # transpose readout in slice direction for 3D Cartesian
    if "mode" in head.user:
        if head.user["mode"] == "3Dcart":
            data = data.transpose(-1, 1, 2, 0, 3) # (z, ch, e, y, x) -> (x, ch, e, z, y)
            
    # decouple separable acquisition
    if "separable" in head.user and head.user["separable"]:
        data = _fft(data, 0)
        
    # set-up transposition
    if "mode" in head.user:
        if head.user["mode"][:2] == "2D":
            head.transpose = [1, 0, 2, 3]
        elif head.user["mode"] == "3Dnoncart":
            head.transpose = [1, 0, 2, 3]
        elif head.user["mode"] == "3Dcart":
            head.transpose = [1, 2, 3, 0]
        
        # remove unused trajectory for cartesian
        if head.user["mode"][2:] == "cart":
            head.traj = None
            head.dcf = None
            
    # clean header
    head.user.pop("mode", None)
    head.user.pop("separable", None)
        
    # cast
    data = torch.as_tensor(np.ascontigousarray(data), dtype=torch.complex64, device=device)
    head.torch(device)

    return data, head

# %% sub routines
def _select_readout(data, head):
    if head._adc[-1] == data.shape[-1]:
        data = data[..., head._adc[0]:]
    else:
        data = data[..., head._adc[0]:head._adc[1]+1]
    return data
    
def _fov_centering(data, head):
    
    if head.traj is not None:
        
        # ndimensions
        ndim = head.traj.shape[-1]
        
        # shift (mm)
        dr = np.asarray(head._shift)[:ndim]
        
        # convert in units of voxels
        dr /= head._resolution[::-1][:ndim]
        
        # apply
        data *= np.exp(1j * 2 * math.pi * (head.traj * dr).sum(axis=-1))
        
    return data

def _remove_oversampling(data, head):
    if data.shape[-1] != head.shape[-1]: # oversampled
        center = int(data.shape[-1] // 2)
        hwidth = int(head.shape[-1] // 2)
        data = _fft(data, -1)
        data = data[..., center-hwidth:center+hwidth]
        data = _fft(data, -1)
        dt = np.diff(head.t)[0]
        head.t = np.arange(data.shape[-1]) * dt
    
    return data, head

def _fft(data, axis):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(data, axes=axis), axis=axis), axes=axis)