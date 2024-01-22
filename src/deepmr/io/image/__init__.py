"""Image IO routines."""

import math
import time

import numpy as np
import torch

from . import dicom as _dicom
from . import nifti as _nifti

from .dicom import *  # noqa
from .nifti import *  # noqa

__all__ = ["read_image"]


def read_image(filepath, acqheader=None, device="cpu", verbose=0):
    """
    Read image data from file.

    Parameters
    ----------
    filepath : str
        Path to image file.
    acqheader : Header, deepmr.optional
        Acquisition header loaded from trajectory.
        If not provided, assume Cartesian acquisition and infer from data.
        The default is None.
    device : str, optional
        Computational device for internal attributes. The default is "cpu".
    verbose : int, optional
        Verbosity level (0=Silent, 1=Less, 2=More). The default is 0.
    
    Returns
    -------
    image : torch.tensor
        Complex image data of shape (ncontrasts, nslices, ny, nx).
    head : deepmr.Header
        Metadata for image reconstruction.
    """
    tstart = time.time()
    if verbose >= 1:
        print(f"Reading image from file {filepath}...", end="\t")
        
    done = False
    
    # convert header to numpy
    if acqheader is not None:
        acqheader.numpy()

    # dicom
    if verbose == 2:
        t0 = time.time()
    try:            
        image, head = _dicom.read_dicom(filepath)
        done = True
    except Exception:
        pass
    
    # nifti
    if verbose == 2:
        t0 = time.time()
    try:            
        image, head = _nifti.read_nifti(filepath)
        done = True
    except Exception:
        pass
    
    if not(done):
        raise RuntimeError(f"File (={filepath}) not recognized!")
    if verbose == 2:
        t1 = time.time()
        print(f"done! Elapsed time: {round(t1-t0, 2)} s")
        
    # load trajectory info from acqheader if present
    if acqheader is not None:
        if acqheader.traj is not None:
            head.traj = acqheader.traj
        if acqheader.dcf is not None:
            head.dcf = acqheader.dcf
        if acqheader.t is not None:
            head.t = acqheader.t
            
    # final report
    if verbose == 2:
        print(f"Image shape: (ncontrasts={image.shape[0]}, nz={image.shape[-3]}, ny={image.shape[-2]},  nx={image.shape[-1]})")
        if head.t is not None:
            print(f"Readout time: {round(float(head.t[-1]), 2)} ms")
        if head.traj is not None:
            print(f"Trajectory shape: (ncontrasts={head.traj.shape[0]}, nviews={head.traj.shape[1]}, nsamples={head.traj.shape[2]}, ndim={head.traj.shape[-1]})")      
        if head.dcf is not None:
            print(f"DCF shape: (ncontrasts={head.dcf.shape[0]}, nviews={head.dcf.shape[1]}, nsamples={head.dcf.shape[2]})")
        if head.FA is not None:
            if len(np.unique(head.FA)) > 1:
                print(f"Flip Angle train length: {len(head.FA)}")
            else:
                FA = float(np.unique(head.FA)[0])
                print(f"Constant FA: {round(abs(FA), 2)} deg")
        if head.TR is not None:
            if len(np.unique(head.TR)) > 1:
                print(f"TR train length: {len(head.TR)}")
            else:
                TR = float(np.unique(head.TR)[0])
                print(f"Constant TR: {round(TR, 2)} ms")
        if head.TE is not None:
            if len(np.unique(head.TE)) > 1:
                print(f"Echo train length: {len(head.TE)}")
            else:
                TE = float(np.unique(head.TE)[0])
                print(f"Constant TE: {round(TE, 2)} ms")
        if head.TI is not None and np.allclose(head.TI, 0.0) is False:
            if len(np.unique(head.TI)) > 1:
                print(f"Inversion train length: {len(head.TI)}")
            else:
                TI = float(np.unique(head.TI)[0])
                print(f"Constant TI: {round(TI, 2)} ms")
          
    # cast
    image = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.complex64, device=device)
    head.torch(device)
    
    tend = time.time()
    if verbose == 1:
        print(f"done! Elapsed time: {round(tend-tstart, 2)} s")
    elif verbose == 2:
        print(f"Total elapsed time: {round(tend-tstart, 2)} s")
        
    return image, head