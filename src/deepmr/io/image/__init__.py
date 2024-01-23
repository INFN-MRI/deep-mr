"""Image IO routines."""

import time

import numpy as np
import torch

from . import dicom as _dicom
from . import nifti as _nifti

# from .dicom import *  # noqa
# from .nifti import *  # noqa

__all__ = ["read_image", "write_image"]

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

def write_image(filename, image, head=None, dataformat="nifti", filepath="./", series_description="", series_number_offset=0, series_number_scale=1000, rescale=False, anonymize=False, verbose=False):
    """
    Write image to disk.

    Parameters
    ----------
    filename : str 
        Name of the file.
    image : np.ndarray
        Complex image data of shape (ncontrasts, nslices, ny, nx).    
    filepath : str, optional
        Path to file. The default is "./".
    head : deepmr.Header, optional
        Structure containing trajectory of shape (ncontrasts, nviews, npts, ndim)
        and meta information (shape, resolution, spacing, etc). If None,
        assume 1mm isotropic resolution, contiguous slices and axial orientation.
        The default is None
    dataformat : str, optional
        Available formats ('dicom' or 'nifti'). The default is 'nifti'.
    series_description : str, optional
        Custom series description. The default is "".
    series_number_offset : int, optional
        Series number offset with respect to the acquired one.
        Final series number is series_number_scale * acquired_series_number + series_number_offset.
        he default is 0.
    series_number_scale : int, optional
        Series number multiplicative scaling with respect to the acquired one. 
        Final series number is series_number_scale * acquired_series_number + series_number_offset.
        The default is 1000.
    rescale : bool, optional
        If true, rescale image intensity between 0 and int16_max.
        Beware! Avoid this if you are working with quantitative maps.
        The default is False.
    anonymize : bool, optional
        If True, remove sensible info from header. The default is "False".
    verbose : bool, optional
        Verbosity flag. The default is "False".
        
    """
    if dataformat == 'dicom':
        _dicom.write_dicom(filename, image, filepath, head, series_description, series_number_offset, series_number_scale, rescale, anonymize, verbose)
    elif dataformat == 'nifti':
        _nifti.write_nifti(filename, image, filepath, head, series_description, series_number_offset, series_number_scale, rescale, anonymize, verbose)
    else:
        raise RuntimeError(f"Data format = {dataformat} not recognized! Please use 'dicom' or 'nifti'")