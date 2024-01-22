"""MRD data reading routines."""

__all__ = ["read_mrd"]

import numpy as np
import numba as nb

import ismrmrd

from ..types import mrd
from ..types.header import Header

from .pathlib import get_filepath

def read_mrd(filepath, external=False):
    """
    Read kspace data from mrd file.

    Parameters
    ----------
    filepath : str
        Path to mrd file.
    external : bool, optional
        If true, skip data and return ordering to sort raw data. 
        Useful if sequence is stored separately from raw data.
        The default is False.
    
    Returns
    -------
    data : np.ndarray
        Complex k-space data of shape (ncoils, ncontrasts, nslices, nview, npts).
    header : deepmr.Header
        Metadata for image reconstruction.

    """
    # get full path
    filepath = get_filepath(filepath, True, "h5")
    
    # load mrd
    with ismrmrd.File(filepath) as file:
        # read data
        acquisitions, mrdhead = _read_mrd(file)
            
        # get all data
        data, trajdcf = _get_data(acquisitions, external)
                
        # sort
        data, traj, dcf, ordering = _sort_data(data, trajdcf, acquisitions, mrdhead)
    
        # get constrats info
        TI = mrd._get_inversion_times(mrdhead)
        TE = mrd._get_echo_times(mrdhead)
        TR = mrd._get_repetition_times(mrdhead)    
        FA = mrd._get_flip_angles(mrdhead)
        
        # get npts
        if data is not None:
            npts = data.shape[-1]
        elif traj is not None:
            npts = traj.shape[-2]
        else:
            raise RuntimeError("HDF5 did not contain data nor trajectory")
            
        # get adc
        adc = (acquisitions[0]["head"]["discard_pre"], npts - acquisitions[0]["head"]["discard_post"])
        
        # build header
        if external:
            firstVolumeIdx = 0
        else:
            _, firstVolumeIdx, _ = mrd._get_slice_locations(acquisitions)
            
        head = Header.from_mrd(mrdhead, acquisitions, firstVolumeIdx, external)
        
        # get slice profile
        tmp = mrd._find_in_user_params(mrdhead.userParameters.userParameterString, "slice_profile")
        if tmp is not None:
            head.user["slice_profile"] = mrd._bytes_to_numpy(tmp["slice_profile"]).astype(np.float32)
            
        # get basis
        tmp = mrd._find_in_user_params(mrdhead.userParameters.userParameterString, "basis")
        if tmp is not None:
            basis = mrd._bytes_to_numpy(tmp["basis"]).astype(np.complex64)
            if np.isreal(basis).all():
                basis = basis.real
            elif np.isreal(basis.imag + 1j * basis.real).all():
                basis = basis.imag
            head.user["basis"] = basis
            
        # get separability
        tmp = mrd._find_in_user_params(mrdhead.userParameters.userParameterString, "separable")
        if tmp is not None:
            separable = tmp["separable"]
            if separable == "True":
                separable = True
            elif separable == "False":
                separable = False
            head.user["separable"] = separable
            
        # get mode
        tmp = mrd._find_in_user_params(mrdhead.userParameters.userParameterString, "mode")
        if tmp is not None:
            mode = tmp["mode"]
            head.user["mode"] = mode
    
    # update header
    head.traj = traj
    head.dcf = dcf
    head.FA = FA
    head.TI = TI
    head.TE = TE
    head.TR = TR    
    head._adc = adc
    
    if external:
        head.user["ordering"] = ordering
        
    return data, head
        

# %% subroutines
def _read_mrd(file):
        
    # read header
    mrdhead = file["dataset"].header
    
    # read acquisitions
    acquisitions = file["dataset"].acquisitions.data
    
    return acquisitions, mrdhead


def _get_data(acquisitions, external):
    
    # number of acquisitions
    nacq = len(acquisitions)
    
    # get data
    if external:
        data = None
    else:
        data = np.stack([acquisitions[n]["data"] for n in range(nacq)], axis=0)
        data = data[..., ::2] + 1j * data[..., 1::2] 
    
    # get trajectory and dcf
    if acquisitions[0]["traj"].size != 0:
        trajdcf = np.stack([acquisitions[n]["traj"] for n in range(nacq)], axis=0)
    else:
        trajdcf = None
        
    # reshape
    if data is not None:
        npts = acquisitions[0]["head"]["number_of_samples"]
        nchannels = acquisitions[0]["head"]["available_channels"]
        data = data.reshape(-1, nchannels, npts)
    
    if trajdcf is not None:
        npts = acquisitions[0]["head"]["number_of_samples"]
        ndims = acquisitions[0]["head"]["trajectory_dimensions"]
        trajdcf = trajdcf.reshape(-1, npts, ndims)
    else:
        trajdcf = None
        
    return data, trajdcf


def _sort_data(data, trajdcf, acquisitions, mrdhead):
    
    # number of acquisitions
    nacq = len(acquisitions)
    
    # order
    icontrast = np.asarray([acquisitions[n]["head"]["idx"]["contrast"] for n in range(nacq)])
    iz = np.asarray([acquisitions[n]["head"]["idx"]["slice"] for n in range(nacq)])
    iview = np.asarray([acquisitions[n]["head"]["idx"]["kspace_encode_step_1"] for n in range(nacq)])
                
    # get geometry from header
    shape = mrdhead.encoding[0].encodingLimits
                
    # get sizes
    ncoils = acquisitions[0]["head"]["available_channels"]
    ncontrasts = shape.contrast.maximum+1
    nviews = shape.kspace_encoding_step_1.maximum+1
    npts = acquisitions[0]["head"]["number_of_samples"]
    ndims = acquisitions[0]["head"]["trajectory_dimensions"] # last tims stores dcfs
    
    if ndims-1 == 2:
        nslices = shape.slice.maximum+1
    elif ndims-1 == 3:
        nslices = 1

    # get fov, matrix size and kspace size
    shape = (ncontrasts, nslices, nviews, npts)
    
    # sort data, trajectory, dcf
    if data is not None:
        datatmp = np.zeros([ncoils] + list(shape), dtype=np.complex64)
        _data_sorting(datatmp, data, icontrast, iz, iview)
        data = np.ascontiguousarray(datatmp.squeeze())
        
    if trajdcf is not None:
        trajdcftmp = np.zeros(list(shape) + [ndims], dtype=np.float32)
        _trajdcf_sorting(trajdcftmp, trajdcf, icontrast, iz, iview)
        trajdcf = np.ascontiguousarray(trajdcftmp.squeeze())           
        traj, dcf = trajdcf[..., :-1], trajdcf[..., -1]
    else:
        # actual sorting
        traj, dcf = None, None
        
    # keep ordering
    ordering = np.stack((icontrast, iz, iview), axis=0)
        
    return data, traj, dcf, ordering


@nb.njit(cache=True)
def _data_sorting(output, input, echo_num, slice_num, view_num):
    # get size
    nframes = input.shape[0]
    
    # actual reordering
    for n in range(nframes):
        iecho = echo_num[n]
        islice = slice_num[n]
        iview = view_num[n]
        output[:, iecho, islice, iview, :] = input[n]
        
        
@nb.njit(cache=True)
def _trajdcf_sorting(output, input, echo_num, slice_num, view_num):
    # get size
    nframes = input.shape[0]
    
    # actual reordering
    for n in range(nframes):
        iecho = echo_num[n]
        islice = slice_num[n]
        iview = view_num[n]
        output[iecho, islice, iview, :, :] = input[n]
        

