"""MRD data reading routines."""

__all__ = ["read_mrd"]

import numpy as np
import numba as nb

import ismrmrd

from ..utils import mrd
from ..utils.header import Header
from ..utils.pathlib import get_filepath

def read_mrd(filepath, return_ordering=False):
    """
    Read kspace data from mrd file.

    Parameters
    ----------
    filepath : str | list | tuple
        Path to mrd file.
    return_ordering : bool, optional
        If true, return ordering to sort raw data. 
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
    acquisitions, mrdhead = _read_mrd(filepath)
        
    # get all data
    data, traj, dcf = _get_data(acquisitions)
    
    # sort
    data, traj, dcf, ordering = _sort_data(data, traj, dcf, acquisitions, mrdhead)
    
    # get constrats info
    TI = mrd._get_inversion_times(mrdhead)
    TE = mrd._get_echo_times(mrdhead)
    TR = mrd._get_repetition_times(mrdhead)    
    FA = mrd._get_flip_angles(mrdhead)
    
    # get slice locations
    _, firstVolumeIdx, _ = mrd._get_slice_locations(acquisitions)
    
    # build header
    head = Header.from_mrd(mrdhead, acquisitions, firstVolumeIdx)
    
    # update header
    head.FA = FA
    head.TI = TI
    head.TE = TE
    head.TR = TR
    
    # get npts
    if data.size != 0:
        npts = data.shape[-1]
    elif traj.size != 0:
        npts = traj.shape[-2]
    else:
        raise RuntimeError("HDF5 did not contain data nor trajectory")
    
    head.adc = (acquisitions[0].discard_pre, npts - acquisitions[0].discard_post)
    dt = acquisitions[0].sample_time_us * 1e-3 # ms    
    head.t = dt * np.arange(npts, dtype=np.float32)
    head.traj = traj
    head.dcf = dcf
    
    if return_ordering:
        head.user["ordering"] = ordering
        
    return data, head
        

# %% subroutines
def _read_mrd(filename):
    # open file
    with ismrmrd.File(filename) as dset:
        
        # read header
        mrdhead = _read_header(dset["dataset"])
        
        # read acquisitions
        acquisitions = dset["acquisitions"]
    
    return acquisitions, mrdhead


def _read_header(dset):
    xml_header = dset["xml"]
    xml_header = xml_header.decode("utf-8")
    return ismrmrd.xsd.CreateFromDocument(xml_header)


def _get_data(acquisitions):
    if acquisitions[0].data.size != 0:
        data = np.stack([acq.data for acq in acquisitions], axis=0)
    else:
        data = None
    
    if acquisitions[0].traj.size != 0:
        trajdcf = np.stack([acq.traj for acq in acquisitions], axis=0)
        traj = trajdcf[..., :-1]
        dcf = trajdcf[..., -1]
    else:
        traj, dcf = None, None
    
    return data, traj, dcf


def _sort_data(data, traj, dcf, acquisitions, mrdhead):
    
    # order
    icontrast = np.asarray([acq.idx.contrast for acq in acquisitions])
    iz = np.asarray([acq.idx.slice for acq in acquisitions])
    iview = np.asarray([acq.idx.kspace_encode_step_1 for acq in acquisitions])
                
    # get geometry from header
    shape = mrdhead.encoding[0].encodingLimits
                
    # get sizes
    ncoils = data.shape[1]
    ncontrasts = shape.contrast.maximum+1
    nslices = shape.slice.maximum+1
    nviews = shape.kspace_encoding_step_1.maximum+1
    npts = data.shape[-1]
    ndims = traj.shape[-1] # last tims stores dcfs

    # get fov, matrix size and kspace size
    shape = (ncontrasts, nslices, nviews, npts)
    
    # sort trajectory, dcf and t
    datatmp = np.zeros([ncoils] + list(shape), dtype=np.complex64)
        
    if traj is not None:
        # preallocate
        trajtmp = np.zeros(list(shape) + [ndims], dtype=np.float32)
        dcftmp = np.zeros(shape, dtype=np.float32)
            
        # actual sorting
        _data_sorting(datatmp, data, icontrast, iz, iview)
        _traj_sorting(trajtmp, traj, icontrast, iz, iview)
        _dcf_sorting(dcftmp, dcf, icontrast, iz, iview)
           
        # assign
        data = np.ascontiguousarray(datatmp.squeeze())
        traj = np.ascontiguousarray(trajtmp.squeeze())
        dcf = np.ascontiguousarray(dcftmp.squeeze())
    else:
        # actual sorting
        _data_sorting(datatmp, data, icontrast, iz, iview)

        # assign
        data = np.ascontiguousarray(datatmp.squeeze())
        
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
def _traj_sorting(output, input, echo_num, slice_num, view_num):
    # get size
    nframes = input.shape[0]
    
    # actual reordering
    for n in range(nframes):
        iecho = echo_num[n]
        islice = slice_num[n]
        iview = view_num[n]
        output[iecho, islice, iview, :, :] = input[n]
        
@nb.njit(cache=True)
def _dcf_sorting(output, input, echo_num, slice_num, view_num):
    # get size
    nframes = input.shape[0]
    
    # actual reordering
    for n in range(nframes):
        iecho = echo_num[n]
        islice = slice_num[n]
        iview = view_num[n]
        output[iecho, islice, iview, :] = input[n]
        
    
