"""GEHC data reading routines."""

__all__ = ["read_gehc"]

import warnings

import numpy as np

try:
    import gehc
    __GEHC_AVAILABLE__ = True
except Exception:
    __GEHC_AVAILABLE__ = False
    
from ..utils import mrd
from ..utils.header import Header
from ..utils.pathlib import get_filepath

def read_gehc(filepath: str, return_ordering: bool = False):
    """
    Read kspace data from GEHC file.

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
    filepath = get_filepath(filepath, True, ".7", "h5")
    
    # try and read
    if __GEHC_AVAILABLE__:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # change the hook
            data, header = gehc.read_rawdata(filepath)
        return data, header
    else:
        print("GEHC reader is private - ask for access")
        return None, None
    
    # # get constrats info
    # TI = mrd._get_inversion_times(mrdhead)
    # TE = mrd._get_echo_times(mrdhead)
    # TR = mrd._get_repetition_times(mrdhead)    
    # FA = mrd._get_flip_angles(mrdhead)
    
    # # get slice locations
    # _, firstVolumeIdx, _ = mrd._get_slice_locations(acquisitions)
    
    # # build header
    # header = Header.from_mrd(mrdhead, acquisitions, firstVolumeIdx)
    
    # # update header
    # header.FA = FA
    # header.TI = TI
    # header.TE = TE
    # header.TR = TR
        
    # if return_ordering:
    #     return data, traj, dcf, header, ordering
    # else:
    #     return data, traj, dcf, header
        
        
    


