"""I/O routines for GEHC raw data."""

__all__ = ["read_gehc_rawdata"]

import warnings

import numpy as np

try:
    import gehc
    __GEHC_AVAILABLE__ = True
except Exception:
    __GEHC_AVAILABLE__ = False

from ..generic.pathlib import get_filepath
from ..types import mrd
from ..types.header import Header

def read_gehc_rawdata(filepath, acqheader=None):
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
    # get full path
    filepath = get_filepath(filepath, True, ".7", "h5")
    
    # try and read
    if __GEHC_AVAILABLE__:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # change the hook
            data, head = gehc.read_rawdata(filepath, acqheader)
            
            # build header
            head = Header.from_gehc(head)
            
        return data, head
    else:
        print("GEHC reader is private - ask for access")
        return None, None

        
        
    


