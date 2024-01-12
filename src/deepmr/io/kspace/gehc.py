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
            
            # build header
            header = Header.from_gehc(header)
            
        return data, header
    else:
        print("GEHC reader is private - ask for access")
        return None, None

        
        
    


