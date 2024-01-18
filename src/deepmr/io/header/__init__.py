"""Acquisition Header IO routines."""

from . import base as _base
# from . import bart as _bart
from . import matlab as _matlab
from . import mrd as _mrd

__all__ = ["read_acquisition_header", "write_acquisition_header"]

def read_acquisition_header(filepath):
    """
    Read acquisition header from file.

    Parameters
    ----------
    filepath : str
        Path to acquisition header file.
    
    Returns
    -------
    head : deepmr.Header
        Deserialized acquisition header.
        
    """
    done = False
    
    # mrd
    if filepath.endswith(".h5"):
        try:
            head = _mrd.read_mrd_acqhead(filepath)
            done = True
        except Exception:
            pass
    
    # matfile
    if filepath.endswith(".mat") and not(done):
        try:
            head = _matlab.read_matlab_acqhead(filepath)
            done = True
        except Exception:
            pass
        
    # bart
    # if filepath.endswith(".cfl") and not(done):
    #     try:
    #         head = _bart.read_bart_acqhead(filepath)
    #         done = True
    #     except Exception:
    #         pass
        
    # fallback
    if filepath.endswith(".h5") and not(done):
        try:
            done = True
            head = _base.read_base_acqheader(filepath)
        except Exception:
            raise RuntimeError("File not recognized!")
            
    return head
            
def write_acquisition_header(head, filepath, dataformat="hdf5"):
    """
    Write acquisition header to file.

    Parameters
    ----------
    head: deepmr.Header
        Structure containing trajectory of shape (ncontrasts, nviews, npts, ndim)
        and meta information (shape, resolution, spacing, etc).
    filepath : str 
        Path to file.
    dataformat: str, optional
        Available formats ('mrd' or 'hdf5'). The default is 'hdf5.'
        
    """
    if dataformat == 'hdf5':
        _base.write_base_acqheader(head, filepath)
    elif dataformat == 'mrd':
        _mrd.write_mrd_acqhead(head, filepath)
    else:
        raise RuntimeError(f"Data format = {dataformat} not recognized! Please use 'mrd' or 'hdf5'")
        
        
        
        
        
        
        
        
        