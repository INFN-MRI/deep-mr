"""I/O routines for MRD raw data."""

__all__ = ["read_mrd_rawdata"]

from ..generic import mrd

def read_mrd_rawdata(filepath):
    """
    Read kspace data from MRD file.

    Parameters
    ----------
    filepath : str
        Path to MRD file.
    
    Returns
    -------
    data : np.ndarray
        Complex k-space data of shape (ncoils, ncontrasts, nslices, nview, npts).
    head : deepmr.Header
        Metadata for image reconstruction.
    """
    data, head = mrd.read_mrd(filepath)
    
    return data, head
        

