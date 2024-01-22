"""Acquisition Header IO routines."""

import copy
import time

from . import base as _base
# from . import bart as _bart
from . import matlab as _matlab
from . import mrd as _mrd

__all__ = ["read_acquisition_header", "write_acquisition_header"]


def read_acquisition_header(filepath, device="cpu", verbose=False, *args):
    """
    Read acquisition header from file.

    Parameters
    ----------
    filepath : str
        Path to acquisition header file.
    device : str, optional
        Computational device for internal attributes. The default is "cpu".
    verbose : bool, optional
        Verbosity flag. The default is False.
    
    Args (matfiles)
    ---------------
    dcfpath : str, optional
        Path to the dcf file.
        The default is None.
    methodpath : str, optional
        Path to the schedule description file.
        The default is None.
    sliceprofpath : str, optional
        Path to the slice profile file.
        The default is None.
    
    
    Returns
    -------
    head : deepmr.Header
        Deserialized acquisition header.
        
    """
    tstart = time.time()
    if verbose:
        print(f"Reading acquisition header from file {filepath}...", end="\t")
        
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
            head = _matlab.read_matlab_acqhead(filepath, *args)
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
            raise RuntimeError(f"File (={filepath}) not recognized!")
    
    # check if we loaded data
    if not(done):
        raise RuntimeError(f"File (={filepath}) not recognized!")
        
    # normalize trajectory
    if head.traj is not None:
        traj_max = ((head.traj**2).sum(axis=-1)**0.5).max()
        head.traj = head.traj / (2 * traj_max) # normalize to (-0.5, 0.5)
        
    # cast
    head.torch(device)
    
    tend = time.time()
    if verbose:
        print(f"done! Elapsed time: {round(tend-tstart, 2)} s...")
            
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
    head = copy.deepcopy(head)
    head.ref_dicom = None
    if dataformat == 'hdf5':
        _base.write_base_acqheader(head, filepath)
    elif dataformat == 'mrd':
        _mrd.write_mrd_acqhead(head, filepath)
    else:
        raise RuntimeError(f"Data format = {dataformat} not recognized! Please use 'mrd' or 'hdf5'")
        
        
        
        
        
        
        
        
        