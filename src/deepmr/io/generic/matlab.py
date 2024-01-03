"""I/O Routines for MATLAB files."""

__all__ = ["read_matfile"]

import scipy.io
import mat73

from ..utils.pathlib import get_filepath

def read_matfile(filename: str) -> dict:
    """
    Read matfile as a Python dictionary.

    Automatically handle legacy and HDF5 (i.e., -v7.3) formats.    

    Parameters
    ----------
    filename : str
        Path of the file on disk.

    Returns
    -------
    dict
        Deserialized matfile.
        
    Example
    -------
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio

    Get the filename for an example .mat file from the tests/data directory.

    >>> dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    >>> filepath = pjoin(dir, 'testdouble_7.4_GLNX86.mat')

    Load the .mat file contents.

    >>> import deepmr.io
    >>> matfile = deepmr.io.read_matfile(filepath)

    The result is a dictionary, one key/value pair for each variable:
    
    >>> matfile.keys()
    ['testdouble']
    >>> matfile['testdouble']
    array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
            3.92699082, 4.71238898, 5.49778714, 6.28318531]])
    
    Contrary to 'scipy.io', this will load matlab v7.3 files as well, using 'mat73' library [1].
    In addition, '__global__', '__header__' and '__version__' fields are automatically
    discarded.
    
    References
    ----------
    [1]: https://github.com/skjerns/mat7.3/tree/master

    """
    filename = get_filepath(filename, True, "mat")    
    try:
        matfile = scipy.io.loadmat(filename)
        matfile.pop("__globals__", None)
        matfile.pop("__header__", None)
        matfile.pop("__version__", None)
    except:
        matfile = mat73.loadmat(filename)
        
    return matfile

