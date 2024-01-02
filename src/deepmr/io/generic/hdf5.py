"""I/O Routines for HDF5 files.
"""

__all__ = ["read_hdf5", "write_hdf5"]

import copy

import h5py
import numpy as np

from ..utils.pathlib import get_filepath

dtypes = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    int,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float16,
    np.float32,
    np.float64,
)

def read_hdf5(filename: str) -> dict:
    """
    Read HDF5 file as a Python dictionary

    Parameters
    ----------
    filename : str
        path to file on disk.

    Returns
    -------
    dict
        deserialized HDF5 file.
        
    Example
    -------
    Define an exemplary dictionary and save to file:

    >>> import os
    >>> import numpy as np
    >>> import deepmr.io
    >>> pydict = {'headerstr': 'someinfo', 'testdouble': np.ones(3, dtype=np.float32)}
    >>> filepath = os.path.realpath('.')
    >>> deepmr.io.write_hdf5(filepath)
    
    Load from disk:
    
    >>> loaded_dict = deepmr.io.read_hdf5(filepath)
    
    Result is the same dictionary created before:
        
    >>> loaded_dict.keys()
    ['headerstr', 'testdouble']
    >>> matfile['testdouble']
    array([1.0, 1.0, 1.0])
    >>> matfile['headerstr']
    'someinfo'

    """
    filename = get_filepath(filename, True, "h5")
    with h5py.File(filename, "r") as h5file:
        return _recursively_load_dict_contents_from_group(h5file, "/")


def write_hdf5(input: dict, filename: str):
    """
    Write a given dictionary to HDF5 file.


    Parameters
    ----------
    input : dict
        Dictionary containing all groupnames as keys and
        datasets as values.
    filename : str
        path of HDF5 file.
        
    Example
    -------
    Define an exemplary dictionary:

    >>> import numpy as np
    >>> pydict = {'headerstr': 'someinfo', 'testdouble': np.ones(3, dtype=np.float32)}

    Save to an hdf5 file the dictionary.

    >>> import os
    >>> import deepmr.io
    >>> filepath = os.path.realpath('.')
    >>> deepmr.io.write_hdf5(filepath)

    """
    input = copy.deepcopy(input)
    with h5py.File(filename, "w") as h5file:
        _recursively_save_dict_contents_to_group(h5file, "/", input)


def _recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            tmp = item[()]
            if isinstance(tmp, bytes):
                tmp = tmp.decode()
            ans[key] = tmp
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


def _recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (*dtypes, str, bytes)):
            h5file[path + key] = item
        elif np.isscalar(item):
            h5file[path + key] = (item,)
        elif isinstance(item, (list, tuple)):
            h5file[path + key] = np.asarray(item)
        elif isinstance(item, np.ndarray):
            h5file[path + key] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        elif item is None:
            pass
        else:
            raise ValueError(f"Cannot save {type(item)} type")