"""Utils for HDF5 file handling."""

__all__ = ["load", "dump"]

import copy
from typing import Dict

import h5py
import numpy as np

# %% utils
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
        # elif item is None:
        #     pass
        else:
            raise ValueError(f"Cannot save {type(item)} type")


# %% actual implementation
def load(filename: str) -> Dict:
    """
    Returns a dictionary containing the groups as keys and the
    datasets as values from given hdf file.

    Args:
        filename (str): path to file.

    Returns:
        dict : The dictionary containing all groupnames as keys and
               datasets as values.
    """
    with h5py.File(filename, "r") as h5file:
        return _recursively_load_dict_contents_from_group(h5file, "/")


def dump(input: Dict, filename: str):
    """
    Adds keys of given dict as groups and values as datasets
    to the given hdf-file or group object.

    Args:
        input (dict): The dictionary containing all groupnames as keys and
                      datasets as values.
       filename (str): path to file.

    """
    input = copy.deepcopy(input)
    with h5py.File(filename, "w") as h5file:
        _recursively_save_dict_contents_to_group(h5file, "/", input)
