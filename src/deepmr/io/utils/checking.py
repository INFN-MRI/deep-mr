"""Utils for type checking."""

__all__ = ["is_builtin_type", "get_type", "get_structured_array_dtype"]

from typing import Dict, get_args

import numpy as np

# %% utils
builtin_types = {
    int,
    float,
    complex,
    bool,
    str,
    bytes,
    bytearray,
    memoryview,
    list,
    tuple,
    range,
    set,
    frozenset,
    dict,
}


# %% implementation
def is_builtin_type(input: object):
    """
    Check wheteher the input is a built-in or Python data.

    Args:
        input (object): an arbitrary Python object.

    Returns:
        bool: True if object is of built-in type, otherwise False.
    """
    return isinstance(input, tuple(builtin_types))

def get_type(input: object):
    """
    Infer type of input value.

    For containers (array, list, etc) infer the underlying datatype, e.g.,
    'List[float]' returns 'float'.

    Args:
        input (object): an arbitrary Python object.

    Returns:
        input base type.
    """
    try:
        input = get_args(input)[0]
    except Exception:
        pass
    return input

def get_structured_array_dtype(input: Dict) -> np.dtype:
    """
    Create numpy dtype from a nested dictionary.

    Args:
        input (dict): Input dictionary.

    Returns:
        np.dtype: object created by the dtypes of the
                  (nested) dictionary fields.
    """
    output = []
    for key in input.keys():
        if isinstance(input[key], dict):
            output.append((key, get_structured_array_dtype(input[key])))
        elif np.isscalar(input[key]):
            try:
                output.append((key, input[key].dtype.descr[-1][-1]))
            except:
                output.append((key, type(input[key])))
        else:
            try:
                output.append((key, input[key].dtype.descr[-1][-1], input[key].shape))
            except:
                output.append((key, type(input[key][0]), len(input[key])))                    
    return output
