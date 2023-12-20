"""Utils for custom objects deseralization."""

__all__ = ["deserialize", "get_dict_from_structured_array"]

from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt


# %% implementation
def deserialize(
    cls, input: Union[np.void, npt.NDArray[np.void]]
) -> Union[object, List[object]]:
    """
    Deerialize a custom object.

    The object class must have a "deserialize" method.
    For a List of inputs, iterate over the elements and
    return a List of custom objects.

    Args:
        cls: class to be deserialized.
        input (np.ndarray, npt.NDArray[np.void]): input structured array
            or List of structured arrays.

    Returns:
        output object or List of objects.
    """
    if isinstance(input, np.ndarray) and isinstance(input[0], np.void):
        return deserialize_list(cls, input)
    else:
        return deserialize_object(cls, input)


def deserialize_object(cls, input):
    return cls.deserialize(input)


def deserialize_list(cls, input):
    return [cls.deserialize(el) for el in input]


def get_dict_from_structured_array(input: np.void) -> Dict:
    """
    Create a dictionary from a structured array element.

    Args:
        input (np.void): Input element of a structed array.

    Returns:
        dict: dictionary corresponding to the input element.
    """
    # get dtype
    dtype = input.dtype
    descr = dtype.descr

    # build output
    output = {}
    for n in range(len(input)):
        if len(dtype[n]) > 0:  # if the element is a structured array, use recursion
            output[descr[n][0]] = get_dict_from_structured_array(input[n])
        else:
            output[descr[n][0]] = input[n]
    return output
