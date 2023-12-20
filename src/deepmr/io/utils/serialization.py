"""Utils for custom objects seralization."""

__all__ = ["get_structured_array_values", "remove_none", "serialize"]

from typing import Dict, List, Tuple, Union

import numpy as np


# %% implementation
def get_structured_array_values(input: Dict) -> Tuple:
    """
    Get structured array values from a nested dictionary.

    Args:
        input (dict): Input dictionary.

    Returns:
        List: nested list mirroring the input dict sturucture.
    """
    output = []
    for key in input.keys():
        if isinstance(input[key], dict):
            output.append(get_structured_array_values(input[key]))
        else:
            output.append(input[key])
    return tuple(output)


def remove_none(input: Dict) -> Dict:
    """
    Remove 'None' entries from a dictionary.

    For nested dictionaries, the function runs recursively.

    Args:
        input: Input dictionary.

    Returns:
        clean dictionary without 'None' entries.
    """
    if isinstance(input, (list, tuple, set)):
        return type(input)(remove_none(x) for x in input if x is not None)

    elif isinstance(input, dict):
        return type(input)(
            (remove_none(k), remove_none(v))
            for k, v in input.items()
            if k is not None and v is not None
        )

    else:
        return input


def serialize(input: Union[object, List]) -> Union[Dict, np.ndarray]:
    """
    Serialize a custom object.

    The object class must have a "serialize" method.
    For a List of objects, iterate over the elements and
    return a numpy structured array.

    Args:
        input (object, List[object]): input object or List of objects.

    Returns:
        output serialized object or array of serialized objects.
    """
    if isinstance(input, list):
        return serialize_list(input)
    else:
        return serialize_object(input)


def serialize_object(input):
    assert hasattr(input, "serialize") and callable(
        input.serialize
    ), "Error! Input object does not have a 'serialize()' method"
    return input.serialize()


def serialize_list(input):
    assert hasattr(input[0], "serialize") and callable(
        input[0].serialize
    ), "Error! Input object does not have a 'serialize()' method"
    assert hasattr(input[0], "dtype")
    dtype = input[0].dtype
    values = [el.serialize() for el in input]
    return np.asarray(values, dtype=dtype)
