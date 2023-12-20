"""
Utils for type casting.

Reference:
    1. https://github.com/pytorch/pytorch/issues/79197

"""
__all__ = ["cast_to"]

from typing import List

from deepmr.io.utils.checking import is_builtin_type


# %% implementation
def cast_to(input, type):
    """
    Cast input to the desired type.

    For Lists, iterate over list content and convert each element.

    Args:
        input: Input value.
        type: Type of the return value.

    Returns:
        converted value of type 'type'.
    """
    if input is not None and is_builtin_type(input) and type is not str:
        if isinstance(input, List):
            return [type(element) for element in input]
        else:
            return type(input)
    return input
