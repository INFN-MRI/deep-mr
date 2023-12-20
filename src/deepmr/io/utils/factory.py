"""
Utils for custom dataclass constructions.

Reference:
    1. https://github.com/pytorch/pytorch/issues/79197

"""

__all__ = ["arraylike_factory"]

from dataclasses import field
from typing import get_args


# %% implementation
def arraylike_factory(input, value=0):
    """
    Initialize an array-like (list, tuple, ...) field of a custom dataclass.

    Args:
        input: the  array-like field to be initialized.
        value: value used to fill the initialized array (defaults to 0).
    """
    return field(default_factory=lambda: init_annotated(input, value))


def init_annotated(input, value):
    # get length and type
    tmp, length = get_args(input)
    type = get_args(tmp)[0]

    return [type(value) for n in range(length)]
