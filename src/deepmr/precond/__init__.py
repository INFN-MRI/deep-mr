"""Preconditioning sub-package."""

__all__ = []

from . import _polynomial  # noqa

from ._polynomial import *  # noqa

__all__.extend(_polynomial.__all__)
