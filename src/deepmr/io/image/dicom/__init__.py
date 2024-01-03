"""DICOM IO routines."""

from . import read as _read
from . import write as _write

from .read import *  # noqa
from .write import *  # noqa

__all__ = []
__all__.extend(_read.__all__)
__all__.extend(_write.__all__)
