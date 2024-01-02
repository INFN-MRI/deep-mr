"""Generic IO routines."""

from . import hdf5 as _hdf5
from . import matlab as _matlab

from .hdf5 import *  # noqa
from .matlab import *  # noqa

__all__ = []
__all__.extend(_hdf5.__all__)
__all__.extend(_matlab.__all__)
