"""Sub-package containing reconstruction calibration routines.

DeepMR provides calibration routines for image reconstruction.
Currently, these include coil sensitivity estimation for Cartesian and Non-Cartesian
imaging.

"""

from . import espirit as _espirit

from .espirit import *  # noqa

__all__ = []
__all__.extend(_espirit.__all__)