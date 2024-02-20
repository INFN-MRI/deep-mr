""""Field generation routines"""

from . import coil as _coil

from .coil import *  # noqa

__all__ = []
__all__.extend(_coil.__all__)