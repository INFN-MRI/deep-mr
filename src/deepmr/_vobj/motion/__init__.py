""""Motion patterns generation routines"""

from . import rigid as _rigid

from .rigid import *  # noqa

__all__ = []
__all__.extend(_rigid.__all__)
