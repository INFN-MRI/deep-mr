"""Sub-package containing parameter inference routines."""

from . import dictmatch as _dictmatch
from . import perk as _perk

from .dictmatch import *  # noqa
from .perk import *  # noqa

__all__ = []
__all__.extend(_dictmatch.__all__)
__all__.extend(_perk.__all__)
