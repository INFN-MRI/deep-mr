"""Sub-package containing parameter inference routines."""

from . import dictmatch as _dictmatch

from .dictmatch import * # noqa

__all__ = []
__all__.extend(_dictmatch.__all__)
