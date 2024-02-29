""""Variable parameters train generation routines"""

from . import flip as _flip
from . import phase as _phase

from .flip import *  # noqa
from .phase import *  # noqa

__all__ = []
__all__.extend(_flip.__all__)
__all__.extend(_phase.__all__)
