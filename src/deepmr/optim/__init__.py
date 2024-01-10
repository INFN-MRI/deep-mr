"""Sub-package containing optimization routines.

DeepMR provides optimization routines for compressed sensing.
All the routines are based on the excellent Deep Inverse (https://github.com/deepinv/deepinv) package.

"""

from . import admm as _admm
from . import pgd as _pgd

from .admm import * # noqa
from .pgd import *  # noqa

__all__ = []
__all__.extend(_admm.__all__)
__all__.extend(_pgd.__all__)

