"""Sub-package containing optimization routines.

DeepMR provides optimization routines for compressed sensing and
network unfolding.

"""

from . import pgd as _pgd
from . import admm as _admm
from . import lstsq as _lstsq

from .pgd import *  # noqa
from .admm import *  # noqa
from .lstsq import *  # noqa

__all__ = []
__all__.extend(_pgd.__all__)
__all__.extend(_admm.__all__)
# __all__.extend(_lstsq.__all__)
