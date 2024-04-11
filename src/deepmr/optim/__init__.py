"""Sub-package containing optimization routines.

DeepMR provides optimization routines for compressed sensing and
network unfolding.

"""

from . import spectral as _spectral
from . import cg as _cg
from . import admm as _admm
from . import pgd as _pgd
from . import lstsq as _lstsq

from .spectral import *  # noqa
from .cg import *  # noqa
from .admm import *  # noqa
from .pgd import *  # noqa
from .lstsq import *  # noqa

__all__ = []
__all__.extend(_spectral.__all__)
__all__.extend(_cg.__all__)
__all__.extend(_admm.__all__)
__all__.extend(_pgd.__all__)
__all__.extend(_lstsq.__all__)
