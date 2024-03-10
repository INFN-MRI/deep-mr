"""Sub-package containing optimization routines.

DeepMR provides optimization routines for compressed sensing.
All the routines are based on the excellent Deep Inverse (https://github.com/deepinv/deepinv) package.

"""

from . import data_fidelity as _data_fidelity
# from . import admm as _admm

# from .admm import *  # noqa
from .data_fidelity import * # noqa

__all__ = []
__all__.extend(_data_fidelity.__all__)
