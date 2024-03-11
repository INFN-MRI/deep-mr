"""Sub-package containing optimization routines.

DeepMR provides optimization routines for compressed sensing.
All the routines are based on the excellent Deep Inverse (https://github.com/deepinv/deepinv) package.

"""

from . import data_fidelity as _data_fidelity
# from . import admm as _admm

# from .admm import *  # noqa
from .data_fidelity import * # noqa
from deepinv.optim.optim_iterators import OptimIterator # noqa
from deepinv.optim.optim_iterators import GDIteration # noqa
from deepinv.optim.optim_iterators import PGDIteration # noqa
from deepinv.optim.optim_iterators import CPIteration # noqa
from deepinv.optim.optim_iterators import DRSIteration # noqa
from deepinv.optim.optim_iterators import HQSIteration # noqa

__all__ = []
__all__.extend(_data_fidelity.__all__)
__all__.extend(["OptimIterator, GDIteration", "PGDIteration", "CPIteration", "DRSIteration", "HQSIteration"])
