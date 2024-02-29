"""
Sub-package containing virtual objects generation routines.

DeepMR contains tools to simulate MR experiments for development and testing.
These tools include numerical phantoms, B0 and B1+ field generators,
random rigid motion generation routines and sampling trajectories (Cartesian and Non-Cartesian).

"""
from . import phantoms as _phantoms
from . import fields as _fields
from . import sampling as _sampling
from . import trains as _trains
from . import motion as _motion

from .phantoms import *  # noqa
from .fields import *  # noqa
from .sampling import *  # noqa
from .trains import *  # noqa
from .motion import *  # noqa

__all__ = []
__all__.extend(_phantoms.__all__)
__all__.extend(_fields.__all__)
__all__.extend(_sampling.__all__)
__all__.extend(_trains.__all__)
__all__.extend(_motion.__all__)
