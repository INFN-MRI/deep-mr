"""
Sub-package containing virtual objects generation routines.

DeepMR contains tools to simulate MR experiments for development and testing.
These tools include numerical phantoms, B0 and B1+ field generators,
random rigid motion generation routines and sampling trajectories (Cartesian and Non-Cartesian).

"""
from . import phantoms as _phantoms

from .phantoms import * # noqa

__all__ = []
__all__.extend(_phantoms.__all__)

