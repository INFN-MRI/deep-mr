"""Sub-package containing reading/writing routines.

DeepMR provides reading and writing routines for common k-space
(ISMRMRD, GEHC, Siemens) and image space (DICOM, NIfTI) formats.

"""

from . import generic as _generic
from . import kspace as _kspace

from .generic import * # noqa
from .kspace import *  # noqa

__all__ = []
__all__.extend(_generic.__all__)
__all__.extend(_kspace.__all__)

