"""Sub-package containing reading/writing routines.

DeepMR provides reading and writing routines for common k-space
(ISMRMRD, GEHC, Siemens) and image space (DICOM, NIfTI) formats.

"""

from . import generic as _generic
from . import image as _image
from . import kspace as _kspace
from . import trajectories as _trajectories

from .generic import * # noqa
from .image import *  # noqa
from .kspace import *  # noqa
from .trajectories import *  # noqa

__all__ = []
__all__.extend(_generic.__all__)
__all__.extend(_image.__all__)
__all__.extend(_kspace.__all__)
__all__.extend(_trajectories.__all__)

