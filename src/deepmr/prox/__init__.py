"""Sub-package containing regularization routines.

DeepMR provides regularization routines for compressed sensing.
All the routines are based on the excellent Deep Inverse (https://github.com/deepinv/deepinv) package.

"""


from . import llr as _llr
from . import wavelet as _wavelet
from . import tv as _tv

from .llr import * # noqa
from .wavelet import *  # noqa
from .tv import * # noqa


__all__ = []
__all__.extend(_llr.__all__)
__all__.extend(_wavelet.__all__)
__all__.extend(_tv.__all__)
