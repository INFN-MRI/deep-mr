"""Sub-package containing regularization routines.

DeepMR provides regularization routines for plug-n-play compressed sensing.

"""


from . import llr as _llr
from . import wavelet as _wavelet
from . import tv as _tv
from . import tgv as _tgv

from .llr import *  # noqa
from .wavelet import *  # noqa
from .tv import *  # noqa
from .tgv import *  # noqa


__all__ = []
__all__.extend(_llr.__all__)
__all__.extend(_wavelet.__all__)
__all__.extend(_tv.__all__)
__all__.extend(_tgv.__all__)
