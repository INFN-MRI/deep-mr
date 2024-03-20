"""Sub-package containing Linear operators for MR imaging.

Each operator is derived from :func:`torch.nn.Module`.
As a consequence, they can be composed to build neural network architectures.

Currently implemented linear operator include:
    
* centered n-dimensional FFT and iFFT with masking and low rank subspace projection;
* n-dimensional NUFFT with embedded low rank subspace projection;
* coil combination and projection operators.

"""

from . import coil as _base
from . import coil as _coil
from . import fft as _fft
from . import nufft as _nufft

from .base import *  # noqa
from .coil import *  # noqa
from .fft import *  # noqa
from .nufft import *  # noqa

__all__ = []
__all__.extend(_base.__all__)
__all__.extend(_coil.__all__)
__all__.extend(_fft.__all__)
__all__.extend(_nufft.__all__)
