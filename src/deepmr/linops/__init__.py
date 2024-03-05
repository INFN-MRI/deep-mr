"""Sub-package containing Linear operators for MR imaging.

Each operator is derived from :func:`deepinv.phyisics.LinearPhysics`.
As a consequence, they are :func:`torch.nn.Module` that can be used
inside neural networks

Currently implemented linear operator include:
    
* centered n-dimensional FFT and iFFT with masking and low rank subspace projection;
* n-dimensional sparse uniform FFT/iFFT with embedded low rank subspace projection;
* n-dimensional NUFFT with embedded low rank subspace projection;
* coil combination operators.

"""

from . import coil as _coil
from . import fft as _fft
from . import nufft as _nufft

from .coil import * # noqa
from .fft import *  # noqa
from .nufft import *  # noqa

__all__ = []
__all__.extend(_coil.__all)
__all__.extend(_fft.__all)
__all__.extend(_nufft.__all)
