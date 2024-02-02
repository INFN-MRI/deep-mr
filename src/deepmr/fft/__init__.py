"""Sub-package containing Fast Fourier transform routines.

FFT routines include:
    
    * centered n-dimensional FFT and iFFT;
    * n-dimensional sparse uniform FFT/iFFT with embedded low rank subspace projection;
    * n-dimensional NUFFT with embedded low rank subspace projection.

"""
from . import fft as _fft

from .fft import *  # noqa

__all__ = []
__all__.extend(_fft.__all__)
