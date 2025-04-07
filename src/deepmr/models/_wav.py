"""Wavelet denoiser"""

__all__ = ["ProxWave"]

from mrops import _sigpy as sp
from mrinufft._array_compat import with_torch
from deepinv.models import WaveletDenoiser


class ProxWave(sp.prox.Prox):
    """
    Wavelet denoiser.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input shape.
    ndim : int
        Spatial dimensions.
    wv : str, optional
        Wavelet type. The default is ``"db4"``.
    level : int, optional
        Decomposition level. The default is ``3``.

    """

    def __init__(self, shape, ndim, wv="db4", level=3):
        self.W = WaveletDenoiser(
            wv=wv,
            wvdim=ndim,
            level=level,
        )
        super().__init__(shape)

    def _prox(self, alpha, input):
        return _wavelet_denoise(self.W, input, alpha)


# %% utils
@with_torch
def _wavelet_denoise(W, x, lamda):
    ishape = x.shape
    ndim = W.dimension
    x = x.reshape(-1, *x.shape[-ndim:])
    x = W(x.real, lamda) + 1j * W(x.imag, lamda)
    return x.reshape(*ishape)
