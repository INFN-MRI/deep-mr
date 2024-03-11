"""Wavelet denoising prior."""

__all__ = ["WaveletPrior", "wavelet_denoise"]

import numpy as np
import torch

from deepinv.optim.prior import PnP
from deepinv.optim.prior import WaveletPrior as _WaveletPrior


def WaveletPrior(ndim, wv="db4", device=None, p=1, level=3, *args, **kwargs):
    r"""
    Wavelet prior :math:`\reg{x} = \|\Psi x\|_{p}`.
    
    This is simply a thin wrapper around ``deepinv.optim.prior.WaveletPrior``
    used to force user to specify the input spatial dimensions.

    :math:`\Psi` is an orthonormal wavelet transform, and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
    :math:`p=0`, :math:`p=1`, or :math:`p=\infty`.

    Notes
    -----
    Following common practice in signal processing, only detail coefficients are regularized, and the approximation
    coefficients are left untouched.

    Warning
    -------
    For 3D data, the computational complexity of the wavelet transform cubically with the size of the support. For
    large 3D data, it is recommended to use wavelets with small support (e.g. db1 to db4).

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions, can be either ``2`` or ``3``. 
    wv : str, optional 
        Wavelet name to choose among those available in `pywt <https://pywavelets.readthedocs.io/en/latest/>`_. 
        Default is ``"db4"``.
    device : str, optional 
        Device on which the wavelet transform is computed. Default is ``None``.
    p : float, optional
        :math:`p`-norm of the prior. Default is ``1``.
    level: int, optional
        Level of the wavelet transform. Default is ``None``.
        
    """
    return PnP(denoiser=ComplexWaveletDenoiser(ndim, wv, device, p, level, *args, **kwargs))


def wavelet_denoise(input, ndim, ths=0.1, wv="db4", device=None, p=1, level=3):
    r"""
    Apply wavelet denoising as :math:`\reg{x} = \|\Psi x\|_{p}`.
    
    This is simply a thin wrapper around ``deepinv.optim.prior.WaveletPrior``
    used to force user to specify the input spatial dimensions.

    :math:`\Psi` is an orthonormal wavelet transform, and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
    :math:`p=0`, :math:`p=1`, or :math:`p=\infty`.

    Notes
    -----
    Following common practice in signal processing, only detail coefficients are regularized, and the approximation
    coefficients are left untouched.

    Warning
    -------
    For 3D data, the computational complexity of the wavelet transform cubically with the size of the support. For
    large 3D data, it is recommended to use wavelets with small support (e.g. db1 to db4).

    Arguments
    ---------
    input : np.ndarray | torch.Tensor
        Input image of shape (..., n_ndim, ..., n_0).
    ndim : int
        Number of spatial dimensions, can be either ``2`` or ``3``. 
    ths : float, optional
        Denoise threshold. Degault is ``0.1``.
    wv : str, optional 
        Wavelet name to choose among those available in `pywt <https://pywavelets.readthedocs.io/en/latest/>`_. 
        Default is ``"db4"``.
    device : str, optional 
        Device on which the wavelet transform is computed. Default is ``"cpu"``.
    p : float, optional
        :math:`p`-norm of the prior. Default is ``1``.
    level: int, optional
        Level of the wavelet transform. Default is ``None``.
        
    Returns
    -------
    output : np.ndarray | torch.Tensor
        Denoised image of shape (..., n_ndim, ..., n_0).
    
    """
    W = ComplexWaveletDenoiser(ndim, wv, device, p, level)
    return W(input, ths)


# %% local utils
class ComplexWaveletDenoiser(torch.nn.Module):
    def __init__(self, ndim, wv, device, p, level, *args, **kwargs):
        super().__init__()
        self.denoiser = _WaveletPrior(level=level, wv=wv, p=p, device=device, wvdim=ndim, *args, **kwargs)
        self.denoiser.device = device
    
    def forward(self, input, ths):
    
        # cast to numpy if required
        if isinstance(input, np.ndarray):
            isnumpy = True
            input = torch.as_tensor(input)
        else:
            isnumpy = False
            
        # get complex
        if torch.is_complex(input):
            iscomplex = True
        else:
            iscomplex = False
            
        # default device
        idevice = input.device
        if self.denoiser.device is None:
            device = idevice
        else:
            device = self.denoiser.device
            
        # get input shape
        ndim = self.denoiser.wvdim
        ishape = input.shape
        
        # reshape for computation
        input = input.reshape(-1, *ishape[-ndim:])
        if iscomplex:
            input = torch.stack((input.real, input.imag), axis=1)
            input = input.reshape(-1, *ishape[-ndim:])
        
        # apply denoising
        output = self.denoiser.prox(input[:, None, ...].to(device), ths).to(idevice)  # perform the denoising on the real-valued tensor
        
        # reshape back
        if iscomplex:
            output = output[::2, ...] + 1j * output[1::2, ...]  # build the denoised complex data
        output = output.reshape(ishape)
        
        # cast back to numpy if requried
        if isnumpy:
            output = output.numpy(force=True)
        
        return output
