"""Local Low Rank denoising prior."""

__all__ = ["LLRPrior", "llr_denoise"]

import numpy as np

import torch
import torch.nn as nn

from .. import _signal

class LLRPrior(nn.Module):
    r"""
    Local Low Rank denoising.

    The solution is available in closed-form, thus the denoiser is cheap to compute.

    Attributes
    ----------
    ndim : int,
        Number of spatial dimensions.
    W : int
        Patch size (assume isotropic).
    S : int
        Patch stride (assume isotropic).
    rand_shift : bool, optional
        If True, randomly shift across spatial dimensions before denoising.
    axis : bool, optional
        Axis assumed as coefficient axis (e.g., coils or contrasts).
        If not provided, use first axis to the left of spatial dimensions.
    device : str, optional 
        Device on which the wavelet transform is computed. Default is ``None``.

    """

    def __init__(self, ndim, W, S=None, rand_shift=True, axis=None, device=None):
        super().__init__()
        self.ndim = ndim
        self.W = [W] * ndim
        if S is None:
            self.S = [W] * ndim
        else:
            self.S = [S] * ndim
        self.rand_shift = rand_shift
        if axis is None:
            self.axis = -self.ndim-1
        else:
            self.axis = axis
        self.device = device

    def forward(self, x, ths=0.1):
        """
        Run the model on a noisy image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy image.
        ths : int, float, torch.Tensor, optional
            Thresholding parameter.
            The default is 0.1.

        """
        # cast to torch if required
        if isinstance(x, np.ndarray):
            isnumpy = True
            x = torch.as_tensor(x)
        else:
            isnumpy = False
                        
        # default device
        idevice = x.device
        if self.device is None:
            device = idevice
        else:
            device = self.device
        x = x.to(device)
            
        # circshift randomly
        if self.rand_shift is True:
            shift = tuple(np.random.randint(0, self.W, size=self.ndim))
            axes = tuple(range(-self.ndim, 0))
            x = torch.roll(x, shift, axes)
        
        # reshape to (..., ncoeff, ny, nx), (..., ncoeff, nz, ny, nx)
        x = x.swapaxes(self.axis, -self.ndim-1)
        x0shape = x.shape
        x = x.reshape(-1, *x0shape[-self.ndim-1:])
        x1shape = x.shape
        
        # build patches
        patches = _signal.tensor2patches(x, self.W, self.S)
        pshape = patches.shape
        patches = patches.reshape(*pshape[:1], -1, int(np.prod(pshape[-self.ndim:])))
                
        # perform SVD and soft-threshold S matrix
        u, s, vh = torch.linalg.svd(patches, full_matrices=False)
        s_st = _soft_thresh(s, ths)
        patches = u * s_st[..., None, :] @ vh
        patches = patches.reshape(*pshape)
        output = _signal.patches2tensor(patches, x1shape[-self.ndim:], self.W, self.S)
        output = output.reshape(x0shape)
        output = output.swapaxes(self.axis, -self.ndim-1)
        
        # randshift back
        if self.rand_shift is True:
            shift = tuple([-s for s in shift])
            output = torch.roll(output, shift, axes)
            
        # cast back to numpy if requried
        if isnumpy:
            output = output.numpy(force=True)
        else:
            output = output.to(idevice)

        return output
      
def llr_denoise(input, ndim, ths, w, s=None, rand_shift=True, axis=None, device=None):
    """
    Apply Local Low Rank denoising.

    The solution is available in closed-form, thus the denoiser is cheap to compute.

    Attributes
    ----------
    ndim : int,
        Number of spatial dimensions.
    W : int
        Patch size (assume isotropic).
    S : int
        Patch stride (assume isotropic).
    rand_shift : bool, optional
        If True, randomly shift across spatial dimensions before denoising.
    axis : bool, optional
        Axis assumed as coefficient axis (e.g., coils or contrasts).
        If not provided, use first axis to the left of spatial dimensions.
    device : str, optional 
        Device on which the wavelet transform is computed. Default is ``None``.
    
    Returns
    -------
    output : np.ndarray | torch.Tensor
        Denoised image of shape (..., n_ndim, ..., n_0).
    
    """
    LLR = LLRPrior(ndim, w, s, rand_shift, axis, device)
    return LLR(input, ths)


# %% local utils
def _soft_thresh(input, ths):
    mask1 = input > ths
    mask2 = input < -ths
    return (
        mask1.float() * (-ths)
        + mask1.float() * input
        + mask2.float() * ths
        + mask2.float() * input
    )
