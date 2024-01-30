"""Array shape manipulation routines."""

__all__ = ["resize", "resample"]

import numpy as np
import torch

from .filter import fermi

def resize(input, oshape):
    """
    Resize with zero-padding or cropping.
    
    Adapted from SigPy [1].
    
    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (..., ishape).
    oshape : Iterable
        Output shape.

    Returns
    -------
    output : torch.Tensor
        Zero-padded or cropped tensor of shape (..., oshape).
        
    Examples
    --------
    >>> import torch
    >>> import deepmr
    
    We can pad tensors to desired shape:
        
    >>> x = torch.tensor([0, 1, 0])
    >>> y = deepmr.resize(x, [5])
    >>> y
    tensor([0, 0, 1, 0, 0])
    
    Batch dimensions are automatically expanded (pad will be applied starting from rightmost dimension):
        
    >>> x = torch.tensor([0, 1, 0])[None, ...]
    >>> x.shape
    torch.Size([1, 3])
    >>> y = deepmr.resize(x, [5]) # len(oshape) == 1
    >>> y.shape
    torch.Size([1, 5])
    
    Similarly, if oshape is smaller than ishape, the tensor will be cropped:
        
    >>> x = torch.tensor([0, 0, 1, 0, 0])
    >>> y = deepmr.resize(x, [3])
    >>> y
    tensor([0, 1, 0])
    
    Again, batch dimensions are automatically expanded:
    
    >>> x = torch.tensor([0, 0, 1, 0, 0])[None, ...]
    >>> x.shape
    torch.Size([1, 5])
    >>> y = deepmr.resize(x, [3]) # len(oshape) == 1
    >>> y.shape
    torch.Size([1, 3])
            
    References
    ----------
    [1] https://github.com/mikgroup/sigpy/blob/main/sigpy/util.py

    """
    if isinstance(oshape, int):
        oshape = [oshape]
        
    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    # shift not supported for now
    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [
        min(i - si, o - so)
        for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)
    ]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape1, dtype=input.dtype, device=input.device)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output

def resample(input, oshape, filt=True, polysmooth=False):
    """
    Resample a n-dimensional signal.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (..., ishape).
    oshape : Iterable
        Output shape.
    filt : bool, optional
        If True and signal is upsampled (i.e., any(oshape > ishape)),
        apply Fermi filter to limit ringing. 
        The default is True.
    polysmooth : bool, optional
        If true, perform polynomial smoothing. 
        The default is False. !!! NOT IMPLEMENTED YET !!!

    Returns
    -------
    output : torch.Tensor
        Resampled tensor of shape (..., oshape).

    """
    if isinstance(oshape, int):
        oshape = [oshape]
        
    # first, get number of dimensions
    ndim = len(oshape)
    axes = list(range(-ndim, 0))
    isreal = torch.isreal(input).all()
    
    # take fourier transform along last ndim axes
    freq = _fftc(input, axes)
    
    # get initial and final shapes
    ishape1, oshape1 = _expand_shapes(input.shape, oshape)
    
    # build filter
    if filt and np.any(np.asarray(oshape1) > np.asarray(ishape1)):
        size = np.max(oshape1)
        width = np.min(oshape1)
        filt = fermi(ndim, size, width)
        filt = resize(filt, oshape1) # crop to match dimension
    else:
        filt = None
    
    # resize in frequency space
    freq = resize(freq, oshape1)
    
    # if required, apply filtering
    if filt is not None:
        freq *= filt
        
    # transform back
    output = _ifftc(freq, axes)
    
    # smooth
    if polysmooth:
        print("Polynomial smoothing not implemented yet; skipping")
        
    # take magnitude if original signal was real
    if isreal:
        output = abs(output)
    
    return output

# %% subroutines
def _expand_shapes(*shapes):
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape for shape in shapes]

    return tuple(shapes_exp)

def _fftc(x, ax):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=ax), dim=ax, norm='ortho'), dim=ax) 

def _ifftc(x, ax):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=ax), dim=ax, norm='ortho'), dim=ax) 


