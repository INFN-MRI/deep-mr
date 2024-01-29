"""Array shape manipulation routines."""

__all__ = ["resize", "resample"]

import numpy as np
import torch

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
        Zero-padded or cropped result of shape (..., oshape).
        
    References
    ----------
    [1] https://github.com/mikgroup/sigpy/blob/main/sigpy/util.py

    """
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
    

    Parameters
    ----------
    input : torch.Tensor
        DESCRIPTION.
    oshape : Iterable
        DESCRIPTION.
    filt : bool, optional
        DESCRIPTION. The default is True.
    polysmooth : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    output : torch.Tensor
        DESCRIPTION.

    """
    output = input
    return output

# %% subroutines
def _expand_shapes(*shapes):
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape for shape in shapes]

    return tuple(shapes_exp)

