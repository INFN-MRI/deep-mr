"""Data compression routines."""

__all__ = ["rss", "svd"]

import numpy as np
import torch

def rss(input, axis=None, keepdim=False):
    """
    Perform root sum-of-squares combination of a signal.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input signal (real- or complex-valued).
    axis : int, optional
        Combination axis.  If ``None``, combine along all dimensions,
        reducing to a scalar. The default is ``None``.
    keepdim : bool, optional
        If ``True``, maintain the combined axis as a singleton dimension. 
        The default is ``False`` (squeeze the combination axis).

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Real-valued output combined signal.
        
    Examples
    --------
    >>> import torch
    >>> import deepmr
    
    Generate an example signal:
        
    >>> signal = torch.ones(10, 4, 4)
    
    We can compute the rss of all signal elements as:
        
    >>> output = deepmr.rss(signal)
    >>> output
    tensor(12.6491)
    
    We can compute rss along the first axis only (i.e., coil combination) as:
        
    >>> output = deepmr.rss(signal, axis=0)
    >>> output.shape
    torch.Tensor([4, 4])
    
    The axis can be explicitly maintained instead of squeezed as
    
    >>> output = deepmr.rss(signal, axis=0, keepdim=True)
    >>> output.shape
    torch.Tensor([1, 4, 4])
    

    """
    if axis is None:
        return (input * input.conj()).sum()**0.5
    
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False
        
    output = torch.as_tensor(input)
    output = (output * output.conj()).sum(axis=axis, keepdim=keepdim)**0.5
    
    if isnumpy:
        output = output.numpy()
        
    return output


def svd(input, ncoeff, axis):
    """
    Perform SVD compression of a signal.
    
    The routine returns the SVD subspace basis, the compressed signal
    and the explained variance of the subspace.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input signal (real- or complex-valued).
    ncoeff : int
        Number of subspace coefficients to be retained.
    axis : int
        Compression axis.

    Returns
    -------
    basis : np.ndarray | torch.Tensor
        Subspace basis of shape (input.shape[axis], ncoeff).
    output : np.ndarray | torch.Tensor
        Compressed signal of shape (..., ncoeff, ...).
    explained_variance : float
        Explained variance of the subspace. Values close to 100% 
        indicates that information content of the signal is preserved
        despite compression.

    """
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False
                
    # cast to tensor
    output = torch.as_tensor(input)
    
    # move specified axis in the right position
    if axis != -1:
        output = output[..., None]
        output = output.swapaxes(axis, -1)
        
    
    # fold to (nbatches, nrows, ncols)
    ishape = output.shape
    nrows = int(np.prod(ishape[:-1]))
    ncols = ishape[-1]
    output = output.reshape(nrows, ncols)
    
    # perform svd
    u, s, vh = torch.linalg.svd(output, full_matrices=None)
    
    # compress data
    basis = vh[..., :ncoeff]
    output = output @ basis
    
    # calculate explained variance
    explained_variance = s**2 / (nrows - 1) # (neigenvalues)
    explained_variance = explained_variance / explained_variance.sum() 
    explained_variance = torch.cumsum(explained_variance)[ncoeff-1]
    
    # reshape
    output = output.reshape(*ishape[:-1], ncoeff)
    
    # permute back
    output = output.swapaxes(axis, -1)[..., 0]
    
    if isnumpy:
        output = output.numpy()
        basis = basis.numpy()
        
    return basis, output, explained_variance
    
    
    
