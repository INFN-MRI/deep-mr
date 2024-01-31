"""Interpolator planning subroutines."""

__all__ = ["plan_interpolator"]

from dataclasses import dataclass

import numpy as np
import numba as nb
import torch

from . import backend

def plan_interpolator(coord, shape, width, beta, device="cpu"):
    """
    Precompute interpolator object.

    Parameters
    ----------
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5, 0.5)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    width : int | Iterable[int]
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
    beta : float | Iterable[float]
        Kaiser-Bessel beta parameter of shape ``(ndim,)``.
        If scalar, it is assumed equal for each axis.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.

    Returns
    -------
    interpolator : dict
        Structure containing sparse interpolator matrix:
            
            * index (``torch.Tensor[int]``): indexes of the non-zero entries of interpolator sparse matrix of shape (ndim, ncoord, width).
            * value (``torch.Tensor[float32]``): values of the non-zero entries of interpolator sparse matrix of shape (ndim, ncoord, width).
            * dshape (``Iterable[int]``): oversample grid shape of shape (ndim,). Order of axes is (z, y, x).
            * ishape (``Iterable[int]``): interpolator shape (ncontrasts, nview, nsamples)
            * ndim (``int``): number of spatial dimensions.
            * device (``str``): computational device.
            
    Notes
    -----
    Non-uniform coordinates axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, kernel width
    and Kaiser Bessel parameters are assumed to be ``(z, y, x)``.
    
    Coordinates tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions 
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:
        
        * ``coord.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``coord.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``
        
    """
    # convert to tensor if nececessary
    coord = torch.as_tensor(coord, dtype=torch.float32)
    
    # expand singleton dimensions
    ishape = coord.shape[:-1]
    ndim = coord.shape[-1]
    if len(coord.shape) < 4:
        nframes = 1
    else:
        nframes = ishape[0]
        ishape = ishape[1:]
        
    # parse input sizes
    npts = np.prod(ishape)
    
    # expand
    if np.isscalar(shape):
        shape = ndim * [shape]
    if np.isscalar(width):
        width = ndim * [width]
    if np.isscalar(beta):
        beta = ndim * [beta]
        
    # revert axis (z, y, x) -> (x, y, z)
    shape = shape[::-1]
    width = width[::-1]
    beta = beta[::-1]
        
    # compute kernel scaling
    scale = _get_kernel_scaling(beta, width)

    # arg reshape
    coord = coord.reshape([nframes*npts, ndim]).T
    
    # preallocate interpolator
    index = []
    value = []

    for i in range(ndim): # (x, y, z)
        # kernel value
        value.append(torch.zeros((nframes*npts, width[i]), dtype=torch.float32))
        
        # kernel index
        index.append(torch.zeros((nframes*npts, width[i]), dtype=torch.int32)) 

    # actual precomputation
    for i in range(-ndim, 0): # (z, y, x)
        _do_prepare_interpolator(value[i], index[i], coord[i] * shape[i], width[i], beta[i], shape[i])

    # reformat for output
    for i in range(ndim):
        index[i] = index[i].reshape([nframes, npts, width[i]]).to(device)
        value[i] = value[i].reshape([nframes, npts, width[i]]).to(device)
        
    # send to numba
    index = [backend.pytorch2numba(idx) for idx in index]
    value = [backend.pytorch2numba(val) for val in value]
    
    # transform to tuples
    index = tuple(index)
    value = tuple(value)
    dshape = tuple(shape)
    ishape = tuple(ishape)

    return Interpolator(index, value, dshape, ishape, scale, ndim, device)

# %% subroutines
@dataclass
class Interpolator:
    index: tuple
    value: tuple
    dshape: tuple
    ishape: tuple
    scale: float
    ndim: int
    device: str
    
    def to(self, device):
        """
        Dispatch internal attributes to selected device.
        
        Parameters
        ----------
        device : str
            Computational device ("cpu" or "cuda:n", with n=0, 1,...nGPUs).

        """
        if device != self.device:
            self.index = list(self.index)
            self.value = list(self.value)
            
            # zero-copy to torch
            self.index = [backend.numba2pytorch(idx) for idx in self.index]
            self.value = [backend.numba2pytorch(val) for val in self.value]
            
            # dispatch
            self.index = [idx.to(device) for idx in self.index]
            self.value = [val.to(device) for val in self.value]
            
            # zero-copy to numba
            self.index = [backend.pytorch2numba(idx) for idx in self.index]
            self.value = [backend.pytorch2numba(val) for val in self.value]
                
            self.index = tuple(self.index)
            self.value = tuple(self.value)
            self.device = device
        
def _get_kernel_scaling(beta, width):
    # init kernel centered on k-space node
    value = []
    
    # fill the three axes
    for ax in range(len(width)):
        start = np.ceil(-width[ax] / 2)
        value.append(np.array([_kaiser_bessel_kernel((start + el) / (width[ax] / 2), beta[ax]) for el in range(width[ax])]))
                    
    value = np.stack(np.meshgrid(*value), axis=0).prod(axis=0)
    
    return value.sum()
            
@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _kaiser_bessel_kernel(x, beta):
    if abs(x) > 1:
        return 0

    x = beta * (1 - x**2)**0.5
    t = x / 3.75
    if x < 3.75:
        return 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
            1.2067492 * t**6 + 0.2659732 * t**8 +\
            0.0360768 * t**10 + 0.0045813 * t**12
    else:
        return x**-0.5 * np.exp(x) * (
            0.39894228 + 0.01328592 * t**-1 +
            0.00225319 * t**-2 - 0.00157565 * t**-3 +
            0.00916281 * t**-4 - 0.02057706 * t**-5 +
            0.02635537 * t**-6 - 0.01647633 * t**-7 +
            0.00392377 * t**-8)

def _prepare_interpolator():
    """Subroutines for interpolator planning."""
    kernel = _kaiser_bessel_kernel

    @nb.njit(fastmath=True, parallel=True)  # pragma: no cover
    def _prepare_interpolator(interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape):

        # get sizes
        npts = coord.shape[0]
        kernel_width = interp_index.shape[-1]

        for i in nb.prange(npts):  # pylint: disable=not-an-iterable
            x_0 = np.ceil(coord[i] - kernel_width / 2)

            for x_i in range(kernel_width):
                val = kernel(((x_0 + x_i) - coord[i]) / (kernel_width / 2), kernel_param)

                # save interpolator
                interp_value[i, x_i] = val
                interp_index[i, x_i] = (x_0 + x_i) % grid_shape

    return _prepare_interpolator

def _do_prepare_interpolator(interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape): # noqa
    """Preparation routine wrapper.""" 
    interp_value = backend.pytorch2numba(interp_value)
    interp_index = backend.pytorch2numba(interp_index)
    coord = backend.pytorch2numba(coord)

    _prepare_interpolator(interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape)

    interp_value = backend.numba2pytorch(interp_value)
    interp_index = backend.numba2pytorch(interp_index, requires_grad=False)
    coord = backend.numba2pytorch(coord)
