"""Sampling pattern planning subroutines."""

__all__ = ["plan_sampling"]

from dataclasses import dataclass

import numpy as np
import numba as nb
import torch

from .. import backend


def plan_sampling(indexes, shape, device="cpu"):
    """
    Precompute interpolator object.

    Parameters
    ----------
    indexes : torch.Tensor
        Non-zero data entries of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Indexes must be normalized between ``(0, N_i)``, with ``i=x,y,z``
        being the number of voxels along the corresponding axis.
    shape : int | Iterable[int]
        Grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.

    Returns
    -------
    interpolator : dict
        Structure containing sparse sampling matrix:

            * index (``torch.Tensor[int]``): indexes of the non-zero entries of interpolator sparse matrix of shape (ndim, ncoord, width).
            * dshape (``Iterable[int]``): grid shape of shape (ndim,). Order of axes is (z, y, x).
            * ishape (``Iterable[int]``): interpolator shape (ncontrasts, nview, nsamples)
            * ndim (``int``): number of spatial dimensions.
            * device (``str``): computational device.

    Notes
    -----
    Sampling mask axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, is
    assumed to be ``(z, y, x)``.

    Sampling tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # convert to tensor if nececessary
    indexes = torch.as_tensor(indexes, dtype=torch.int16)

    # expand singleton dimensions
    ishape = list(indexes.shape[:-1])
    ndim = indexes.shape[-1]
    
    while len(ishape) < 3:
        ishape = [1] + ishape

    nframes = ishape[0]
    ishape = ishape[1:]

    # parse input sizes
    npts = np.prod(ishape)

    # expand
    if np.isscalar(shape):
        shape = ndim * [shape]

    # revert axis (z, y, x) -> (x, y, z)
    shape = shape[::-1]

    # arg reshape
    indexes = indexes.reshape([nframes, npts, ndim])
    indexes = indexes.permute(2, 0, 1)

    # send to numba
    index = [backend.pytorch2numba(idx) for idx in indexes]

    # transform to tuples
    index = tuple(index)
    dshape = tuple(shape)
    ishape = tuple(ishape)

    return Sampling(index, dshape, ishape, ndim, device)

# %% subroutines
@dataclass
class Sampling:
    index: tuple
    dshape: tuple
    ishape: tuple
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

            # zero-copy to torch
            self.index = [backend.numba2pytorch(idx) for idx in self.index]

            # dispatch
            self.index = [idx.to(device) for idx in self.index]

            # zero-copy to numba
            self.index = [backend.pytorch2numba(idx) for idx in self.index]

            self.index = tuple(self.index)
            self.device = device
