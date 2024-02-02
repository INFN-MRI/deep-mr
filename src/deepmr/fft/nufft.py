"""FFT subroutines."""

__all__ = ["plan_nufft", "apply_nufft", "apply_nufft_adj"]

import numpy as np
import torch

from .._signal import resize as _resize
from .._signal import interp as _interp

from . import fft as _fft


def plan_nufft(coord, shape, width=3, oversamp=1.125, device="cpu"):
    """
    Precompute NUFFT object.

    Parameters
    ----------
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5, 0.5)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    width : int | Iterable[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``3``.
    oversamp : float | Iterable[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.125``.
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
    # get parameters
    ndim = coord.shape[-1]

    if np.isscalar(width):
        width = np.asarray([width] * ndim, dtype=np.int16)
    else:
        width = np.asarray(width, dtype=np.int16)

    # calculate Kaiser-Bessel beta parameter
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.array(shape, dtype=np.int16)

    os_shape = _get_oversamp_shape(shape, oversamp, ndim)

    # adjust coordinates
    coord = _scale_coord(coord, shape, oversamp)

    # compute interpolator
    sparse_coeff = _interp.prepare_interpolator(coord, os_shape, width, beta, device)

    return {'ndim': ndim,
            'oversamp': oversamp,
            'width': width,
            'beta': beta,
            'os_shape': os_shape,
            'oshape': shape,
            'sparse_coeff': sparse_coeff,
            'device': device}


def apply_nufft(image, interpolator):
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    os_shape = interpolator['os_shape']
    sparse_coeff = interpolator['sparse_coeff']
    device = interpolator['device']
    basis = interpolator['basis']

    # Transpose and conjugate subspace basis (coefficient -> time)
    if basis is not None:
        adjoint_basis = basis.conj().T
    else:
        adjoint_basis = None

    # Copy input to avoid original data modification
    image = image.clone()

    # Original device
    odevice = image.device

    # Offload to computational device
    image = image.to(device)

    # Apodize
    _apodize(image, ndim, oversamp, width, beta)

    # Zero-pad
    image /= np.prod(image.shape[-ndim:])**0.5
    image = _resize.resize(image, list(image.shape[:-ndim]) + os_shape)

    # FFT
    kdata = _fft.fft(image, axes=range(-ndim, 0), norm=None)

    # Interpolate
    kdata = _interp.interpolate(kdata, sparse_coeff, adjoint_basis)
    kdata /= np.prod(width)

    # Bring back to original device
    kdata = kdata.to(odevice)

    return kdata


def apply_nufft_adj(kdata, interpolator):
    # unpack interpolator
    ndim = interpolator['ndim']
    oversamp = interpolator['oversamp']
    width = interpolator['width']
    beta = interpolator['beta']
    os_shape = interpolator['os_shape']
    oshape = interpolator['oshape']
    sparse_coeff = interpolator['sparse_coeff']
    device = interpolator['device']
    basis = interpolator['basis']

    # Original device
    odevice = kdata.device

    # Offload to computational device
    kdata = kdata.to(device)

    # Gridding
    kdata = _interp.gridding(kdata, sparse_coeff, basis)
    kdata /= np.prod(width)

    # IFFT
    image = _fft.ifft(kdata, axes=range(-ndim, 0), norm=None)

    # Crop
    image = _resize.resize(image, list(image.shape[:-ndim]) + oshape.tolist())
    image *= np.prod(os_shape[-ndim:]) / np.prod(oshape[-ndim:])**0.5

    # Apodize
    __apodize(image, ndim, oversamp, width, beta)

    # Bring back to original device
    kdata = kdata.to(odevice)
    image = image.to(odevice)

    return image

# %% local utils
def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.clone()
    for i in range(-ndim, 0):
        scale = np.ceil(oversamp * shape[i]) / shape[i]
        shift = np.ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output

def _get_oversamp_shape(shape, oversamp, ndim):
    return list(shape)[:-ndim] + [np.ceil(oversamp * i).astype(np.int16) for i in shape[-ndim:]]

def _apodize(data_in, ndim, oversamp, width, beta):
    data_out = data_in
    for axis in range(-ndim, 0):
        i = data_out.shape[axis]
        os_i = np.ceil(oversamp * i)
        idx = torch.arange(  # pylint: disable=no-member
            i, dtype=torch.float32)

        # Calculate apodization
        apod = (beta[axis]**2 - (np.pi * width[axis] * (idx - i // 2) / os_i)**2)**0.5
        apod /= torch.sinh(apod)  # pylint: disable=no-member
        data_out *= apod.reshape([i] + [1] * (-axis - 1))

    return data_out