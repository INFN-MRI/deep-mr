"""Sparse FFT subroutines."""

__all__ = [
    "prepare_sampling",
    "plan_toeplitz_fft",
    "apply_sparse_fft",
    "apply_sparse_ifft",
    "apply_sparse_fft_selfadj",
]

import gc

import numpy as np
import torch

from .._signal import sparse as _sparse

from . import fft as _fft


def prepare_sampling(indexes, shape, device="cpu"):
    """
    Precompute sparse sampling mask object.

    Parameters
    ----------
    indexes : torch.Tensor
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
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
    Sampled point indexes axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape is
    assumed to be ``(z, y, x)``.

    Indexes tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # get parameters
    ndim = indexes.shape[-1]

    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.array(shape, dtype=np.int16)

    return _sparse.plan_sampling(indexes, shape, device)


def plan_toeplitz_fft(coord, shape, basis=None, device="cpu"):
    """
    Compute spatio-temporal kernel for fast self-adjoint operation.

    Parameters
    ----------
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape[i], 0.5 * shape[i])``,
        with ``i = (z, y, x)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.

    Returns
    -------
    toeplitz_kernel : GramMatrix
        Structure containing Toeplitz kernel (i.e., Fourier transform of system tPSF).

    """
    return _sparse.plan_toeplitz(coord, shape, basis, device)


def apply_sparse_fft(
    image, sampling_mask, basis_adjoint=None, device=None, threadsperblock=128
):
    """
    Apply sparse Fast Fourier Transform.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    sampling_mask : dict
        Pre-formatted sampling mask in sparse COO format.
    basis_adjoint : torch.Tensor, optional
        Adjoint low rank subspace projection operator
        of shape ``(ncoeffs, ncontrasts)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None`` (same as interpolator).
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    kspace : torch.Tensor
        Output sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    """
    # check if it is numpy
    if isinstance(image, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # convert to tensor if nececessary
    image = torch.as_tensor(image)

    # make sure datatype is correct
    if image.dtype in (torch.float16, torch.float32, torch.float64):
        image = image.to(torch.float32)
    else:
        image = image.to(torch.complex64)

    # handle basis
    if basis_adjoint is not None:
        basis_adjoint = torch.as_tensor(basis_adjoint)

        # make sure datatype is correct
        if basis_adjoint.dtype in (torch.float16, torch.float32, torch.float64):
            basis_adjoint = basis_adjoint.to(torch.float32)
        else:
            basis_adjoint = basis_adjoint.to(torch.complex64)

    # cast tp device is necessary
    if device is not None:
        sampling_mask.to(device)

    # unpack plan
    ndim = sampling_mask.ndim
    device = sampling_mask.device

    # Copy input to avoid original data modification
    image = image.clone()

    # Original device
    odevice = image.device

    # Offload to computational device
    image = image.to(device)

    # FFT
    kspace = _fft.fft(image, axes=range(-ndim, 0), norm=None)

    # Interpolate
    kspace = _sparse.apply_sampling(
        kspace, sampling_mask, basis_adjoint, device, threadsperblock
    )

    # Bring back to original device
    kspace = kspace.to(odevice)
    image = image.to(odevice)

    # transform back to numpy if required
    if isnumpy:
        kspace = kspace.numpy(force=True)

    # collect garbage
    gc.collect()

    return kspace


def apply_sparse_ifft(
    kspace, sampling_mask, basis=None, device=None, threadsperblock=128
):
    """
    Apply adjoint Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    kspace : torch.Tensor
        Input sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    sampling_mask : dict
        Pre-formatted sampling mask in sparse COO format.
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None ``(same as interpolator).
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    image : torch.Tensor
        Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    """
    # check if it is numpy
    if isinstance(kspace, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # convert to tensor if nececessary
    kspace = torch.as_tensor(kspace)

    # make sure datatype is correct
    if kspace.dtype in (torch.float16, torch.float32, torch.float64):
        kspace = kspace.to(torch.float32)
    else:
        kspace = kspace.to(torch.complex64)

    # handle basis
    if basis is not None:
        basis = torch.as_tensor(basis)

        # make sure datatype is correct
        if basis.dtype in (torch.float16, torch.float32, torch.float64):
            basis = basis.to(torch.float32)
        else:
            basis = basis.to(torch.complex64)

    # cast to device is necessary
    if device is not None:
        sampling_mask.to(device)

    # unpack plan
    ndim = sampling_mask.ndim
    device = sampling_mask.device

    # Original device
    odevice = kspace.device

    # Offload to computational device
    kspace = kspace.to(device)

    # Gridding
    kspace = _sparse.apply_zerofill(
        kspace, sampling_mask, basis, device, threadsperblock
    )

    # IFFT
    image = _fft.ifft(kspace, axes=range(-ndim, 0), norm=None)

    # Bring back to original device
    kspace = kspace.to(odevice)
    image = image.to(odevice)

    # transform back to numpy if required
    if isnumpy:
        image = image.numpy(force=True)

    # collect garbage
    gc.collect()

    return image


def apply_sparse_fft_selfadj(image, toeplitz_kern, device=None, threadsperblock=128):
    """
    Apply self-adjoint Fast Fourier Transform via Toeplitz convolution.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    toeplitz_kern : GramMatrix
        Pre-calculated Toeplitz kernel.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None ``(same as interpolator).
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    image : torch.Tensor
        Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    """
    # check if it is numpy
    if isinstance(image, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # convert to tensor if nececessary
    image = torch.as_tensor(image)

    # make sure datatype is correct
    if image.dtype in (torch.float16, torch.float32, torch.float64):
        image = image.to(torch.float32)
    else:
        image = image.to(torch.complex64)

    # cast to device is necessary
    if device is not None:
        toeplitz_kern.to(device)

    # unpack plan
    ndim = toeplitz_kern.ndim
    device = toeplitz_kern.device

    # original device
    odevice = image.device

    # offload to computational device
    image = image.to(device)

    # FFT
    kspace = _fft.fft(image, axes=range(-ndim, 0), norm="ortho", centered=False)

    # Toeplitz convolution
    tmp = torch.zeros_like(kspace)
    tmp = _sparse.apply_toeplitz(tmp, kspace, toeplitz_kern, device, threadsperblock)

    # IFFT
    image = _fft.ifft(tmp, axes=range(-ndim, 0), norm="ortho", centered=False)

    # bring back to original device
    image = image.to(odevice)

    # transform back to numpy if required
    if isnumpy:
        image = image.numpy(force=True)

    # collect garbage
    gc.collect()

    return image
