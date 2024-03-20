"""NUFFT subroutines."""

__all__ = [
    "plan_nufft",
    "plan_toeplitz_nufft",
    "apply_nufft",
    "apply_nufft_adj",
    "apply_nufft_selfadj",
]

import gc
import math

from dataclasses import dataclass

import numpy as np
import torch
import torch.autograd as autograd

from .._signal import resize as _resize

from . import fft as _fft
from . import _interp
from . import toeplitz as _toeplitz


def plan_nufft(coord, shape, width=4, oversamp=1.25, device="cpu"):
    """
    Precompute NUFFT object.

    Parameters
    ----------
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5 * shape)``.
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
    interpolator : NUFFTPlan
        Structure containing sparse interpolator matrix:

        * ndim (``int``): number of spatial dimensions.
        * oversampling (``Iterable[float]``): grid oversampling factor (z, y, x).
        * width (``Iterable[int]``): kernel width (z, y, x).
        * beta (``Iterable[float]``): Kaiser Bessel parameter (z, y, x).
        * os_shape (``Iterable[int]``): oversampled grid shape (z, y, x).
        * shape (``Iterable[int]``): grid shape (z, y, x).
        * interpolator (``Interpolator``): precomputed interpolator object.
        * device (``str``): computational device.

    Notes
    -----
    Non-uniform coordinates axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, kernel width
    and oversampling factors are assumed to be ``(y, x)`` and ``(z, y, x)``.

    Coordinates tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

    * ``coord.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
    * ``coord.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # make sure this is a tensor
    coord = torch.as_tensor(coord)

    # copy coord and switch to cpu
    coord = coord.clone().cpu().to(torch.float32)

    # get parameters
    ndim = coord.shape[-1]

    if np.isscalar(width):
        width = np.asarray([width] * ndim, dtype=np.int16)
    else:
        width = np.asarray(width, dtype=np.int16)

    if np.isscalar(oversamp):
        oversamp = np.asarray([oversamp] * ndim, dtype=np.float32)
    else:
        oversamp = np.asarray(oversamp, dtype=np.float32)

    # calculate Kaiser-Bessel beta parameter
    beta = math.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.asarray(shape, dtype=np.int16)[-ndim:]

    # check for Cartesian axes
    is_cart = [
        np.allclose(shape[ax] * coord[..., ax], np.round(shape[ax] * coord[..., ax]))
        for ax in range(ndim)
    ]
    is_cart = np.asarray(is_cart[::-1])  # (z, y, x)

    # Cartesian axes have osf = 1.0 and kernel width = 1 (no interpolation)
    oversamp[is_cart] = 1.0
    width[is_cart] = 1

    # get oversampled grid shape
    os_shape = _get_oversamp_shape(shape, oversamp, ndim)

    # rescale trajectory
    coord = _scale_coord(coord, shape[::-1], oversamp[::-1])

    # compute interpolator
    interpolator = _interp.plan_interpolator(coord, os_shape, width, beta, device)

    # transform to tuples
    ndim: int
    oversamp = tuple(oversamp)
    width = tuple(width)
    beta = tuple(beta)
    os_shape = tuple(os_shape)
    shape = tuple(shape)

    return NUFFTPlan(ndim, oversamp, width, beta, os_shape, shape, interpolator, device)


def plan_toeplitz_nufft(coord, shape, basis=None, dcf=None, width=4, device="cpu"):
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
    dcf : torch.Tensor, optional
        Density compensation function of shape ``(ncontrasts, nviews, nsamples)``.
        The default is a tensor of ``1.0``.
    width : int | Iterable[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``3``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.

    Returns
    -------
    toeplitz_kernel : GramMatrix
        Structure containing Toeplitz kernel (i.e., Fourier transform of system tPSF).

    """
    return _toeplitz.plan_toeplitz(coord, shape, basis, dcf, width, device)


class ApplyNUFFT(autograd.Function):
    @staticmethod
    def forward(image, nufft_plan, basis_adjoint, weight, device, threadsperblock):
        return _apply_nufft(
            image, nufft_plan, basis_adjoint, weight, device, threadsperblock
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, nufft_plan, basis_adjoint, weight, device, threadsperblock = inputs
        ctx.set_materialize_grads(False)
        ctx.nufft_plan = nufft_plan
        ctx.basis_adjoint = basis_adjoint
        ctx.weight = weight
        ctx.device = device
        ctx.threadsperblock = threadsperblock

    @staticmethod
    def backward(ctx, kspace):
        nufft_plan = ctx.nufft_plan
        basis_adjoint = ctx.basis_adjoint
        if basis_adjoint is not None:
            basis = basis_adjoint.conj().t()
        else:
            basis = None
        weight = ctx.weight
        device = ctx.device
        threadsperblock = ctx.threadsperblock

        return (
            _apply_nufft_adj(
                kspace, nufft_plan, basis, weight, device, threadsperblock
            ),
            None,
            None,
            None,
            None,
            None,
        )


def apply_nufft(
    image, nufft_plan, basis_adjoint=None, weight=None, device=None, threadsperblock=128
):
    """
    Apply Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    image : np.ndarray | torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    nufft_plan : NUFFTPlan
        Pre-calculated NUFFT plan coefficients in sparse COO format.
    basis_adjoint : torch.Tensor, optional
        Adjoint low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    weight : np.ndarray | torch.Tensor, optional
        Optional weight for output data samples. Useful to force adjointeness.
        The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None`` (same as interpolator).
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    kspace : np.ndarray | torch.Tensor
        Output Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    """
    return ApplyNUFFT.apply(
        image, nufft_plan, basis_adjoint, weight, device, threadsperblock
    )


class ApplyNUFFTAdjoint(autograd.Function):
    @staticmethod
    def forward(kspace, nufft_plan, basis, weight, device, threadsperblock):
        return _apply_nufft_adj(
            kspace, nufft_plan, basis, weight, device, threadsperblock
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, nufft_plan, basis, weight, device, threadsperblock = inputs
        ctx.set_materialize_grads(False)
        ctx.nufft_plan = nufft_plan
        ctx.basis = basis
        ctx.weight = weight
        ctx.device = device
        ctx.threadsperblock = threadsperblock

    @staticmethod
    def backward(ctx, image):
        nufft_plan = ctx.nufft_plan
        basis = ctx.basis
        if basis is not None:
            basis_adjoint = basis.conj().t()
        else:
            basis_adjoint = None
        weight = ctx.weight
        device = ctx.device
        threadsperblock = ctx.threadsperblock

        return (
            _apply_nufft(
                image, nufft_plan, basis_adjoint, weight, device, threadsperblock
            ),
            None,
            None,
            None,
            None,
            None,
        )


def apply_nufft_adj(
    kspace, nufft_plan, basis=None, weight=None, device=None, threadsperblock=128
):
    """
    Apply adjoint Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    kspace : torch.Tensor
        Input kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    nufft_plan : NUFFTPlan
        Pre-calculated NUFFT plan coefficients in sparse COO format.
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    weight : np.ndarray | torch.Tensor, optional
        Optional weight for output data samples. Useful to force adjointeness.
        The default is ``None``.
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
    return ApplyNUFFTAdjoint.apply(
        kspace, nufft_plan, basis, weight, device, threadsperblock
    )


class ApplyNUFFTSelfAdjoint(autograd.Function):
    @staticmethod
    def forward(image, toeplitz_kern, device, threadsperblock):
        return _apply_nufft_selfadj(image, toeplitz_kern, device, threadsperblock)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, toeplitz_kern, device, threadsperblock = inputs
        ctx.set_materialize_grads(False)
        ctx.toeplitz_kern = toeplitz_kern
        ctx.device = device
        ctx.threadsperblock = threadsperblock

    @staticmethod
    def backward(ctx, image):
        toeplitz_kern = ctx.toeplitz_kern
        device = ctx.device
        threadsperblock = ctx.threadsperblock
        return (
            _apply_nufft_selfadj(image, toeplitz_kern, device, threadsperblock),
            None,
            None,
            None,
        )


def apply_nufft_selfadj(image, toeplitz_kern, device=None, threadsperblock=128):
    """
    Apply self-adjoint Non-Uniform Fast Fourier Transform via Toeplitz Convolution.

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
    return ApplyNUFFTSelfAdjoint.apply(image, toeplitz_kern, device, threadsperblock)


# %% local utils
@dataclass
class NUFFTPlan:
    ndim: int
    oversamp: tuple
    width: tuple
    beta: tuple
    os_shape: tuple
    shape: tuple
    interpolator: object
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
            self.interpolator.to(device)
            self.device = device

        return self


def _get_oversamp_shape(shape, oversamp, ndim):
    return np.ceil(oversamp * shape).astype(np.int16)


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.clone()
    for i in range(-ndim, 0):
        scale = np.ceil(oversamp[i] * shape[i]) / shape[i]
        shift = np.ceil(oversamp[i] * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def _apodize(data_in, ndim, oversamp, width, beta):
    data_out = data_in
    for n in range(1, ndim + 1):
        axis = -n
        if width[axis] != 1:
            i = data_out.shape[axis]
            os_i = np.ceil(oversamp[axis] * i)
            idx = torch.arange(i, dtype=torch.float32, device=data_in.device)

            # Calculate apodization
            apod = (
                beta[axis] ** 2 - (math.pi * width[axis] * (idx - i // 2) / os_i) ** 2
            ) ** 0.5
            apod /= torch.sinh(apod)

            # normalize by DC
            apod = apod / apod[int(i // 2)]

            # avoid NaN
            apod = torch.nan_to_num(apod, nan=1.0)

            # apply to axis
            data_out *= apod.reshape([i] + [1] * (-axis - 1))

    return data_out


def _apply_nufft(image, nufft_plan, basis_adjoint, weight, device, threadsperblock):
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
        nufft_plan.to(device)

    # unpack plan
    ndim = nufft_plan.ndim
    oversamp = nufft_plan.oversamp
    width = nufft_plan.width
    beta = nufft_plan.beta
    os_shape = nufft_plan.os_shape
    interpolator = nufft_plan.interpolator
    device = nufft_plan.device

    # copy input to avoid original data modification
    image = image.clone()

    # original device
    odevice = image.device

    # offload to computational device
    image = image.to(device)

    # apodize
    _apodize(image, ndim, oversamp, width, beta)

    # zero-pad
    image = _resize(image, list(image.shape[:-ndim]) + list(os_shape))

    # FFT
    kspace = _fft.fft(image, axes=range(-ndim, 0), norm=None)

    # interpolate
    kspace = _interp.apply_interpolation(
        kspace, interpolator, basis_adjoint, device, threadsperblock
    )

    # apply weight
    if weight is not None:
        weight = torch.as_tensor(weight, dtype=torch.float32, device=kspace.device)
        kspace = weight * kspace

    # bring back to original device
    kspace = kspace.to(odevice)
    image = image.to(odevice)

    # transform back to numpy if required
    if isnumpy:
        kspace = kspace.numpy(force=True)

    # collect garbage
    gc.collect()

    return kspace


def _apply_nufft_adj(kspace, nufft_plan, basis, weight, device, threadsperblock):
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
        nufft_plan.to(device)

    # unpack plan
    ndim = nufft_plan.ndim
    oversamp = nufft_plan.oversamp
    width = nufft_plan.width
    beta = nufft_plan.beta
    shape = nufft_plan.shape
    interpolator = nufft_plan.interpolator
    device = nufft_plan.device

    # original device
    odevice = kspace.device

    # offload to computational device
    kspace = kspace.to(device)

    # apply weight
    if weight is not None:
        weight = torch.as_tensor(weight, dtype=torch.float32, device=kspace.device)
        kspace = weight * kspace

    # gridding
    kspace = _interp.apply_gridding(
        kspace, interpolator, basis, device, threadsperblock
    )

    # IFFT
    image = _fft.ifft(kspace, axes=range(-ndim, 0), norm=None)

    # crop
    image = _resize(image, list(image.shape[:-ndim]) + list(shape))

    # apodize
    _apodize(image, ndim, oversamp, width, beta)

    # bring back to original device
    kspace = kspace.to(odevice)
    image = image.to(odevice)

    # transform back to numpy if required
    if isnumpy:
        image = image.numpy(force=True)

    # collect garbage
    gc.collect()

    return image


def _apply_nufft_selfadj(image, toeplitz_kern, device, threadsperblock):
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
    shape = toeplitz_kern.shape
    ndim = toeplitz_kern.ndim
    device = toeplitz_kern.device

    # original shape
    oshape = image.shape[-ndim:]

    # original device
    odevice = image.device

    # offload to computational device
    image = image.to(device)

    # zero-pad
    image = _resize(image, list(image.shape[:-ndim]) + list(shape))

    # FFT
    kspace = _fft.fft(image, axes=range(-ndim, 0), norm="ortho", centered=False)

    # Toeplitz convolution
    tmp = torch.zeros_like(kspace)
    tmp = _interp.apply_toeplitz(tmp, kspace, toeplitz_kern, device, threadsperblock)

    # IFFT
    image = _fft.ifft(tmp, axes=range(-ndim, 0), norm="ortho", centered=False)

    # crop
    image = _resize(image, list(image.shape[:-ndim]) + list(oshape))

    # bring back to original device
    image = image.to(odevice)

    # transform back to numpy if required
    if isnumpy:
        image = image.numpy(force=True)

    # collect garbage
    gc.collect()

    return image
