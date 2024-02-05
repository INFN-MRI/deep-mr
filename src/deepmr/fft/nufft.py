"""FFT subroutines."""

__all__ = ["plan_nufft", "apply_nufft", "apply_nufft_adj"]

from dataclasses import dataclass

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
    and oversampling factors are assumed to be ``(z, y, x)``.

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
        
    if np.isscalar(oversamp):
        oversamp = np.asarray([oversamp] * ndim, dtype=np.int16)
    else:
        oversamp = np.asarray(oversamp, dtype=np.int16)

    # calculate Kaiser-Bessel beta parameter
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.array(shape, dtype=np.int16)

    os_shape = _get_oversamp_shape(shape, oversamp, ndim)

    # compute interpolator
    interpolator = _interp.prepare_interpolator(coord, os_shape, width, beta, device)

    # transform to tuples
    ndim: int
    oversamp = tuple(oversamp)
    width = tuple(width)
    beta = tuple(beta)
    os_shape = tuple(os_shape)
    shape = tuple(shape)
    
    return NUFFTPlan(ndim, oversamp, width, beta, os_shape, shape, interpolator, device)

def apply_nufft(image, nufft_plan, basis_adjoint=None, device=None, threadsperblock=128):
    """
    Apply Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    nufft_plan : dict
        Pre-calculated NUFFT plan coefficients in sparse COO format.
    basis_adjoint : torch.Tensor, optional
        Adjoint low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None`` (same as interpolator).
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    kspace : torch.Tensor
        Output Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    """
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
    kspace = _fft.fft(image, axes=range(-ndim, 0), norm=None)

    # Interpolate
    kspace = _interp.apply_interpolation(kspace, interpolator, basis_adjoint, device, threadsperblock)
    kspace /= np.prod(width)

    # Bring back to original device
    kspace = kspace.to(odevice)
    image = image.to(odevice)

    return kspace

def apply_nufft_adj(kspace, nufft_plan, basis=None, device=None, threadsperblock=128):
    """
    Apply adjoint Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    kspace : torch.Tensor
        Input kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    nufft_plan : dict
        Pre-calculated NUFFT plan coefficients in sparse COO format.
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncoeffs, ncontrasts)``; can be ``None``. The default is ``None``.
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
    os_shape = nufft_plan.os_shape
    shape = nufft_plan.shape
    interpolator = nufft_plan.interpolator
    device = nufft_plan.device

    # Original device
    odevice = kspace.device

    # Offload to computational device
    kspace = kspace.to(device)

    # Gridding
    kspace = _interp.apply_gridding(kspace, interpolator, basis, device, threadsperblock)
    kspace /= np.prod(width)

    # IFFT
    image = _fft.ifft(kspace, axes=range(-ndim, 0), norm=None)

    # Crop
    image = _resize.resize(image, list(image.shape[:-ndim]) + shape.tolist())
    image *= np.prod(os_shape[-ndim:]) / np.prod(shape[-ndim:])**0.5

    # Apodize
    _apodize(image, ndim, oversamp, width, beta)

    # Bring back to original device
    kspace = kspace.to(odevice)
    image = image.to(odevice)

    return image

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
            
def _get_oversamp_shape(shape, oversamp, ndim):
    return list(shape)[:-ndim] + [np.ceil(oversamp * i).astype(np.int16) for i in shape[-ndim:]]

def _apodize(data_in, ndim, oversamp, width, beta):
    data_out = data_in
    for axis in range(-ndim, 0):
        i = data_out.shape[axis]
        os_i = np.ceil(oversamp[axis] * i)
        idx = torch.arange(i, dtype=torch.float32)

        # Calculate apodization
        apod = (beta[axis]**2 - (np.pi * width[axis] * (idx - i // 2) / os_i)**2)**0.5
        apod /= torch.sinh(apod)
        data_out *= apod.reshape([i] + [1] * (-axis - 1))

    return data_out