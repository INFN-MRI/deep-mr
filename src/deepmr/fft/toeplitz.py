"""Toeplitz interpolation planning."""

__all__ = ["plan_toeplitz"]

from dataclasses import dataclass

import numpy as np
import torch

from .. import fft as _fft
from .. import _signal

from .._utils import backend


def plan_toeplitz(
    coord,
    shape,
    basis=None,
    dcf=None,
    width=3,
    device="cpu",
):
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
    # convert to tensor if nececessary
    coord = torch.as_tensor(coord, dtype=torch.float32)
    if basis is not None:
        basis = torch.as_tensor(basis)

    # expand singleton dimensions
    ndim = coord.shape[-1]

    # kernel oversampling
    oversamp = np.asarray([2.0] * ndim)

    # shape
    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.asarray(shape, dtype=np.int16)[-ndim:]

    # if dcf are not provided, assume uniform sampling density
    if dcf is None:
        dcf = torch.ones(coord.shape[:-1], dtype=torch.float32, device=device)
    else:
        dcf = dcf.to(device)

    # if spatio-temporal basis is provided, check reality and offload to device
    if basis is not None:
        islowrank = True
        isreal = not torch.is_complex(basis)
        ncoeff, _ = basis.shape
        basis = basis.to(device)
        adjoint_basis = basis.conj().T.to(device)
    else:
        islowrank = False
        isreal = False
        ncoeff = coord.shape[0]

    if isreal:
        dtype = torch.float32
    else:
        dtype = torch.complex64

    if basis is not None:
        basis = basis.to(dtype)
        adjoint_basis = adjoint_basis.to(dtype)

    if basis is not None:
        # initialize temporary arrays
        delta = np.ones([ncoeff] + list(coord.shape[:-1]), dtype=np.float32)
        delta = (delta.T * adjoint_basis.numpy(force=True)).T
        delta = torch.as_tensor(delta, device=device)
    else:
        # initialize temporary arrays
        delta = torch.ones(list(coord.shape[:-1]), dtype=torch.float32, device=device)

    # calculate PSF
    st_kernel = _fft.nufft_adj(
        dcf * delta, coord, shape, basis, device, width=width, oversamp=oversamp
    )

    # check for Cartesian axes
    is_cart = [
        np.allclose(shape[ax] * coord[..., ax], np.round(shape[ax] * coord[..., ax]))
        for ax in range(ndim)
    ]
    is_cart = np.asarray(is_cart[::-1])  # (z, y, x)

    # Cartesian axes have osf = 1.0 and kernel width = 1 (no interpolation)
    oversamp[is_cart] = 1.0

    # get oversampled grid shape
    shape = _get_oversamp_shape(shape, oversamp, ndim)

    # pad and FFT
    st_kernel = _signal.resize(st_kernel, list(st_kernel.shape[:-ndim]) + list(shape))
    st_kernel = _fft.fft(st_kernel, axes=range(-ndim, 0))

    # squeeze
    if st_kernel.shape[0] == 1:
        st_kernel = st_kernel[0]

    # keep only real part if basis is real
    if isreal:
        st_kernel = st_kernel.real

    # fftshift kernel to accelerate computation
    st_kernel = torch.fft.ifftshift(st_kernel, dim=list(range(-ndim, 0)))

    if basis is not None:
        st_kernel = st_kernel.reshape(
            *st_kernel.shape[:2], np.prod(st_kernel.shape[2:])
        )
        st_kernel = st_kernel.permute(2, 1, 0).contiguous()

    # normalize
    st_kernel /= torch.mean(abs(st_kernel[st_kernel != 0]))

    # remove NaN
    st_kernel = torch.nan_to_num(st_kernel)

    return GramMatrix(st_kernel, tuple(shape), ndim, device, islowrank)


# %% local utils
def _get_oversamp_shape(shape, oversamp, ndim):
    return np.ceil(oversamp * shape).astype(np.int16)


@dataclass
class GramMatrix:
    value: torch.Tensor
    shape: tuple
    ndim: int
    device: str
    islowrank: bool

    def to(self, device):
        """
        Dispatch internal attributes to selected device.

        Parameters
        ----------
        device : str
            Computational device ("cpu" or "cuda:n", with n=0, 1,...nGPUs).

        """
        if device != self.device:
            # zero-copy to torch
            self.value = backend.numba2pytorch(self.value)

            # dispatch
            self.value = self.value.to(device)

            # zero-copy to numba
            self.value = backend.pytorch2numba(self.value)

            self.device = device

        return self
