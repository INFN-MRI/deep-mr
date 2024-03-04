"""Toeplitz interpolation subroutines."""

__all__ = ["plan_toeplitz", "apply_toeplitz"]

import gc

from dataclasses import dataclass

import numpy as np
import numba as nb
import torch

from .. import backend

from .plan import plan_sampling
from .sparse2dense import apply_zerofill


def plan_toeplitz(indexes, shape, basis=None, device="cpu"):
    """
    Compute spatio-temporal kernel for fast self-adjoint operation.

    Parameters
    ----------
    indexes : torch.Tensor
        Non-zero data entries of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Indexes must be normalized between ``(0, N_i)``, with ``i=x,y,z``
        being the number of voxels along the corresponding axis.
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
    # convert to tensor if nececessary
    indexes = torch.as_tensor(indexes, dtype=torch.int32)
    if basis is not None:
        basis = torch.as_tensor(basis)

    # expand singleton dimensions
    ishape = list(indexes.shape[:-1])
    ndim = indexes.shape[-1]

    while len(ishape) < 3:
        ishape = [1] + ishape

    ishape = ishape[1:]

    # if spatio-temporal basis is provided, check reality and offload to device
    if basis is not None:
        islowrank = True
        isreal = not torch.is_complex(basis)  # pylint: disable=no-member
        ncoeff, _ = basis.shape
        basis = basis.to(device)
        adjoint_basis = basis.conj().T.to(device)

    else:
        islowrank = False
        isreal = False
        ncoeff = indexes.shape[0]

    if isreal:
        dtype = torch.float32
    else:
        dtype = torch.complex64

    if basis is not None:
        basis = basis.to(dtype)
        adjoint_basis = adjoint_basis.to(dtype)

    if basis is not None:
        # initialize temporary arrays
        delta = np.ones([ncoeff] + list(indexes.shape[:-1]), dtype=np.float32)
        delta = (delta.T * adjoint_basis.numpy(force=True)).T
        delta = torch.as_tensor(delta, device=device)
    else:
        # initialize temporary arrays
        delta = torch.ones(list(indexes.shape[:-1]), dtype=torch.float32, device=device)

    # calculate interpolator
    interpolator = plan_sampling(indexes, shape, device)
    st_kernel = apply_zerofill(delta, interpolator, basis)

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

    return GramMatrix(st_kernel, ndim, device, islowrank)


def apply_toeplitz(
    data_out, data_in, toeplitz_kernel, device=None, threadsperblock=128
):
    """
    Perform in-place fast self-adjoint by multiplication in k-space with spatio-temporal kernel.

    Parameters
    ----------
    data_out : torch.Tensor
        Output tensor of oversampled gridded k-space data of shape (..., (nz), ny, nx).
    data_in : torch.Tensor
        Output tensor of oversampled gridded k-space data of shape (..., (nz), ny, nx).
    toeplitz_kernel : GramMatrix
        Structure containing Toeplitz kernel (i.e., Fourier transform of system tPSF).
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None`` (same as interpolator).
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    """
    # get number of dimensions
    ndim = toeplitz_kernel.ndim

    # convert to tensor if nececessary
    data_in = torch.as_tensor(data_in, device=device)

    # make sure datatype is correct
    if data_in.dtype in (torch.float16, torch.float32, torch.float64):
        data_in = data_in.to(torch.float32)
    else:
        data_in = data_in.to(torch.complex64)

    # cast tp device is necessary
    if device is not None:
        toeplitz_kernel.to(device)

    if toeplitz_kernel.islowrank is True:
        # keep original shape
        shape = data_in.shape

        # reshape
        data_out = data_out.reshape(
            -1, data_out.shape[-ndim - 1], np.prod(data_out.shape[-ndim:])
        )  # (nbatches, ncontrasts, nvoxels)
        data_in = data_in.reshape(
            -1, data_in.shape[-ndim - 1], np.prod(data_in.shape[-ndim:])
        )  # (nbatches, ncontrasts, nvoxels)

        # transpose
        data_out = data_out.permute(
            2, 0, 1
        ).contiguous()  # (nvoxels, nbatches, ncontrasts)
        data_in = data_in.permute(
            2, 0, 1
        ).contiguous()  # (nvoxels, nbatches, ncontrasts)

        # actual interpolation
        if toeplitz_kernel.device == "cpu":
            do_selfadjoint_interpolation(data_out, data_in, toeplitz_kernel.value)
        else:
            do_selfadjoint_interpolation_cuda(
                data_out, data_in, toeplitz_kernel.value, threadsperblock
            )

        # transpose
        data_out = data_out.permute(1, 2, 0).contiguous()

        # reshape back
        data_out = data_out.reshape(*shape)
    else:
        data_out = toeplitz_kernel.value * data_in

    # collect garbage
    gc.collect()

    return data_out


# %% subroutines
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


@dataclass
class GramMatrix:
    value: torch.Tensor
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


def do_selfadjoint_interpolation(data_out, data_in, toeplitz_matrix):
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)
    toeplitz_matrix = backend.pytorch2numba(toeplitz_matrix)

    _interp_selfadjoint(data_out, data_in, toeplitz_matrix)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)
    toeplitz_matrix = backend.numba2pytorch(toeplitz_matrix)


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(out, in_a, in_b):
    row, col = in_b.shape

    for i in range(row):
        for j in range(col):
            out[j] += in_b[i][j] * in_a[j]

    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp_selfadjoint(data_out, data_in, toeplitz_matrix):
    # get data dimension
    nvoxels, batch_size, _ = data_in.shape

    for i in nb.prange(nvoxels * batch_size):
        voxel = i // batch_size
        batch = i % batch_size

        _dot_product(
            data_out[voxel][batch], data_in[voxel][batch], toeplitz_matrix[voxel]
        )


# %% CUDA
if torch.cuda.is_available():
    from numba import cuda

    def do_selfadjoint_interpolation_cuda(
        data_out, data_in, toeplitz_matrix, threadsperblock
    ):
        # calculate size
        nvoxels, batch_size, ncoeff = data_out.shape

        # define number of blocks
        blockspergrid = (
            (nvoxels * batch_size) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)
        toeplitz_matrix = backend.pytorch2numba(toeplitz_matrix)

        # run kernel
        _interp_selfadjoint_cuda[blockspergrid, threadsperblock](
            data_out, data_in, toeplitz_matrix
        )

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)
        toeplitz_matrix = backend.numba2pytorch(toeplitz_matrix)

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_cuda(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_selfadjoint_cuda(data_out, data_in, toeplitz_matrix):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = nb.cuda.grid(1)
        if i < nvoxels * batch_size:
            voxel = i // batch_size
            batch = i % batch_size

            _dot_product_cuda(
                data_out[voxel][batch], data_in[voxel][batch], toeplitz_matrix[voxel]
            )
