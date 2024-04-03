"""Toeplitz interpolation subroutines."""

__all__ = ["apply_toeplitz"]

import gc

import numpy as np
import numba as nb
import torch

from ..._utils import backend


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
        if toeplitz_kernel.device == "cpu" or toeplitz_kernel.device == torch.device("cpu"):
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
