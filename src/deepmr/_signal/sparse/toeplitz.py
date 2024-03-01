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
    # get dimensions
    ndim = indexes.shape[-1]
    npts = indexes.shape[-2]

    # if spatio-temporal basis is provided, check reality and offload to device
    if basis is not None:
        islowrank = True
        isreal = not torch.is_complex(basis)  # pylint: disable=no-member
        ncoeff, nframes = basis.shape
        basis = basis.to(device)
        adjoint_basis = basis.conj().T.to(device)

    else:
        islowrank = False
        isreal = False
        nframes, ncoeff = indexes.shape[0], indexes.shape[0]

    if isreal:
        dtype = torch.float32
    else:
        dtype = torch.complex64

    if basis is not None:
        basis = basis.to(dtype)
        adjoint_basis = adjoint_basis.to(dtype)

    if basis is not None:
        # initialize temporary arrays
        delta = torch.ones((nframes, ncoeff, npts), dtype=torch.complex64, device=device)
        delta = delta * adjoint_basis[:, :, None]

    else:
        # initialize temporary arrays
        delta = torch.ones((nframes, npts), dtype=torch.complex64, device=device)

    # calculate interpolator
    interpolator = plan_sampling(indexes, shape, device)
    st_kernel = apply_zerofill(delta, interpolator, basis)

    # keep only real part if basis is real
    if isreal:
        st_kernel = st_kernel.real

    # fftshift kernel to accelerate computation
    st_kernel = torch.fft.ifftshift(st_kernel, dim=list(range(-ndim, 0)))

    if basis is not None:
        st_kernel = st_kernel.reshape(*st_kernel.shape[:2], np.prod(st_kernel.shape[2:]))

    # normalize
    st_kernel /= torch.quantile(st_kernel, 0.95)

    # remove NaN
    st_kernel = torch.nan_to_num(st_kernel)

    # get device identifier
    if device == "cpu":
        device_str = device
    else:
        device_str = device[:4]

    return GramMatrix(st_kernel, device_str, islowrank)


def apply_toeplitz(data_out, data_in, toeplitz_kernel, device=None, threadsperblock=128):
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
        if toeplitz_kernel.device == "cpu":
            do_selfadjoint_interpolation(data_out, data_in, toeplitz_kernel.value)
        else:
            do_selfadjoint_interpolation_cuda(data_out, data_in, toeplitz_kernel.value)
    else:
        data_out = toeplitz_kernel.value * data_in
        
    # collect garbage
    gc.collect()
        

# %% subroutines
@dataclass
class GramMatrix:
    value: torch.Tensor
    device: str
    islowrank : bool

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
    row, col = in_a.shape

    for j in range(col):
        for i in range(row):
            out[j] += in_a[i][j] * in_b[j]

    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp_selfadjoint(data_out, toeplitz_matrix, data_in):

    n_coeff, batch_size, _ = data_in.shape

    for i in nb.prange(n_coeff*batch_size):
        coeff = i // batch_size
        batch = i % batch_size

        _dot_product(data_out[coeff][batch], toeplitz_matrix[coeff], data_in[coeff][batch])

    return data_out

# %% CUDA
if torch.cuda.is_available():
    from numba import cuda
    
    def do_selfadjoint_interpolation_cuda(data_out, data_in, toeplitz_matrix, threadsperblock):
        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)
        toeplitz_matrix = backend.pytorch2numba(toeplitz_matrix)

        # run kernel
        _interp_selfadjoint_cuda[blockspergrid, threadsperblock](data_out, toeplitz_matrix, data_in)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)
        toeplitz_matrix = backend.numba2pytorch(toeplitz_matrix)
        
        
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_cuda(out, in_a, in_b):
        row, col = in_a.shape

        for j in range(col):
            for i in range(row):
                out[j] += in_a[i][j] * in_b[j]

        return out
        
    
    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_selfadjoint_cuda(data_out, toeplitz_matrix, data_in):

        n_coeff, batch_size, _ = data_in.shape

        i = nb.cuda.grid(1)
        if i < n_coeff*batch_size:
            coeff = i // batch_size
            batch = i % batch_size

            _dot_product_cuda(data_out[coeff][batch], toeplitz_matrix[coeff], data_in[coeff][batch])

        return data_out
        
        