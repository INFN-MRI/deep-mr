"""ND zerofill (sparse -> zero-filled) subroutines."""

__all__ = ["apply_zerofill"]

import gc

import numpy as np
import numba as nb
import torch

from .. import backend


def apply_zerofill(data_in, mask, basis=None, device=None, threadsperblock=128):
    """
    Zerofill of points specified by coordinates to array.

    Parameters
    ----------
    data_in : torch.Tensor
        Input sparse array of shape ``(..., ncontrasts, nviews, nsamples)``.
    mask : dict
        Pre-formatted sampling pattern in sparse COO format.
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
    data_out : torch.Tensor
        Output zero-filled array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    """
    # convert to tensor if nececessary
    data_in = torch.as_tensor(data_in)

    # make sure datatype is correct
    if data_in.dtype in (torch.float16, torch.float32, torch.float64):
        data_in = data_in.to(torch.float32)
    else:
        data_in = data_in.to(torch.complex64)

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
        mask.to(device)

    # unpack input
    index = mask.index
    dshape = mask.dshape
    ishape = mask.ishape
    ndim = mask.ndim
    device = mask.device

    # cast to device
    data_in = data_in.to(device)
    if basis is not None:
        basis = basis.to(device)

    # get input sizes
    nframes = index[0].shape[0]
    npts = np.prod(ishape)

    # reformat data for computation
    if nframes == 1:
        batch_shape = data_in.shape[:-2]
    else:
        batch_shape = data_in.shape[:-3]
    batch_size = int(np.prod(batch_shape))  # ncoils * nslices

    # get number of coefficients
    if basis is not None:
        ncoeff = basis.shape[0]
    else:
        ncoeff = nframes

    # argument reshape
    data_in = data_in.reshape([batch_size, nframes, npts])
    data_in = data_in.swapaxes(0, 1).contiguous()

    # collect garbage
    gc.collect()

    # preallocate output data
    data_out = torch.zeros(
        (ncoeff, batch_size, *dshape), dtype=data_in.dtype, device=device
    )

    # do actual zerofill
    if device == "cpu":
        do_zerofill[ndim - 1](data_out, data_in, index, basis)
    else:
        do_zerofill_cuda[ndim - 1](data_out, data_in, index, basis, threadsperblock)

    # collect garbage
    gc.collect()

    # reformat for output
    if nframes == 1:
        data_out = data_out[0].reshape([*batch_shape, *dshape])
    else:
        data_out = data_out.swapaxes(0, 1)
        data_out = data_out.reshape([*batch_shape, ncoeff, *dshape])

    return data_out


def _do_zerofill1(data_out, data_in, index, basis):
    """1D zerofill routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)

    if basis is None:
        _zerofill1(data_out, data_in, index)
    else:
        basis = backend.pytorch2numba(basis)
        _zerofill_lowrank1(data_out, data_in, index, basis)
        basis = backend.numba2pytorch(basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)


def _do_zerofill2(data_out, data_in, index, basis):
    """2D zerofill routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)

    if basis is None:
        _zerofill2(data_out, data_in, index)
    else:
        basis = backend.pytorch2numba(basis)
        _zerofill_lowrank2(data_out, data_in, index, basis)
        basis = backend.numba2pytorch(basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)


def _do_zerofill3(data_out, data_in, index, basis):
    """3D zerofill routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)

    if basis is None:
        _zerofill3(data_out, data_in, index)
    else:
        basis = backend.pytorch2numba(basis)
        _zerofill_lowrank3(data_out, data_in, index, basis)
        basis = backend.numba2pytorch(basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)


# main handle
do_zerofill = [_do_zerofill1, _do_zerofill2, _do_zerofill3]


# %% subroutines
@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill1(cart_data, noncart_data, index):  # noqa
    # get sizes
    nframes, batch_size, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = index[0]

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            idx = xindex[frame, point]
            cart_data[frame, batch, idx] += noncart_data[frame, batch, point]


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill2(cart_data, noncart_data, index):  # noqa
    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    yindex, xindex = index

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            cart_data[frame, batch, idy, idx] += noncart_data[frame, batch, point]


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill3(cart_data, noncart_data, index):  # noqa
    # get sizes
    nframes, batch_size, _, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = index

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            cart_data[frame, batch, idz, idy, idx] += noncart_data[frame, batch, point]


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill_lowrank1(cart_data, noncart_data, index, basis):  # noqa
    # get sizes
    ncoeff, batch_size, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = index[0]

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                idx = xindex[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while spreading data
                cart_data[coeff, batch, idx] += (
                    basis[coeff, frame] * noncart_data[frame, batch, point]
                )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill_lowrank2(cart_data, noncart_data, index, basis):  # noqa
    # get sizes
    ncoeff, batch_size, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    yindex, xindex = index

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                idy = yindex[frame, point]
                idx = xindex[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while spreading data
                cart_data[coeff, batch, idy, idx] += (
                    basis[coeff, frame] * noncart_data[frame, batch, point]
                )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill_lowrank3(cart_data, noncart_data, index, basis):  # noqa
    # get sizes
    ncoeff, batch_size, _, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = index

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                idz = zindex[frame, point]
                idy = yindex[frame, point]
                idx = xindex[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while gathering data
                cart_data[coeff, batch, idz, idy, idx] += (
                    basis[coeff, frame] * noncart_data[frame, batch, point]
                )


# %% CUDA
if torch.cuda.is_available():
    from numba import cuda

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_real(output, index, value):
        cuda.atomic.add(output, index, value)

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_complex(output, index, value):
        cuda.atomic.add(output.real, index, value.real)
        cuda.atomic.add(output.imag, index, value.imag)

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_cuda1_real(cart_data, noncart_data, index):
        # get sizes
        nframes, batch_size, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = index[0]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            idx = xindex[frame, point]
            _update_real(
                cart_data, (frame, batch, idx), noncart_data[frame, batch, point]
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_cuda2_real(cart_data, noncart_data, index):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idy = yindex[frame, point]
            idx = xindex[frame, point]
            _update_real(
                cart_data, (frame, batch, idy, idx), noncart_data[frame, batch, point]
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_cuda3_real(cart_data, noncart_data, index):
        # get sizes
        nframes, batch_size, _, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            _update_real(
                cart_data,
                (frame, batch, idz, idy, idx),
                noncart_data[frame, batch, point],
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_lowrank_cuda1_real(cart_data, noncart_data, index, basis):
        # get sizes
        ncoeff, batch_size, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = index[0]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idx = xindex[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while spreading data
            for coeff in range(ncoeff):
                _update_real(
                    cart_data,
                    (coeff, batch, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_lowrank_cuda2_real(cart_data, noncart_data, index, basis):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idy = yindex[frame, point]
            idx = xindex[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while spreading data
            for coeff in range(ncoeff):
                _update_real(
                    cart_data,
                    (coeff, batch, idy, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_lowrank_cuda3_real(cart_data, noncart_data, index, basis):
        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while gathering data
            for coeff in range(ncoeff):
                _update_real(
                    cart_data,
                    (coeff, batch, idz, idy, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_cuda1_cplx(cart_data, noncart_data, index):
        # get sizes
        nframes, batch_size, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = index[0]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # spread data within kernel radius
            idx = xindex[frame, point]
            _update_complex(
                cart_data, (frame, batch, idx), noncart_data[frame, batch, point]
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_cuda2_cplx(cart_data, noncart_data, index):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idy = yindex[frame, point]
            idx = xindex[frame, point]
            _update_complex(
                cart_data, (frame, batch, idy, idx), noncart_data[frame, batch, point]
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_cuda3_cplx(cart_data, noncart_data, index):
        # get sizes
        nframes, batch_size, _, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            _update_complex(
                cart_data,
                (frame, batch, idz, idy, idx),
                noncart_data[frame, batch, point],
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_lowrank_cuda1_cplx(cart_data, noncart_data, index, basis):
        # get sizes
        ncoeff, batch_size, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = index[0]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idx = xindex[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while spreading data
            for coeff in range(ncoeff):
                _update_complex(
                    cart_data,
                    (coeff, batch, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_lowrank_cuda2_cplx(cart_data, noncart_data, index, basis):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idy = yindex[frame, point]
            idx = xindex[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while spreading data
            for coeff in range(ncoeff):
                _update_complex(
                    cart_data,
                    (coeff, batch, idy, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _zerofill_lowrank_cuda3_cplx(cart_data, noncart_data, index, basis):
        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while gathering data
            for coeff in range(ncoeff):
                _update_complex(
                    cart_data,
                    (coeff, batch, idz, idy, idx),
                    basis[coeff, frame] * noncart_data[frame, batch, point],
                )

    def _zerofill_cuda1(
        blockspergrid, threadsperblock, data_out, data_in, index, is_complex
    ):
        if is_complex:
            return _zerofill_cuda1_cplx[blockspergrid, threadsperblock](
                data_out, data_in, index
            )
        else:
            return _zerofill_cuda1_real[blockspergrid, threadsperblock](
                data_out, data_in, index
            )

    def _zerofill_cuda2(
        blockspergrid, threadsperblock, data_out, data_in, index, is_complex
    ):
        if is_complex:
            return _zerofill_cuda2_cplx[blockspergrid, threadsperblock](
                data_out, data_in, index
            )
        else:
            return _zerofill_cuda2_real[blockspergrid, threadsperblock](
                data_out, data_in, index
            )

    def _zerofill_cuda3(
        blockspergrid, threadsperblock, data_out, data_in, index, is_complex
    ):
        if is_complex:
            return _zerofill_cuda3_cplx[blockspergrid, threadsperblock](
                data_out, data_in, index
            )
        else:
            return _zerofill_cuda3_real[blockspergrid, threadsperblock](
                data_out, data_in, index
            )

    def _zerofill_lowrank_cuda1(
        blockspergrid,
        threadsperblock,
        data_out,
        data_in,
        index,
        basis,
        is_complex,
    ):
        if is_complex:
            return _zerofill_lowrank_cuda1_cplx[blockspergrid, threadsperblock](
                data_out, data_in, index, basis
            )
        else:
            return _zerofill_lowrank_cuda1_real[blockspergrid, threadsperblock](
                data_out, data_in, index, basis
            )

    def _zerofill_lowrank_cuda2(
        blockspergrid,
        threadsperblock,
        data_out,
        data_in,
        index,
        basis,
        is_complex,
    ):
        if is_complex:
            return _zerofill_lowrank_cuda2_cplx[blockspergrid, threadsperblock](
                data_out, data_in, index, basis
            )
        else:
            return _zerofill_lowrank_cuda2_real[blockspergrid, threadsperblock](
                data_out, data_in, index, basis
            )

    def _zerofill_lowrank_cuda3(
        blockspergrid,
        threadsperblock,
        data_out,
        data_in,
        index,
        basis,
        is_complex,
    ):
        if is_complex:
            return _zerofill_lowrank_cuda3_cplx[blockspergrid, threadsperblock](
                data_out, data_in, index, basis
            )
        else:
            return _zerofill_lowrank_cuda3_real[blockspergrid, threadsperblock](
                data_out, data_in, index, basis
            )

    def _do_zerofill_cuda1(data_out, data_in, index, basis, threadsperblock):
        # get if function is complex
        is_complex = torch.is_complex(data_in)
        if basis is not None:
            is_complex = is_complex or torch.is_complex(basis)

        # calculate size
        _, batch_size, _ = data_out.shape
        nframes = data_in.shape[0]
        npts = data_in.shape[-1]

        # define number of blocks
        blockspergrid = (
            (nframes * batch_size * npts) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)

        # run kernel
        if basis is None:
            _zerofill_cuda1(
                blockspergrid,
                threadsperblock,
                data_out,
                data_in,
                index,
                is_complex,
            )
        else:
            basis = backend.pytorch2numba(basis)
            _zerofill_lowrank_cuda1(
                blockspergrid,
                threadsperblock,
                data_out,
                data_in,
                index,
                basis,
                is_complex,
            )
            basis = backend.numba2pytorch(basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)

    def _do_zerofill_cuda2(data_out, data_in, index, basis, threadsperblock):
        # get if function is complex
        is_complex = torch.is_complex(data_in)
        if basis is not None:
            is_complex = is_complex or torch.is_complex(basis)

        # calculate size
        _, batch_size, _, _ = data_out.shape
        nframes = data_in.shape[0]
        npts = data_in.shape[-1]

        # define number of blocks
        blockspergrid = (
            (nframes * batch_size * npts) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)

        # run kernel
        if basis is None:
            _zerofill_cuda2(
                blockspergrid,
                threadsperblock,
                data_out,
                data_in,
                index,
                is_complex,
            )
        else:
            basis = backend.pytorch2numba(basis)
            _zerofill_lowrank_cuda2(
                blockspergrid,
                threadsperblock,
                data_out,
                data_in,
                index,
                basis,
                is_complex,
            )
            basis = backend.numba2pytorch(basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)

    def _do_zerofill_cuda3(data_out, data_in, index, basis, threadsperblock):
        # get if function is complex
        is_complex = torch.is_complex(data_in)
        if basis is not None:
            is_complex = is_complex or torch.is_complex(basis)

        # calculate size
        _, batch_size, _, _, _ = data_out.shape
        nframes = data_in.shape[0]
        npts = data_in.shape[-1]

        # define number of blocks
        blockspergrid = (
            (nframes * batch_size * npts) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)

        # run kernel
        if basis is None:
            _zerofill_cuda3(
                blockspergrid,
                threadsperblock,
                data_out,
                data_in,
                index,
                is_complex,
            )
        else:
            basis = backend.pytorch2numba(basis)
            _zerofill_lowrank_cuda3(
                blockspergrid,
                threadsperblock,
                data_out,
                data_in,
                index,
                basis,
                is_complex,
            )
            basis = backend.numba2pytorch(basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)

    # main handle
    do_zerofill_cuda = [_do_zerofill_cuda1, _do_zerofill_cuda2, _do_zerofill_cuda3]
