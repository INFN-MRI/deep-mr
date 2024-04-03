"""ND interpolation (uniform -> non-uniform) subroutines."""

__all__ = ["apply_interpolation"]

import gc

import numpy as np
import numba as nb
import torch

from ..._utils import backend


def apply_interpolation(
    data_in, interpolator, basis_adjoint=None, device=None, threadsperblock=128
):
    """
    Interpolation from array to points specified by coordinates.

    Parameters
    ----------
    data_in : torch.Tensor
        Input Cartesian array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    interpolator : dict
        Pre-calculated interpolation coefficients in sparse COO format.
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
    data_out : torch.Tensor
        Output Non-Cartesian array of shape ``(..., ncontrasts, nviews, nsamples)``.

    """
    # convert to tensor if nececessary
    data_in = torch.as_tensor(data_in)

    # make sure datatype is correct
    if data_in.dtype in (torch.float16, torch.float32, torch.float64):
        data_in = data_in.to(torch.float32)
    else:
        data_in = data_in.to(torch.complex64)

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
        interpolator.to(device)

    # unpack input
    index = interpolator.index
    value = interpolator.value
    dshape = interpolator.dshape
    ishape = interpolator.ishape
    ndim = interpolator.ndim
    scale = interpolator.scale
    device = interpolator.device

    # cast to device
    data_in = data_in.to(device)
    if basis_adjoint is not None:
        basis_adjoint = basis_adjoint.to(device)

    # get input sizes
    nframes = index[0].shape[0]
    npts = np.prod(ishape)

    # reformat data for computation
    if nframes == 1:
        batch_shape = data_in.shape[:-ndim]
    else:
        batch_shape = data_in.shape[: -ndim - 1]
    batch_size = int(np.prod(batch_shape))  # ncoils * nslices * [int]

    # reshape
    data_in = data_in.reshape(batch_size, nframes, *dshape)
    data_in = data_in.swapaxes(0, 1).contiguous()

    # collect garbage
    gc.collect()

    # preallocate output data
    data_out = torch.zeros(
        (nframes, batch_size, npts), dtype=data_in.dtype, device=device
    )

    # do actual interpolation
    if device == "cpu" or device == torch.device("cpu"):
        do_interpolation[ndim - 1](data_out, data_in, value, index, basis_adjoint)
    else:
        do_interpolation_cuda[ndim - 1](
            data_out, data_in, value, index, basis_adjoint, threadsperblock
        )

    # collect garbage
    gc.collect()

    # reformat for output
    if nframes == 1:
        data_out = data_out[0].reshape(*batch_shape, *ishape)
    else:
        data_out = data_out.swapaxes(0, 1)
        data_out = data_out.reshape(*batch_shape, nframes, *ishape)

    return data_out / scale


# %% subroutines
@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate1(noncart_data, cart_data, interp_value, interp_index):  # noqa
    # get sizes
    nframes, batch_size, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = interp_index[0]
    xvalue = interp_value[0]

    # get interpolator width
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_x in range(xwidth):
            idx = xindex[frame, point, i_x]
            val = xvalue[frame, point, i_x]

            noncart_data[frame, batch, point] += val * cart_data[frame, batch, idx]


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate2(noncart_data, cart_data, interp_value, interp_index):  # noqa
    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    yindex, xindex = interp_index
    yvalue, xvalue = interp_value

    # get interpolator width
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_y in range(ywidth):
            idy = yindex[frame, point, i_y]
            valy = yvalue[frame, point, i_y]

            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = valy * xvalue[frame, point, i_x]

                noncart_data[frame, batch, point] += (
                    val * cart_data[frame, batch, idy, idx]
                )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate3(noncart_data, cart_data, interp_value, interp_index):  # noqa
    # get sizes
    nframes, batch_size, _, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = interp_index
    zvalue, yvalue, xvalue = interp_value

    # get interpolator width
    zwidth = zindex.shape[-1]
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_z in range(zwidth):
            idz = zindex[frame, point, i_z]
            valz = zvalue[frame, point, i_z]

            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = valz * yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    noncart_data[frame, batch, point] += (
                        val * cart_data[frame, batch, idz, idy, idx]
                    )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate_lowrank1(
    noncart_data, cart_data, interp_value, interp_index, basis_adjoint
):  # noqa
    # get sizes
    ncoeff, batch_size, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    xindex = interp_index[0]
    xvalue = interp_value[0]

    # get interpolator width
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_x in range(xwidth):
            idx = xindex[frame, point, i_x]
            val = xvalue[frame, point, i_x]

            # do adjoint low rank projection (low-rank subspace -> time domain)
            # while gathering data
            for coeff in range(ncoeff):
                noncart_data[frame, batch, point] += (
                    val * basis_adjoint[frame, coeff] * cart_data[coeff, batch, idx]
                )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate_lowrank2(
    noncart_data, cart_data, interp_value, interp_index, basis_adjoint
):  # noqa
    # get sizes
    ncoeff, batch_size, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    yindex, xindex = interp_index
    yvalue, xvalue = interp_value

    # get interpolator width
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_y in range(ywidth):
            idy = yindex[frame, point, i_y]
            valy = yvalue[frame, point, i_y]

            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = valy * xvalue[frame, point, i_x]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while gathering data
                for coeff in range(ncoeff):
                    noncart_data[frame, batch, point] += (
                        val
                        * basis_adjoint[frame, coeff]
                        * cart_data[coeff, batch, idy, idx]
                    )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate_lowrank3(
    noncart_data, cart_data, interp_value, interp_index, basis_adjoint
):  # noqa
    # get sizes
    ncoeff, batch_size, _, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = interp_index
    zvalue, yvalue, xvalue = interp_value

    # get interpolator width
    zwidth = zindex.shape[-1]
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_z in range(zwidth):
            idz = zindex[frame, point, i_z]
            valz = zvalue[frame, point, i_z]

            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = valz * yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    # do adjoint low rank projection (low-rank subspace -> time domain)
                    # while gathering data
                    for coeff in range(ncoeff):
                        noncart_data[frame, batch, point] += (
                            val
                            * basis_adjoint[frame, coeff]
                            * cart_data[coeff, batch, idz, idy, idx]
                        )


def _do_interpolation1(data_out, data_in, value, index, basis_adjoint):
    """2D Interpolation routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)

    if basis_adjoint is None:
        _interpolate1(data_out, data_in, value, index)
    else:
        basis_adjoint = backend.pytorch2numba(basis_adjoint)
        _interpolate_lowrank1(data_out, data_in, value, index, basis_adjoint)
        basis_adjoint = backend.numba2pytorch(basis_adjoint)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)


def _do_interpolation2(data_out, data_in, value, index, basis_adjoint):
    """2D Interpolation routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)

    if basis_adjoint is None:
        _interpolate2(data_out, data_in, value, index)
    else:
        basis_adjoint = backend.pytorch2numba(basis_adjoint)
        _interpolate_lowrank2(data_out, data_in, value, index, basis_adjoint)
        basis_adjoint = backend.numba2pytorch(basis_adjoint)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)


def _do_interpolation3(data_out, data_in, value, index, basis_adjoint):
    """3D Interpolation routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)

    if basis_adjoint is None:
        _interpolate3(data_out, data_in, value, index)
    else:
        basis_adjoint = backend.pytorch2numba(basis_adjoint)
        _interpolate_lowrank3(data_out, data_in, value, index, basis_adjoint)
        basis_adjoint = backend.numba2pytorch(basis_adjoint)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)


# main handle
do_interpolation = [_do_interpolation1, _do_interpolation2, _do_interpolation3]

# %% CUDA
if torch.cuda.is_available():
    from numba import cuda

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_cuda1(noncart_data, cart_data, interp_value, interp_index):
        # get sizes
        nframes, batch_size, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = interp_index[0]
        xvalue = interp_value[0]

        # get interpolator width
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                noncart_data[frame, batch, point] += val * cart_data[frame, batch, idx]

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_cuda2(noncart_data, cart_data, interp_value, interp_index):
        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = interp_index
        yvalue, xvalue = interp_value

        # get interpolator width
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    noncart_data[frame, batch, point] += (
                        val * cart_data[frame, batch, idy, idx]
                    )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_cuda3(noncart_data, cart_data, interp_value, interp_index):
        # get sizes
        nframes, batch_size, _, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        noncart_data[frame, batch, point] += (
                            val * cart_data[frame, batch, idz, idy, idx]
                        )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_lowrank_cuda1(
        noncart_data, cart_data, interp_value, interp_index, basis_adjoint
    ):
        # get sizes
        ncoeff, batch_size, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        xindex = interp_index[0]
        xvalue = interp_value[0]

        # get interpolator width
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = xvalue[frame, point, i_x]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while gathering data
                for coeff in range(ncoeff):
                    noncart_data[frame, batch, point] += (
                        val * basis_adjoint[frame, coeff] * cart_data[coeff, batch, idx]
                    )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_lowrank_cuda2(
        noncart_data, cart_data, interp_value, interp_index, basis_adjoint
    ):
        # get sizes
        ncoeff, batch_size, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        yindex, xindex = interp_index
        yvalue, xvalue = interp_value

        # get interpolator width
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    # do adjoint low rank projection (low-rank subspace -> time domain)
                    # while gathering data
                    for coeff in range(ncoeff):
                        noncart_data[frame, batch, point] += (
                            val
                            * basis_adjoint[frame, coeff]
                            * cart_data[coeff, batch, idy, idx]
                        )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_lowrank_cuda3(
        noncart_data, cart_data, interp_value, interp_index, basis_adjoint
    ):
        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while gathering data
                        for coeff in range(ncoeff):
                            noncart_data[frame, batch, point] += (
                                val
                                * basis_adjoint[frame, coeff]
                                * cart_data[coeff, batch, idz, idy, idx]
                            )

    def _do_interpolation_cuda1(
        data_out, data_in, value, index, basis_adjoint, threadsperblock
    ):
        # calculate size
        _, batch_size, _ = data_in.shape
        nframes = data_out.shape[0]
        npts = data_out.shape[-1]

        # define number of blocks
        blockspergrid = (
            (nframes * batch_size * npts) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)

        # run kernel
        if basis_adjoint is None:
            _interpolate_cuda1[blockspergrid, threadsperblock](
                data_out, data_in, value, index
            )
        else:
            basis_adjoint = backend.pytorch2numba(basis_adjoint)
            _interpolate_lowrank_cuda1[blockspergrid, threadsperblock](
                data_out, data_in, value, index, basis_adjoint
            )
            basis_adjoint = backend.numba2pytorch(basis_adjoint)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)

    def _do_interpolation_cuda2(
        data_out, data_in, value, index, basis_adjoint, threadsperblock
    ):
        # calculate size
        _, batch_size, _, _ = data_in.shape
        nframes = data_out.shape[0]
        npts = data_out.shape[-1]

        # define number of blocks
        blockspergrid = (
            (nframes * batch_size * npts) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)

        # run kernel
        if basis_adjoint is None:
            _interpolate_cuda2[blockspergrid, threadsperblock](
                data_out, data_in, value, index
            )
        else:
            basis_adjoint = backend.pytorch2numba(basis_adjoint)
            _interpolate_lowrank_cuda2[blockspergrid, threadsperblock](
                data_out, data_in, value, index, basis_adjoint
            )
            basis_adjoint = backend.numba2pytorch(basis_adjoint)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)

    def _do_interpolation_cuda3(
        data_out, data_in, value, index, basis_adjoint, threadsperblock
    ):
        # calculate size
        _, batch_size, _, _, _ = data_in.shape
        nframes = data_out.shape[0]
        npts = data_out.shape[-1]

        # define number of blocks
        blockspergrid = (
            (nframes * batch_size * npts) + (threadsperblock - 1)
        ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)

        # run kernel
        if basis_adjoint is None:
            _interpolate_cuda3[blockspergrid, threadsperblock](
                data_out, data_in, value, index
            )
        else:
            basis_adjoint = backend.pytorch2numba(basis_adjoint)
            _interpolate_lowrank_cuda3[blockspergrid, threadsperblock](
                data_out, data_in, value, index, basis_adjoint
            )
            basis_adjoint = backend.numba2pytorch(basis_adjoint)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)

    # main handle
    do_interpolation_cuda = [
        _do_interpolation_cuda1,
        _do_interpolation_cuda2,
        _do_interpolation_cuda3,
    ]
