"""2D and 3D interpolation (uniform -> non-uniform) subroutines."""

__all__ = ["apply_interpolation"]

import gc

import numpy as np
import numba as nb
import torch

from . import backend

def apply_interpolation(data_in, sparse_coeff, adjoint_basis=None, threadsperblock=128, device=None):
    """
    Interpolation from array to points specified by coordinates.

    Parameters
    ----------
    data_in : torch.Tensor
        Input Cartesian array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    sparse_coeff : dict
        Pre-calculated interpolation coefficients in sparse COO format.
    adjoint_basis : torch.Tensor, optional
        Adjoint low rank subspace projection operator 
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``None`` (same as interpolator).

    Returns
    -------
    data_out : torch.Tensor
        Output Non-Cartesian array of shape ``(..., ncontrasts, nviews, nsamples)``.
                
    """
    # convert to tensor if nececessary
    data_in = torch.as_tensor(data_in, dtype=torch.float32)
    
    # cast tp device is necessary
    if device is not None:
        sparse_coeff.to(device)
    
    # unpack input
    index = sparse_coeff.index
    value = sparse_coeff.value
    dshape = sparse_coeff.dshape
    ishape = sparse_coeff.ishape
    ndim = sparse_coeff.ndim
    scale = sparse_coeff.scale
    device = sparse_coeff.device
    
    # cast to device
    data_in = data_in.to(device)

    # get input sizes    
    nframes = index.shape[0]
    npts = np.prod(ishape)

    # reformat data for computation
    if nframes == 1:
        batch_shape = data_in.shape[:-ndim]
    else:
        batch_shape = data_in.shape[:-ndim-1]
    batch_size = np.prod(batch_shape)  # ncoils * nslices * [int]

    # reshape
    data_in = data_in.reshape([batch_size, nframes, *dshape])
    data_in = data_in.swapaxes(0, 1)
    
    # collect garbage
    gc.collect()

    # preallocate output data
    data_out = torch.zeros((nframes, batch_size, npts), dtype=data_in.dtype, device=device)

    # do actual interpolation
    if device == 'cpu':
        do_interpolation[ndim-2](data_out, data_in, value, index, adjoint_basis)
    else:
        do_interpolation_cuda[ndim-2](data_out, data_in, value, index, adjoint_basis, threadsperblock)
        
    # collect garbage
    gc.collect()

    # reformat for output
    if nframes == 1:
        data_out = data_out[0].reshape([*batch_shape, *ishape[1:]])
    else:
        data_out = data_out.swapaxes(0, 1)
        data_out = data_out.reshape([*batch_shape, nframes, *ishape[1:]])

    return data_out / scale

#%% subroutines
@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate2(noncart_data, cart_data, interp_value, interp_index): # noqa

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
    for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

        # get current frame and k-space index
        frame = i // (batch_size*npts)
        tmp = i % (batch_size*npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        for i_y in range(ywidth):
            idy = yindex[frame, point, i_y]
            valy = yvalue[frame, point, i_y]

            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = valy * xvalue[frame, point, i_x]

                noncart_data[frame, batch, point] += val * cart_data[frame, batch, idy, idx]

    return noncart_data

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate3(noncart_data, cart_data, interp_value, interp_index):  # noqa

    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = interp_index
    zvalue, yvalue, xvalue = interp_value

    # get interpolator width
    zwidth = zindex.shape[-1]
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

        # get current frame and k-space index
        frame = i // (batch_size*npts)
        tmp = i % (batch_size*npts)
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

                    noncart_data[frame, batch, point] += val * cart_data[frame, batch, idz, idy, idx]

    return noncart_data

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate_lowrank2(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):  # noqa

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
    for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

        # get current frame and k-space index
        frame = i // (batch_size*npts)
        tmp = i % (batch_size*npts)
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
                    noncart_data[frame, batch, point] += val * adjoint_basis[frame, coeff] * cart_data[coeff, batch, idy, idx]

    return noncart_data

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interpolate_lowrank3(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):  # noqa

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
    for i in nb.prange(nframes*batch_size*npts):  # pylint: disable=not-an-iterable

        # get current frame and k-space index
        frame = i // (batch_size*npts)
        tmp = i % (batch_size*npts)
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
                        noncart_data[frame, batch, point] += val * adjoint_basis[frame, coeff] * cart_data[coeff, batch, idz, idy, idx]

    return noncart_data

def _do_interpolation2(data_out, data_in, value, index, adjoint_basis):
    """2D Interpolation routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)
    value = [backend.pytorch2numba(val) for val in value]
    index = [backend.pytorch2numba(ind) for ind in index]

    if adjoint_basis is None:
        _interpolate2(data_out, data_in, value, index)
    else:
        adjoint_basis = backend.pytorch2numba(adjoint_basis)
        _interpolate_lowrank2(data_out, data_in, value, index, adjoint_basis)
        adjoint_basis = backend.numba2pytorch(adjoint_basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)
    value = [backend.numba2pytorch(val) for val in value]
    index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]

def _do_interpolation3(data_out, data_in, value, index, adjoint_basis):
    """3D Interpolation routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)
    value = [backend.pytorch2numba(val) for val in value]
    index = [backend.pytorch2numba(ind) for ind in index]

    if adjoint_basis is None:
        _interpolate3(data_out, data_in, value, index)
    else:
        adjoint_basis = backend.pytorch2numba(adjoint_basis)
        _interpolate_lowrank3(data_out, data_in, value, index, adjoint_basis)
        adjoint_basis = backend.numba2pytorch(adjoint_basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)
    value = [backend.numba2pytorch(val) for val in value]
    index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]

# main handle
do_interpolation = [_do_interpolation2, _do_interpolation3]

# %% CUDA
if torch.cuda.is_available():
    
    __all__.extend(["_interpolate_cuda2", "_interpolate_cuda3", "_interpolate_lowrank_cuda2", "_interpolate_lowrank_cuda3"])
    
    @nb.cuda.jit(fastmath=True)  # pragma: no cover
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
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes*batch_size*npts:

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    noncart_data[frame, batch, point] += val * cart_data[frame, batch, idy, idx]

        return noncart_data

    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_cuda3(noncart_data, cart_data, interp_value, interp_index):

        # get sizes
        nframes, batch_size, _, _ = cart_data.shape
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        zvalue, yvalue, xvalue = interp_value

        # get interpolator width
        zwidth = zindex.shape[-1]
        ywidth = yindex.shape[-1]
        xwidth = xindex.shape[-1]

        # parallelize over frames, batches and k-space points
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes*batch_size*npts:

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
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

                        noncart_data[frame, batch, point] += val * cart_data[frame, batch, idz, idy, idx]

        return noncart_data
    
    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_lowrank_cuda2(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):

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
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes*batch_size*npts:

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
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
                        noncart_data[frame, batch, point] += val * adjoint_basis[frame, coeff] * cart_data[coeff, batch, idy, idx]

        return noncart_data

    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolate_lowrank_cuda3(noncart_data, cart_data, interp_value, interp_index, adjoint_basis):

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
        i = nb.cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes*batch_size*npts:

            # get current frame and k-space index
            frame = i // (batch_size*npts)
            tmp = i % (batch_size*npts)
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
                            noncart_data[frame, batch, point] += val * adjoint_basis[frame, coeff] * cart_data[coeff, batch, idz, idy, idx]

        return noncart_data
    
    def _do_interpolation_cuda2(data_out, data_in, value, index, adjoint_basis, threadsperblock):
        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)
        value = [backend.pytorch2numba(val) for val in value]
        index = [backend.pytorch2numba(ind) for ind in index]

        # run kernel
        if adjoint_basis is None:
            _interpolate_cuda2[blockspergrid, threadsperblock](data_out, data_in, value, index)
        else:
            adjoint_basis = backend.pytorch2numba(adjoint_basis)
            _interpolate_lowrank_cuda2[blockspergrid, threadsperblock](data_out, data_in, value, index, adjoint_basis)
            adjoint_basis = backend.numba2pytorch(adjoint_basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)
        value = [backend.numba2pytorch(val) for val in value]
        index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]
        
    def _do_interpolation_cuda3(data_out, data_in, value, index, adjoint_basis, threadsperblock):
        
        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)
        value = [backend.pytorch2numba(val) for val in value]
        index = [backend.pytorch2numba(ind) for ind in index]

        # run kernel
        if adjoint_basis is None:
            _interpolate_cuda3[blockspergrid, threadsperblock](data_out, data_in, value, index)
        else:
            adjoint_basis = backend.pytorch2numba(adjoint_basis)
            _interpolate_lowrank_cuda3[blockspergrid, threadsperblock](data_out, data_in, value, index, adjoint_basis)
            adjoint_basis = backend.numba2pytorch(adjoint_basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)
        value = [backend.numba2pytorch(val) for val in value]
        index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]

    # main handle
    do_interpolation_cuda = [_do_interpolation_cuda2, _do_interpolation_cuda3]
