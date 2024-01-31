"""2D and 3D gridding (non-uniform -> uniform) subroutines."""

__all__ = ["apply_gridding"]

import gc

import numpy as np
import numba as nb
import torch

from . import backend

def apply_gridding(data_in, sparse_coeff,  basis=None, threadsperblock=128, device=None):
    """
    Gridding of points specified by coordinates to array.

    Parameters
    ----------
    data_in : torch.Tensor
        Input Non-Cartesian array of shape ``(..., ncontrasts, nviews, nsamples)``.
    sparse_coeff : dict
        Pre-calculated interpolation coefficients in sparse COO format.
    adjoint_basis : torch.Tensor, optional
        Low rank subspace projection operator 
        of shape ``(ncoeff, ncontrasts)``; can be ``"None"``. The default is ``"None"``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is 128.
    device : str, optional
        Computational device (``"cpu"`` or ``"cuda:n"``, with ``n=0, 1,...nGPUs``).
        The default is ``"None" ``(same as interpolator).

    Returns
    -------
    data_out : torch.Tensor
        Output Cartesian array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

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

    # get number of coefficients
    if basis is not None:
        ncoeff = basis.shape[0]
    else:
        ncoeff = nframes

    # argument reshape
    data_in = data_in.reshape([batch_size, nframes, npts])
    data_in = data_in.swapaxes(0, 1)
    
    # collect garbage
    gc.collect()

    # preallocate output data
    data_out = torch.zeros((ncoeff, batch_size, *dshape), dtype=data_in.dtype, device=device)

    # do actual gridding
    if device == 'cpu':
        do_gridding[ndim-2](data_out, data_in, value, index, basis)
    else:
        do_gridding_cuda[ndim-2](data_out, data_in, value, index, basis, threadsperblock)
        
    # collect garbage
    gc.collect()

    # reformat for output
    if nframes == 1:
        data_out = data_out[0].reshape([*batch_shape, *ishape[1:]])
    else:
        data_out = data_out.swapaxes(0, 1)
        data_out = data_out.reshape([*batch_shape, ncoeff, *ishape[1:]])

    return data_out / scale

def _do_gridding2(data_out, data_in, value, index, basis):
    """2D Gridding routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)
    value = [backend.pytorch2numba(val) for val in value]
    index = [backend.pytorch2numba(ind) for ind in index]

    if basis is None:
        _gridding2(data_out, data_in, value, index)
    else:
        basis = backend.pytorch2numba(basis)
        _gridding_lowrank2(data_out, data_in, value, index, basis)
        basis = backend.numba2pytorch(basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)
    value = [backend.numba2pytorch(val) for val in value]
    index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]

def _do_gridding3(data_out, data_in, value, index, basis):
    """3D Gridding routine wrapper."""
    data_out = backend.pytorch2numba(data_out)
    data_in = backend.pytorch2numba(data_in)
    value = [backend.pytorch2numba(val) for val in value]
    index = [backend.pytorch2numba(ind) for ind in index]

    if basis is None:
        _gridding3(data_out, data_in, value, index)
    else:
        basis = backend.pytorch2numba(basis)
        _gridding_lowrank3(data_out, data_in, value, index, basis)
        basis = backend.numba2pytorch(basis)

    data_out = backend.numba2pytorch(data_out)
    data_in = backend.numba2pytorch(data_in)
    value = [backend.numba2pytorch(val) for val in value]
    index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]

# main handle
do_gridding = [_do_gridding2, _do_gridding3]

#%% subroutines
@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _gridding2(cart_data, noncart_data, interp_value, interp_index): # noqa

    # get sizes
    nframes, batch_size, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    yindex, xindex = interp_index
    yvalue, xvalue = interp_value

    # get interpolator width
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames and batches
    for i in nb.prange(nframes*batch_size):  # pylint: disable=not-an-iterable

        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):

            # spread data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    cart_data[frame, batch, idy, idx] += val * noncart_data[frame, batch, point]

    return cart_data

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _gridding3(cart_data, noncart_data, interp_value, interp_index): # noqa

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

    # parallelize over frames and batches
    for i in nb.prange(nframes*batch_size):  # pylint: disable=not-an-iterable

        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):

            # spread data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        cart_data[frame, batch, idz, idy, idx] += val * noncart_data[frame, batch, point]

    return cart_data

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _gridding_lowrank2(cart_data, noncart_data, interp_value, interp_index, basis): # noqa

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

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff*batch_size):  # pylint: disable=not-an-iterable

        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):

            # iterate over non-cartesian point of current frame
            for point in range(npts):

                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while spreading data
                        cart_data[coeff, batch, idy, idx] += \
                            val * basis[coeff, frame] * \
                            noncart_data[frame, batch, point]

    return cart_data

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _gridding_lowrank3(cart_data, noncart_data, interp_value, interp_index, basis): # noqa

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

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff*batch_size):  # pylint: disable=not-an-iterable

        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):

            # iterate over non-cartesian point of current frame
            for point in range(npts):

                # spread data within kernel radius
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
                            cart_data[coeff, batch, idz, idy, idx] += \
                                val * basis[coeff, frame] * \
                                noncart_data[frame, batch, point]

    return cart_data

# %% CUDA
if torch.cuda.is_available():
    
    __all__.extend(["_gridding2_cuda", "_gridding3_cuda", "_gridding_lowrank2_cuda", "_gridding_lowrank3_cuda"])
    
    @nb.cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_real(output, index, value):
        nb.cuda.atomic.add(output, index, value)

    @nb.cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_complex(output, index, value):
        nb.cuda.atomic.add(output.real, index, value.real)
        nb.cuda.atomic.add(output.imag, index, value.imag)
        
    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _gridding_cuda2(cart_data, noncart_data, interp_value, interp_index, iscomplex):
        
        # get function
        if iscomplex:
            _update = _update_complex
        else:
            _update = _update_real

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

            # spread data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    _update(cart_data, (frame, batch, idy, idx), val * noncart_data[frame, batch, point])

        return cart_data

    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _gridding_cuda3(cart_data, noncart_data, interp_value, interp_index, iscomplex):
        
        # get function
        if iscomplex:
            _update = _update_complex
        else:
            _update = _update_real

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

            # spread data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        _update(cart_data, (frame, batch, idz, idy, idx), val * noncart_data[frame, batch, point])

        return cart_data
    
    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _gridding_lowrank_cuda2(cart_data, noncart_data, interp_value, interp_index, basis, iscomplex):
        
        # get function
        if iscomplex:
            _update = _update_complex
        else:
            _update = _update_real

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

            # spread data within kernel radius
            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    # do adjoint low rank projection (low-rank subspace -> time domain)
                    # while spreading data
                    for coeff in range(ncoeff):
                        _update(cart_data, (coeff, batch, idy, idx), val * basis[coeff, frame] * noncart_data[frame, batch, point])

        return cart_data

    @nb.cuda.jit(fastmath=True)  # pragma: no cover
    def _gridding_lowrank_cuda3(cart_data, noncart_data, interp_value, interp_index, basis, iscomplex):
        
        # get function
        if iscomplex:
            _update = _update_complex
        else:
            _update = _update_real

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

            # spread data within kernel radius
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
                            _update(cart_data, (coeff, batch, idz, idy, idx), val * basis[coeff, frame] * noncart_data[frame, batch, point])

        return cart_data
    
    def _do_gridding_cuda2(data_out, data_in, value, index, basis, threadsperblock):
        # get if function is complex
        is_complex = torch.is_complex(data_in)

        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1)) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)
        value = [backend.pytorch2numba(val) for val in value]
        index = [backend.pytorch2numba(ind) for ind in index]

        # run kernel
        if basis is None:
            _gridding_cuda2[blockspergrid, threadsperblock](data_out, data_in, value, index, is_complex)
        else:
            basis = backend.pytorch2numba(basis)
            _gridding_lowrank_cuda2[blockspergrid, threadsperblock](data_out, data_in, value, index, basis, is_complex)
            basis = backend.numba2pytorch(basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)
        value = [backend.numba2pytorch(val) for val in value]
        index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]

    def _do_gridding_cuda3(data_out, data_in, value, index, basis, threadsperblock):
        # get if function is complex
        is_complex = torch.is_complex(data_in)

        # define number of blocks
        blockspergrid = (data_out.size + (threadsperblock - 1) ) // threadsperblock

        data_out = backend.pytorch2numba(data_out)
        data_in = backend.pytorch2numba(data_in)
        value = [backend.pytorch2numba(val) for val in value]
        index = [backend.pytorch2numba(ind) for ind in index]

        # run kernel
        if basis is None:
            _gridding_cuda3[blockspergrid, threadsperblock](data_out, data_in, value, index, is_complex)
        else:
            basis = backend.pytorch2numba(basis)
            _gridding_lowrank_cuda3[blockspergrid, threadsperblock](data_out, data_in, value, index, basis, is_complex)
            basis = backend.numba2pytorch(basis)

        data_out = backend.numba2pytorch(data_out)
        data_in = backend.numba2pytorch(data_in)
        value = [backend.numba2pytorch(val) for val in value]
        index = [backend.numba2pytorch(ind, requires_grad=False) for ind in index]
    
    # main handle
    do_gridding_cuda = [_do_gridding_cuda2, _do_gridding_cuda3]