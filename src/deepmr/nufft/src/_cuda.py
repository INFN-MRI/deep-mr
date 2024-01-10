# -*- coding: utf-8 -*-
"""
CUDA specific subroutines.

@author: Matteo Cencini
"""
# pylint: disable=not-an-iterable
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=protected-access
# pylint: disable=too-many-function-args
# pylint: disable=unsubscriptable-object
# pylint: disable=arguments-differ

from typing import Callable

from numba import cuda

from nufftorch.src import _common


_dot_product = cuda.jit(_common._dot_product, device=True, inline=True)


@cuda.jit(fastmath=True)  # pragma: no cover
def _batched_dot_product(data_out, data_in, matrix):

    n_points, batch_size, _ = data_in.shape

    i = cuda.grid(1)
    if i < n_points*batch_size:
        point = i // batch_size
        batch = i % batch_size
        _dot_product(data_out[point][batch], matrix[point], data_in[point][batch])


class _DeGridding:

    apply: Callable

    def __init__(self, data_size, kernel_tuple, basis_adjoint, threadsperblock):

        # unpack kernel dict
        kernel_sparse_coefficients = kernel_tuple
        kernel_width = kernel_tuple[-2]
        ndim = kernel_tuple[0].shape[0]

        # get kernel neighbourhood
        kernel_neighbourhood = cuda.to_device(_iterator._get_neighbourhood(*kernel_width))

        # calculate blocks per grid
        blockspergrid = int((data_size + (threadsperblock - 1)) // threadsperblock)

        # select correct sub-routine
        if basis_adjoint is None:
            callback = _DeGridding._get_callback(ndim)

            def _apply(self, noncart_data, cart_data):
                return callback[blockspergrid, threadsperblock](noncart_data, cart_data,
                                                                kernel_sparse_coefficients,
                                                                kernel_neighbourhood)

        else:
            callback = _DeGridding._get_lowrank_callback(ndim)

            def _apply(self, noncart_data, cart_data):
                return callback[blockspergrid, threadsperblock](noncart_data, cart_data,
                                                                kernel_sparse_coefficients,
                                                                kernel_neighbourhood,
                                                                basis_adjoint)

        # assign
        _DeGridding.__call__ = _apply
        
    @staticmethod
    def _get_callback(ndim):

        # kernel function
        kernel = _kernel._evaluate()[ndim-1]

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, target = _get_target_point(i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(frame, target, point, kvalue, kidx, grid_off)

                    # update
                    gather(noncart_data, cart_data, frame, batch, target, source, value)


        return _callback

    @staticmethod
    def _get_lowrank_callback(ndim):

        # kernel function
        kernel = _kernel._evaluate()[ndim-1]

        # iterator function
        _get_target_point = _iterator._get_noncart_points_parallelize_over_all

        # gather function
        gather = _gather._data_lowrank

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(noncart_data, cart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      basis_adjoint):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis_adjoint.shape[-1]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, grid_off = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, target = _get_target_point(i, batch_size, npts)

                # gather data within kernel radius
                for point in kernel_neighbourhood:
                    value, source = kernel(frame, target, point, kvalue, kidx, grid_off)

                    # update
                    gather(noncart_data, cart_data, frame, batch, target, source, value, basis_adjoint, ncoeff)


        return _callback


class _Gridding:

    apply: Callable

    def __init__(self, data_size, kernel_tuple, basis, threadsperblock):

        # unpack kernel dict
        kernel_sparse_coefficients = kernel_tuple
        kernel_width = kernel_tuple[-2]
        ndim = kernel_tuple[0].shape[0]
        
        # get kernel neighbourhood
        kernel_neighbourhood = cuda.to_device(_iterator._get_neighbourhood(*kernel_width))

        # calculate blocks per grid
        blockspergrid = int((data_size + (threadsperblock - 1)) // threadsperblock)

        # select correct sub-routine
        if basis is None:
            callback = _Gridding._get_callback(ndim)

            def _apply(self, cart_data, noncart_data):
                return callback[blockspergrid, threadsperblock](cart_data, noncart_data,
                                                                kernel_sparse_coefficients,
                                                                kernel_neighbourhood)

        else:
            callback = _Gridding._get_lowrank_callback(ndim)

            def _apply(self, cart_data, noncart_data):
                return callback[blockspergrid, threadsperblock](cart_data, noncart_data,
                                                                kernel_sparse_coefficients,
                                                                kernel_neighbourhood,
                                                                basis)

        # assign
        _Gridding.__call__ = _apply

    @staticmethod
    def _get_callback(ndim):
        
        # kernel function
        kernel = _kernel._evaluate()[ndim-1]
        
        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_all

        # spread function
        spread = _spread._data

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, gshape = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, source = _get_source_point(i, batch_size, npts)

                # spread data within kernel radius
                for point in kernel_neighbourhood:
                    value, target = kernel(frame, source, point, kvalue, kidx, gshape)

                    # update
                    spread(cart_data, noncart_data, frame, batch, source, target, value)


        return _callback

    @staticmethod
    def _get_lowrank_callback(ndim):
        
        # kernel function
        kernel = _kernel._evaluate()[ndim-1]

        # iterator function
        _get_source_point = _iterator._get_noncart_points_parallelize_over_all

        # spread function
        spread = _spread._data_lowrank

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _callback(cart_data, noncart_data,
                      kernel_sparse_coefficients,
                      kernel_neighbourhood,
                      basis):

            # get shapes
            nframes, batch_size, npts = noncart_data.shape
            ncoeff = basis.shape[0]

            # unpack kernel tuple: kernel value, index and width (x, y, z) + grid shape (nx, ny, nz)
            kvalue, kidx, _, gshape = kernel_sparse_coefficients

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)
            if i < nframes*batch_size*npts:

                # get current frame and k-space index
                frame, batch, source = _get_source_point(i, batch_size, npts)

                # spread data within kernel radius
                for point in kernel_neighbourhood:
                    value, target = kernel(frame, source, point, kvalue, kidx, gshape)

                    # update
                    spread(cart_data, noncart_data, frame, batch, source, target, value, basis, ncoeff)

        return _callback


class _iterator(_common._iterator):

    _get_noncart_points_parallelize_over_all = staticmethod(cuda.jit(
        _common._iterator._get_noncart_points_parallelize_over_all,
        device=True, inline=True))


class _kernel(_common._kernel):
    
    @staticmethod
    def _evaluate():
        
        # get base functions
        _evaluate_1d, _evaluate_2d, _evaluate_3d = _common._kernel._make_evaluate()
        
        # jit 
        _evaluate_1d = cuda.jit(_evaluate_1d, device=True, inline=True)
        _evaluate_2d = cuda.jit(_evaluate_2d, device=True, inline=True)
        _evaluate_3d = cuda.jit(_evaluate_3d, device=True, inline=True)
        
        return _evaluate_1d, _evaluate_2d, _evaluate_3d


class _gather(_common._gather):

    _data = staticmethod(
        cuda.jit(_common._gather._data, device=True, inline=True))

    _data_lowrank = staticmethod(
        cuda.jit(_common._gather._data_lowrank, device=True, inline=True))


@cuda.jit(device=True, inline=True)  # pragma: no cover
def _update(output, index, value):
    cuda.atomic.add(
        output.real, index, value.real)
    cuda.atomic.add(
        output.imag, index, value.imag)
        
      
class _spread:

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _data(data_out, data_in,
              frame, batch, index_in,
              index_out, kernel_value):

        # get input and output locations
        idx_in = (frame, batch, index_in)
        idx_out = (frame, batch, index_out)

        # update data point
        _update(data_out, idx_out, kernel_value * data_in[idx_in])

    @staticmethod
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _data_lowrank(data_out, data_in,
                      frame, batch, index_in,
                      index_out, kernel_value,
                      basis, ncoeff):

        # get input locations
        idx_in = (frame, batch, index_in)

        # iterate over low rank coefficients
        for coeff in range(ncoeff):
            # get output frame
            idx_out = (coeff, batch, index_out)

            # get total weight
            weight = kernel_value * basis[coeff, frame]

            # update data point
            _update(data_out, idx_out, weight * data_in[idx_in])
