"""Python implementation of the GRAPPA operator formalism. Adapted for convenience from PyGRAPPA"""

__all__ = ["grog_interp"]

from types import SimpleNamespace

import numpy as np
import numba as nb

import torch

import scipy

from .. import backend


def grog_interp(
    input, calib, coord, shape, lamda=0.01, nsteps=11, device=None, threadsperblock=128
):
    """
    GRAPPA Operator Gridding (GROP) interpolation of Non-Cartesian datasets.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    calib : np.ndarray | torch.Tensor
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    coord : np.ndarray | torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5 * shape)``.
    shape : Iterable[int]
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01``.
    nsteps : int, optional
        K-space interpolation grid discretization. Defaults to ``11``
        steps (i.e., ``dk = -0.5, -0.4, ..., 0.0, ..., 0.4, 0.5``)
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Output sparse Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    indexes : np.ndarray | torch.Tensor
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    weights : np.ndarray | torch.Tensor
        Number of occurrences of each k-space sample of shape ``(ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.

    """
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False
        input = torch.as_tensor(input)

    # get device
    if device is None:
        device = input.device

    # cast everything
    idevice = input.device
    input = input.to(device)
    coord = torch.as_tensor(coord, device=device)
    calib = torch.as_tensor(calib, device=device)

    # default to odds steps to explicitly have 0
    nsteps = 2 * (nsteps // 2) + 1

    # get number of spatial dimes
    ndim = coord.shape[-1]

    # get grappa operator
    kern = _calc_grappaop(calib, ndim, lamda)

    # get coord shape
    cshape = coord.shape

    # reshape coordinates
    ishape = input.shape
    input = input.reshape(*ishape[: -(len(cshape) - 1)], int(np.prod(cshape[:-1])))

    # bring coil axes to the front
    input = input[..., None]
    input = input.swapaxes(-3, -1)
    dshape = input.shape
    input = input.reshape(
        -1, *input.shape[-2:]
    )  # (nslices, nsamples, ncoils) or (nsamples, ncoils)

    # perform product
    deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)

    # get Gx, Gy, Gz
    Gx = _weight_grid(kern.Gx, deltas)  # (nslices, nsteps, nc, nc)
    Gy = _weight_grid(kern.Gy, deltas)  # (nslices, nsteps, nc, nc)
    if ndim == 3:
        Gz = _weight_grid(kern.Gz, deltas)  # (nslices, nsteps, nc, nc)
    else:
        Gz = None

    # build G
    if ndim == 2:
        Gx = Gx[None, ...]
        Gy = Gy[:, None, ...]
        Gx = np.repeat(Gx, nsteps, axis=0)
        Gy = np.repeat(Gy, nsteps, axis=1)
        Gx = Gx.reshape(-1, *Gx.shape[-3:])
        Gy = Gy.reshape(-1, *Gy.shape[-3:])
        G = Gx @ Gy
    elif ndim == 3:
        Gx = Gx[None, None, ...]
        Gy = Gy[None, :, None, ...]
        Gz = Gz[:, None, None, ...]
        Gx = np.repeat(Gx, nsteps, axis=0)
        Gx = np.repeat(Gx, nsteps, axis=1)
        Gy = np.repeat(Gy, nsteps, axis=0)
        Gy = np.repeat(Gy, nsteps, axis=2)
        Gz = np.repeat(Gz, nsteps, axis=1)
        Gz = np.repeat(Gz, nsteps, axis=2)
        Gx = Gx.reshape(-1, *Gx.shape[-2:])
        Gy = Gy.reshape(-1, *Gy.shape[-2:])
        Gz = Gz.reshape(-1, *Gz.shape[-2:])
        G = Gx @ Gy @ Gz

    # build indexes
    indexes = torch.round(coord)
    lut = indexes - coord
    lut = torch.floor(10 * lut).to(int) + int(nsteps // 2)
    lut = lut.reshape(-1, ndim)  # (nsamples, ndim)
    lut = lut * torch.as_tensor([1, nsteps, nsteps**2])[:ndim]
    lut = lut.sum(axis=-1)

    if ndim == 2:
        input = input.swapaxes(0, 1)  # (nsamples, nslices, ncoils)

    # perform interpolation
    if device == "cpu":
        output = do_interpolation(input, G, lut)
    else:
        output = do_interpolation_cuda(input, G, lut, threadsperblock)

    # finalize indexes
    if np.isscalar(shape):
        shape = [shape] * ndim
    indexes = indexes + torch.as_tensor(list(shape[-ndim:])[::-1]) // 2
    indexes = indexes.to(int)

    # flatten indexes
    unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[: ndim - 1]
    flattened_idx = torch.as_tensor(unfolding, dtype=int) * indexes
    flattened_idx = flattened_idx.sum(axis=-1).flatten()

    # count elements
    _, idx, counts = torch.unique(
        flattened_idx, return_inverse=True, return_counts=True
    )
    weights = counts[idx]

    # count
    # max_value = torch.max(weights)
    # counts = torch.bincount(weights, minlength=max_value+1)
    # weights = counts[weights]

    weights = weights.reshape(*indexes.shape[:-1])
    weights = weights.to(torch.float32)
    weights = 1 / weights

    # finalize data
    if ndim == 2:
        output = output.swapaxes(0, 1)  # (nslices, nsamples, ncoils)
    output = output.reshape(*dshape)
    output = output.swapaxes(-3, -1)
    output = output[..., 0]
    output = output.reshape(ishape)

    output = output.to(idevice)
    indexes = indexes.to(idevice)
    weights = weights.to(idevice)

    if isnumpy:
        output = output.numpy(force=True)
        indexes = indexes.to(idevice)
        weights = weights.to(idevice)

    return output, indexes, weights


# %% subroutines
def _calc_grappaop(calib, ndim, lamda):
    # as Tensor
    calib = torch.as_tensor(calib)

    # expand
    if len(calib.shape) == 3:  # single slice (nc, ny, nx)
        calib = calib[:, None, :, :].clone()

    # compute kernels
    if ndim == 2:
        gy, gx = _grappa_op_2d(calib, lamda)
    elif ndim == 3:
        gz, gy, gx = _grappa_op_3d(calib, lamda)

    # prepare output
    GrappaOp = SimpleNamespace()
    GrappaOp.Gx, GrappaOp.Gy = (gx.numpy(force=True), gy.numpy(force=True))

    if ndim == 3:
        GrappaOp.Gz = gz.numpy(force=True)
    else:
        GrappaOp.Gz = None

    return GrappaOp


def _grappa_op_2d(calib, lamda):
    """Return a batch of 2D GROG operators (one for each z)."""
    # coil axis in the back
    calib = torch.moveaxis(calib, 0, -1)
    nz, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sy = torch.reshape(calib[:, :-1, :, :], (nz, -1, nc))
    Sx = torch.reshape(calib[:, :, :-1, :], (nz, -1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Ty = torch.reshape(calib[:, 1:, :, :], (nz, -1, nc))
    Tx = torch.reshape(calib[:, :, 1:, :], (nz, -1, nc))

    # train the operators:
    Syh = Sy.conj().permute(0, 2, 1)
    lamda0 = lamda * torch.linalg.norm(Syh, dim=(1, 2)) / Syh.shape[1]
    Gy = torch.linalg.solve(
        _bdot(Syh, Sy) + lamda0[:, None, None] * torch.eye(Syh.shape[1])[None, ...],
        _bdot(Syh, Ty),
    )

    Sxh = Sx.conj().permute(0, 2, 1)
    lamda0 = lamda * torch.linalg.norm(Sxh, dim=(1, 2)) / Sxh.shape[1]
    Gx = torch.linalg.solve(
        _bdot(Sxh, Sx) + lamda0[:, None, None] * torch.eye(Sxh.shape[1])[None, ...],
        _bdot(Sxh, Tx),
    )

    return Gy.clone(), Gx.clone()


def _grappa_op_3d(calib, lamda):
    """Return 3D GROG operator."""
    # coil axis in the back
    calib = torch.moveaxis(calib, 0, -1)
    _, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sz = torch.reshape(calib[:-1, :, :, :], (-1, nc))
    Sy = torch.reshape(calib[:, :-1, :, :], (-1, nc))
    Sx = torch.reshape(calib[:, :, :-1, :], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tz = torch.reshape(calib[1:, :, :, :], (-1, nc))
    Ty = torch.reshape(calib[:, 1:, :, :], (-1, nc))
    Tx = torch.reshape(calib[:, :, 1:, :], (-1, nc))

    # train the operators:
    Szh = Sz.conj().permute(1, 0)
    lamda0 = lamda * torch.linalg.norm(Szh) / Szh.shape[0]
    Gz = torch.linalg.solve(Szh @ Sz + lamda0 * torch.eye(Szh.shape[0]), Szh @ Tz)

    Syh = Sy.conj().permute(1, 0)
    lamda0 = lamda * torch.linalg.norm(Syh) / Syh.shape[0]
    Gy = torch.linalg.solve(Syh @ Sy + lamda0 * torch.eye(Syh.shape[0]), Syh @ Ty)

    Sxh = Sx.conj().permute(1, 0)
    lamda0 = lamda * torch.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = torch.linalg.solve(Sxh @ Sx + lamda0 * torch.eye(Sxh.shape[0]), Sxh @ Tx)

    return Gz.clone(), Gy.clone(), Gx.clone()


def _bdot(a, b):
    return torch.einsum("...ij,...jk->...ik", a, b)


def _weight_grid(A, weight):
    return np.stack([_matrix_power(A, w) for w in weight], axis=0)


def _matrix_power(A, t):
    if len(A.shape) == 2:
        return scipy.linalg.fractional_matrix_power(A, t)
    else:
        return np.stack([scipy.linalg.fractional_matrix_power(a, t) for a in A])


def do_interpolation(noncart, G, lut):
    cart = torch.zeros(noncart.shape, dtype=noncart.dtype, device=noncart.device)
    cart = backend.pytorch2numba(cart)
    noncart = backend.pytorch2numba(noncart)
    lut = backend.pytorch2numba(lut)

    _interp(cart, noncart, G, lut)

    noncart = backend.numba2pytorch(noncart)
    cart = backend.numba2pytorch(cart)
    lut = backend.numba2pytorch(lut)

    return cart


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(out, in_a, in_b):
    row, col = in_b.shape

    for i in range(row):
        for j in range(col):
            out[j] += in_b[i][j] * in_a[j]

    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp(data_out, data_in, interp, lut):
    # get data dimension
    nsamples, batch_size, _ = data_in.shape

    for i in nb.prange(nsamples * batch_size):
        sample = i // batch_size
        batch = i % batch_size
        idx = lut[sample]

        _dot_product(
            data_out[sample][batch], data_in[sample][batch], interp[idx][batch]
        )


# %% CUDA
if torch.cuda.is_available():
    from numba import cuda

    def do_interpolation_cuda(noncart, G, lut, threadsperblock):
        # calculate size
        nsamples, batch_size, ncoils = noncart.shape

        # define number of blocks
        blockspergrid = (
            (nsamples * batch_size) + (threadsperblock - 1)
        ) // threadsperblock

        cart = torch.zeros(noncart.shape, dtype=noncart.dtype, device=noncart.device)
        cart = backend.pytorch2numba(cart)
        noncart = backend.pytorch2numba(noncart)
        lut = backend.pytorch2numba(lut)

        # run kernel
        _interp_cuda[blockspergrid, threadsperblock](cart, noncart, G, lut)

        noncart = backend.numba2pytorch(noncart)
        cart = backend.numba2pytorch(cart)
        lut = backend.numba2pytorch(lut)

        return cart

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_cuda(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_cuda(data_out, data_in, interp, lut):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = nb.cuda.grid(1)
        if i < nvoxels * batch_size:
            sample = i // batch_size
            batch = i % batch_size
            idx = lut[sample]

            _dot_product_cuda(
                data_out[sample][batch], data_in[sample][batch], interp[idx][batch]
            )
