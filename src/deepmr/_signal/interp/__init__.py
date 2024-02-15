"""Interpolation routines."""

__all__ = ["interpolate", "gridding"]

from . import grid
from . import interp
from . import plan

from .grid import apply_gridding
from .interp import apply_interpolation
from .plan import plan_interpolator


def interpolate(
    data_in,
    coord,
    shape=None,
    basis_adjoint=None,
    device="cpu",
    threadsperblock=128,
    width=2,
    beta=1.0,
):
    """
    Interpolation from array to points specified by coordinates.

    Parameters
    ----------
    data_in : torch.Tensor
        Input Cartesian array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5 * shape)``.
    shape : int | Iterable[int], optional
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
        The default is ``None`` (grid size equals to input data size, i.e. ``osf = 1``).
    basis_adjoint : torch.Tensor, optional
        Adjoint low rank subspace projection operator
        of shape ``(ncoeffs, ncontrasts)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.
    width : int | Iterable[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``2``.
    beta : float | Iterable[float], optional
        Kaiser-Bessel beta parameter of shape ``(ndim,)``.
        If scalar, it is assumed equal for each axis.
        The default is ``1.0``.

    Returns
    -------
    data_out : torch.Tensor
        Output Non-Cartesian array of shape ``(..., ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Non-uniform coordinates axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, kernel width
    and Kaiser Bessel parameters are assumed to be ``(z, y, x)``.

    Coordinates tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

    * ``coord.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
    * ``coord.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # get number of dimensions
    ndim = coord.shape[-1]

    # get shape if not provided
    if shape is None:
        shape = data_in.shape[-ndim:]

    # plan interpolator
    interpolator = plan.plan_interpolator(coord, shape, width, beta, device)

    # perform actual interpolation
    return interp.apply_interpolation(
        data_in, interpolator, basis_adjoint, threadsperblock=threadsperblock
    )


def gridding(
    data_in,
    coord,
    shape,
    basis=None,
    device="cpu",
    threadsperblock=128,
    width=2,
    beta=1.0,
):
    """
    Gridding of points specified by coordinates to array.

    Parameters
    ----------
    data_in : torch.Tensor
        Input Non-Cartesian array of shape ``(..., ncontrasts, nviews, nsamples)``.
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5 * shape)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.
    width : int | Iterable[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``2``.
    beta : float | Iterable[float], optional
        Kaiser-Bessel beta parameter of shape ``(ndim,)``.
        If scalar, it is assumed equal for each axis.
        The default is ``1.0``.

    Returns
    -------
    data_out : torch.Tensor
        Output Cartesian array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    Notes
    -----
    Non-uniform coordinates axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, kernel width
    and Kaiser Bessel parameters are assumed to be ``(z, y, x)``.

    Coordinates tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

    * ``coord.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
    * ``coord.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # plan interpolator
    interpolator = plan.plan_interpolator(coord, shape, width, beta, device)

    # perform actual interpolation
    return grid.apply_gridding(
        data_in, interpolator, basis, threadsperblock=threadsperblock
    )
