"""Sampling/Zerofill routines."""

__all__ = ["sample", "zerofill"]

from . import dense2sparse
from . import sparse2dense
from . import plan

from .dense2sparse import apply_sampling  # noqa
from .sparse2dense import apply_zerofill  # noqa
from .plan import plan_sampling  # noqa
from .toeplitz import *  # noqa


def sample(
    data_in,
    indexes,
    basis_adjoint=None,
    device="cpu",
    threadsperblock=128,
):
    """
    Interpolation from array to points specified by coordinates.

    Parameters
    ----------
    data_in : torch.Tensor
        Input zero-filled array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    indexes : torch.Tensor
        Non-zero data entries of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Indexes must be normalized between ``(0, N_i)``, with ``i=x,y,z``
        being the number of voxels along the corresponding axis.
    basis_adjoint : torch.Tensor, optional
        Adjoint low rank subspace projection operator
        of shape ``(ncoeffs, ncontrasts)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    data_out : torch.Tensor
        Output sparse array of shape ``(..., ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Sampling mask axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, is
    assumed to be ``(z, y, x)``.

    Sampling tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # get number of dimensions
    ndim = indexes.shape[-1]

    # get shape if not provided
    shape = data_in.shape[-ndim:]

    # plan interpolator
    indexes = plan.plan_sampling(indexes, shape, device)

    # perform actual interpolation
    return dense2sparse.apply_sampling(
        data_in, indexes, basis_adjoint, threadsperblock=threadsperblock
    )


def zerofill(
    data_in,
    indexes,
    shape,
    basis=None,
    device="cpu",
    threadsperblock=128,
):
    """
    Gridding of points specified by coordinates to array.

    Parameters
    ----------
    data_in : torch.Tensor
        Input sparse array of shape ``(..., ncontrasts, nviews, nsamples)``.
    indexes : torch.Tensor
        Non-zero data entries of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Indexes must be normalized between ``(0, N_i)``, with ``i=x,y,z``
        being the number of voxels along the corresponding axis.
    shape : int | Iterable[int]
        Grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    data_out : torch.Tensor
        Output zero-filled array of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    Notes
    -----
    Sampling mask axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, is
    assumed to be ``(z, y, x)``.

    Sampling tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # plan interpolator
    mask = plan.plan_sampling(indexes, shape, device)

    # perform actual interpolation
    return sparse2dense.apply_zerofill(
        data_in, mask, basis, threadsperblock=threadsperblock
    )
