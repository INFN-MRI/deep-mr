"""Sub-package containing Fast Fourier transform routines.

FFT routines include:
    
    * centered n-dimensional FFT and iFFT;
    * n-dimensional sparse uniform FFT/iFFT with embedded low rank subspace projection;
    * n-dimensional NUFFT with embedded low rank subspace projection.

"""

from . import fft as _fft
from . import sparse_fft as _sparse_fft
from . import nufft as _nufft

from .fft import *  # noqa
from .nufft import * # noqa
from .sparse_fft import * # noqa

__all__ = _fft.__all__
__all__.extend(["sparse_fft", "sparse_ifft", "nufft", "nufft_adj"])

def sparse_fft(
    image,
    indexes,
    basis_adjoint=None,
    device="cpu",
    threadsperblock=128,
):
    """
    N-dimensional sparse Fast Fourier Transform.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    indexes : torch.Tensor
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
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

    Returns
    -------
    kspace : torch.Tensor
        Output sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Sampled points indexes axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape is assumed to be ``(z, y, x)``.

    Indexes tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # get number of dimensions
    ndim = indexes.shape[-1]

    # get shape if not provided
    shape = image.shape[-ndim:]

    # plan interpolator
    sampling_mask = _sparse_fft.prepare_sampling(indexes, shape, device)

    # perform actual interpolation
    return _sparse_fft.apply_sparse_fft(
        image, sampling_mask, basis_adjoint, threadsperblock=threadsperblock
    )


def sparse_ifft(
    kspace,
    indexes,
    shape,
    basis=None,
    device="cpu",
    threadsperblock=128,
):
    """
    N-dimensional inverse sparse Fast Fourier Transform.

    Parameters
    ----------
    kspace : torch.Tensor
        Input sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    indexes : torch.Tensor
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
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

    Returns
    -------
    image : torch.Tensor
        Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    Notes
    -----
    Sampled points indexes axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape is assumed to be ``(z, y, x)``.

    Sampled points indexes axes ordering is assumed to be ``(x, y)`` for 2D signals
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # plan interpolator
    sampling_mask = _sparse_fft.prepare_sampling(indexes, shape, device)

    # perform actual interpolation
    return _sparse_fft.apply_sparse_ifft(
        kspace, sampling_mask, basis, threadsperblock=threadsperblock
    )


def nufft(
    image,
    coord,
    shape=None,
    basis_adjoint=None,
    device="cpu",
    threadsperblock=128,
    width=3,
    oversamp=1.125,
):
    """
    N-dimensional Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
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
    oversamp : float | Iterable[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.125``.

    Returns
    -------
    kspace : torch.Tensor
        Output Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

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
        shape = image.shape[-ndim:]

    # plan interpolator
    nufft_plan = _nufft.plan_nufft(coord, shape, width, oversamp, device)

    # perform actual interpolation
    return _nufft.apply_nufft(
        image, nufft_plan, basis_adjoint, threadsperblock=threadsperblock
    )


def nufft_adj(
    kspace,
    coord,
    shape,
    basis=None,
    device="cpu",
    threadsperblock=128,
    width=3,
    oversamp=1.125,
):
    """
    N-dimensional adjoint Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    kspace : torch.Tensor
        Input Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5  * shape)``.
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
    oversamp : float | Iterable[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.125``.

    Returns
    -------
    image : torch.Tensor
        Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
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
    nufft_plan = _nufft.plan_nufft(coord, shape, width, oversamp, device)

    # perform actual interpolation
    return _nufft.apply_nufft_adj(
        kspace, nufft_plan, basis, threadsperblock=threadsperblock
    )

