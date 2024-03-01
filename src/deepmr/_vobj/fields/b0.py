"""B0 field maps generation routines."""

__all__ = ["b0field"]

import numpy as np
import torch

from ... import fft


def b0field(chi, b0range=(-200, 200), mask=None):
    """
    Simulate inhomogeneous B0 fields.

    Output field units is ``[Hz]``. The field
    is created by convolving the dipole kernel with an input
    magnetic susceptibility map.

    Parameters
    ----------
    chi : np.ndarray | torch.Tensor
        Object magnetic susceptibility map in ``[ppb]`` of
        shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
    b0range : Iterable[float]
        Range of B0 field in ``[Hz]``. The default is ``(-200, 200)``.
    mask : np.ndarray | torch.Tensor, optional
        Region of support of the object of
        shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
        The default is ``None``.

    Returns
    -------
    B0map : torch.Tensor
        Spatially varying B0 maps of shape ``(ny, nx)`` (2D)
        or ``(nz, ny, nx)`` (3D) in ``[Hz]``, arising from the object susceptibility.

    Example
    -------
    >>> import deepmr

    We can generate a 2D B0 field map of shape ``(ny=128, nx=128)`` starting from a
    magnetic susceptibility distribution:

    >>> chi = deepmr.shepp_logan(128, qmr=True)["chi"]
    >>> b0map = deepmr.b0field(chi)

    B0 values range can be specified using ``b0range`` argument:

    >>> b0map = deepmr.b0field(chi, b0range=(-500, 500))

    """
    # make sure this is a torch tensor
    chi = torch.as_tensor(chi, dtype=torch.float32)

    # get input shape
    ishape = chi.shape

    # get k space coordinates
    kgrid = [
        np.arange(-ishape[n] // 2, ishape[n] // 2, dtype=np.float32)
        for n in range(len(ishape))
    ]
    kgrid = np.meshgrid(*kgrid, indexing="ij")
    kgrid = np.stack(kgrid, axis=-1)

    knorm = (kgrid**2).sum(axis=-1) + np.finfo(np.float32).eps
    dipole_kernel = 1 / 3 - (kgrid[..., 0] ** 2 / knorm)
    dipole_kernel = torch.as_tensor(dipole_kernel, dtype=torch.float32)

    # apply convolution
    B0map = fft.ifft(dipole_kernel * fft.fft(chi)).real

    # rescale
    B0map = B0map - B0map.min()  # (min, max) -> (0, max - min)
    B0map = B0map / B0map.max()  # (0, max - min) -> (0, 1)
    B0map = (
        B0map * (b0range[1] - b0range[0]) + b0range[0]
    )  # (0, 1) -> (b0range[0], b0range[1])

    # mask
    if mask is not None:
        mask = torch.as_tensor(mask != 0)
        B0map = mask * B0map

    return torch.as_tensor(B0map, dtype=torch.float32)
