"""Utils for image  intensity rescaling."""

__all__ = ["intensity_scaling"]

import torch

from ... import fft as _fft
from ... import _signal


def intensity_scaling(input, ndim):
    """
    Rescale intensity range of the input image.

    This has the main purpose of enabling easier tuning of
    regularization strength in interative reconstructions.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input signal of shape ``(..., ny, nx)`` (2D) or
        ``(..., nz, ny, nx)`` (3D).
    ndim : int, optional
        PNumber of spatial dimensions.

    Returns
    -------
    output : np.ndarray | torch.Tensor.
        Rescaled signal.

    """
    ksp = _fft.fft(torch.as_tensor(input), axes=range(-ndim, 0))
    ksp_lores = _signal.resize(ksp, ndim * [32])
    img_lores = _fft.ifft(ksp_lores, axes=range(-ndim, 0))
    for n in range(len(img_lores.shape) - ndim):
        img_lores = torch.linalg.norm(img_lores, axis=0)

    # get scaling
    img_lores = torch.nan_to_num(img_lores, posinf=0.0, neginf=0.0, nan=0.0)
    scale = torch.quantile(abs(img_lores.ravel()), 0.95)

    return input / scale
