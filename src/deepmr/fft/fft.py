"""FFT subroutines."""

__all__ = ["fft", "ifft"]

import numpy as np
import torch


def fft(input, axes=None, norm="ortho"):
    """
    Centered Fast Fourier Transform.

    Adapted from [1].

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input signal.
    axes : Iterable[int], optional
        Axes over which to compute the FFT.
        If not specified, apply FFT over all the axes.
    norm : str, optional
        FFT normalization. The default is ``ortho``.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Output signal.

    Examples
    --------
    >>> import torch
    >>> import deepmr

    First, create test image:

    >>> image = torch.zeros(32, 32, dtype=torch.complex64)
    >>> image = image[16, 16] = 1.0

    We now perform a 2D FFT:

    >>> kspace = deepmr.fft.fft(image)

    We can visualize the data:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 2)
    >>> im = ax[0].imshow(abs(image))
    >>> ax[0].set_title("Image", color="orangered", fontweight="bold")
    >>> ax[0].axis("off")
    >>> ax[0].set_alpha(0.0)
    >>> fig.colorbar(im, ax=ax[0], shrink=0.5)
    >>> ksp = ax[1].imshow(abs(kspace))
    >>> ax[1].set_title("k-Space", color="orangered", fontweight="bold")
    >>> ax[1].axis("off")
    >>> ax[1].set_alpha(0.0)
    >>> fig.colorbar(ksp, ax=ax[1], shrink=0.5)
    >>> plt.show()

    References
    ----------
    [1] https://github.com/mikgroup/sigpy

    """
    # check if we are using numpy arrays
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # make sure this is a tensor
    input = torch.as_tensor(input)
    ax = _normalize_axes(axes, input.ndim)
    output = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(input, dim=ax), dim=ax, norm=norm), dim=ax
    )

    if isnumpy:
        output = np.asarray(output)

    return output


def ifft(input, axes=None, norm="ortho"):
    """
    Centered inverse Fast Fourier Transform.

    Adapted from [1].

    Parameters
    ----------
    input :  np.ndarray | torch.Tensor
        Input signal.
    axes : Iterable[int]
        Axes over which to compute the iFFT.
        If not specified, apply iFFT over all the axes.
    norm : str, optional
        FFT normalization. The default is ``ortho``.

    Returns
    -------
    output :  np.ndarray | torch.Tensor
        Output signal.

    Examples
    --------
    >>> import torch
    >>> import deepmr

    First, create test image:

    >>> kspace = torch.ones(32, 32, dtype=torch.complex64)

    We now perform a 2D iFFT:

    >>> image = deepmr.fft.ifft(kspace)

    We can visualize the data:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 2)
    >>> ksp = ax[1].imshow(abs(kspace))
    >>> ax[0].set_title("k-Space", color="orangered", fontweight="bold")
    >>> ax[0].axis("off")
    >>> ax[0].set_alpha(0.0)
    >>> fig.colorbar(ksp, ax=ax[0], shrink=0.5)
    >>> im = ax[0].imshow(abs(image))
    >>> ax[1].set_title("Image", color="orangered", fontweight="bold")
    >>> ax[1].axis("off")
    >>> ax[1].set_alpha(0.0)
    >>> fig.colorbar(im, ax=ax[1], shrink=0.5)
    >>> plt.show()

    References
    ----------
    [1] https://github.com/mikgroup/sigpy

    """
    # check if we are using numpy arrays
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # make sure this is a tensor
    input = torch.as_tensor(input)
    ax = _normalize_axes(axes, input.ndim)
    output = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(input, dim=ax), dim=ax, norm=norm), dim=ax
    )

    if isnumpy:
        output = np.asarray(output)

    return output


# %% local subroutines
def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))
