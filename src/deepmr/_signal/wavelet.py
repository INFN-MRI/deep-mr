"""
Wavelet transform routines; adapted from Sigpy [1].


References
----------
[1] https://github.com/mikgroup/sigpy/tree/main

"""

__all__ = ["fwt", "iwt"]

import torch
import numpy as np

import ptwt
import pywt

from .resize import resize


def fwt(input, ndim=None, device=None, wave_name="db4", level=None):
    """
    Forward wavelet transform.

    Adapted from Sigpy [1].

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input signal of shape (..., nz, ny, nx).
    ndim : int, optional
        Number of spatial dimensions over to which compute
        wavelet transform (``1``, ``2``, ``3``).
        Assume spatial axis are the rightmost ones.
        The default is ``None`` (``ndim = min(3, len(input.shape))``).
    device : str, optional
        Computational device for Wavelet transform.
        If not specified, use ``input.device``.
        The default is ``None``.
    wave_name : str, optional
        Wavelet name. The default is ``"db4"``.
    axes : Iterable[int], optional
        Axes to perform wavelet transform.
        The default is ``None`` (all axes).
    level : int, optional
        Number of wavelet levels. The default is ``None``.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Output wavelet decomposition.
    shape : Iterable[int]
        Input signal shape (``input.shape``) for synthesis.

    Examples
    --------
    >>> import torch
    >>> import deepmr

    First, generate a 2D phantom and add some noise:

    >>> img = deepmr.shepp_logan(128) + 0.05 * torch.randn(128, 128)

    Now, run wavelet decomposition:

    >>> coeff, shape = deepmr.fwt(img)

    The function returns a ``coeff`` tuple, containing the Wavelet coefficients,
    and a ``shape`` tuple, containing the original image shape for image synthesis via
    ``deepmr.iwt``:

    >>> shape
    torch.Size([128, 128])

    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # cast to tensor
    input = torch.as_tensor(input)

    # get device
    idevice = input.device
    if device is None:
        device = idevice
    input = input.to(device)

    # get default ndim
    if ndim is None:
        ndim = min(3, len(input.shape))

    # pad to nearest even value
    ishape = input.shape
    zshape = [((ishape[n] + 1) // 2) * 2 for n in range(-ndim, 0)]
    zinput = resize(
        input.reshape(-1, *ishape[-ndim:]), [int(np.prod(ishape[:-ndim]))] + zshape
    )

    # select wavelet
    wavelet = pywt.Wavelet(wave_name)

    # select transform
    if ndim == 1:
        _fwt = ptwt.wavedec
    elif ndim == 2:
        _fwt = ptwt.wavedec2
    elif ndim == 3:
        _fwt = ptwt.wavedec3
    else:
        raise ValueError(
            f"Number of dimensions (={ndim}) not recognized; we support only 1, 2 and 3."
        )

    # compute
    output = _fwt(zinput, wavelet, mode="zero", level=level)
    output = list(output)
    output[0] = output[0].to(idevice)
    for n in range(1, len(output)):
        output[n] = [o.to(idevice) for o in output[n]]

    # cast to numpy if required
    if isnumpy:
        output[0] = output.numpy(force=True)
        for n in range(1, len(output)):
            output[n] = [o.numpy(force=True) for o in output[n]]

    return output, ishape


def iwt(input, shape, device=None, wave_name="db4", level=None):
    """
    Inverse wavelet transform.

    Adapted from Sigpy [1].

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input wavelet decomposition.
    shape : Iterable[int], optional
        Spatial matrix size of output signal ``(nx)`` (1D signals),
        ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
    device : str, optional
        Computational device for Wavelet transform.
        If not specified, use ``input.device``.
        The default is ``None``.
    wave_name : str, optional
        Wavelet name. The default is ``"db4"``.
    axes : Iterable[int], optional
        Axes to perform wavelet transform.
        The default is ``None`` (all axes).
    level : int, optional
        Number of wavelet levels. The default is ``None``.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Output signal of shape (..., nz, ny, nx).

    Examples
    --------
    >>> import torch
    >>> import deepmr

    First, generate a 2D phantom and add some noise:

    >>> img0 = deepmr.shepp_logan(128) + 0.05 * torch.randn(128, 128)

    Now, run wavelet decomposition:

    >>> coeff, shape = deepmr.fwt(img0)

    The image can be synthesized from ``coeff`` and ``shape`` as:

    >>> img = deepmr.iwt(coeff, shape)

    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # cast to tensor
    output = list(input)
    output[0] = torch.as_tensor(output[0])
    for n in range(1, len(output)):
        output[n] = [torch.as_tensor(o) for o in output[n]]

    # get device
    idevice = output[0].device
    if device is None:
        device = idevice

    # transfer to device
    output[0] = output[0].to(idevice)
    for n in range(1, len(output)):
        output[n] = [o.to(idevice) for o in output[n]]

    # convert to tuple
    for n in range(1, len(output)):
        output[n] = tuple(output[n])
    output = tuple(output)

    # select wavelet
    wavelet = pywt.Wavelet(wave_name)

    # select transform
    ndim = len(shape)
    if ndim == 1:
        _iwt = ptwt.waverec
    elif ndim == 2:
        _iwt = ptwt.waverec2
    elif ndim == 3:
        _iwt = ptwt.waverec3
    else:
        raise ValueError(
            f"Number of dimensions (={ndim}) not recognized; we support only 1, 2 and 3."
        )

    # compute
    zoutput = _iwt(output, wavelet)
    zoutput = zoutput.reshape(*shape[:-ndim], *zoutput.shape[-ndim:])
    output = resize(zoutput, shape)
    output = output.to(idevice)

    # cast to numpy if required
    if isnumpy:
        output = output.numpy(force=True)

    # erase singleton dimension
    if len(output.shape) == ndim + 1 and output.shape[0] == 1:
        output = output[0]

    return output
