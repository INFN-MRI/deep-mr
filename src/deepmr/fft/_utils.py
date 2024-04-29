"""Utils for B0 informed reconstruction."""

__all__ = []

import math

import numpy as np
import torch


def mri_exp_approx(zmap, t, lseg=6, bins=(40, 40), mask=None, toeplitz=False):
    r"""
    From Sigpy: https://github.com/mikgroup/sigpy and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/
    Creates B [L, nt] and Ct [L, (nz), ny, nx] matrices to approximate exp(-2i*pi*b0*t) [nt, (nz), ny, nx]

    Parameters
    ----------
    zmap : torch.Tensor
        Rate map defined as ``zmap = R2*_map + 1j * B0_map``.
        ``*_map`` and ``t`` should have reciprocal units.
        If ``zmap`` is real, assume ``zmap = B0_map``.
        Expected shape is ``(nz, ny, nx)``.
    t : torch.Tensor
        Readout time in ``[s]`` of shape ``(npts,)``.
    lseg : int, optional
        Number of time segments. The default is ``6``.
    bins : int || tuple(int), optional
        Number of histogram bins to use for ``(B0, T2)``. The default is ``(40, 40)``
        If it is a scalar, assume ``bins = (bins, 40)``.
    mask : torch.Tensor, optional
        Boolean mask to avoid histogram of background values.
        The default is ``None`` (use the whole map).
    toeplitz : bool, optional
        If ``True``, compute histogram autocorrelation. The default is ``False.``

    Returns
    -------
    b : torch.Tensor
        Temporal interpolator of shape ``(lseg, npts)``.
    ct : torch.Tensor
        Off-resonance phase map at each time segment center of shape
        ``(lseg, *zmap.shape)``.

    TODO(guahuaw@umich.edu): The SVD approach and pure pytorch implementation.

    """
    # default
    if isinstance(bins, (list, tuple)) is False:
        bins = (bins, 5)

    # set acorr
    acorr = toeplitz

    # transform to list
    bins = list(bins)

    # get field map
    if torch.isreal(zmap).all().item():
        r2star = None
        b0 = zmap
        zmap = 0.0 + 1j * b0
    else:
        r2star = zmap.real
        b0 = zmap.imag

    # default mask
    if mask is None:
        mask = torch.ones_like(zmap, dtype=bool)

    # Hz to radians / s
    zmap = 2 * math.pi * zmap

    # create histograms
    if r2star is not None:
        z = zmap[mask].ravel()
        z = torch.stack((z.imag, z.real), axis=1)
        hk, ze = torch.histogramdd(z, bins=bins)
        ze = list(ze)

        # get bin centers
        zc = [e[1:] - e[1] / 2 for e in ze]

        # autocorr of histogram, for Toeplitz
        if acorr:
            hk = _corr2d(hk, hk)
            zc[0] = torch.arange(-(bins[0] - 1), bins[0]) * (zc[0][1] - zc[0][0])
            zc[1] = torch.linspace(2 * zc[1].min(), 2 * zc[1].max(), 2 * bins[1] - 1)

        zk = _outer_sum(1j * zc[0], zc[1])  # [K1 K2]
    else:
        z = zmap[mask].ravel()
        hk, ze = torch.histogram(z, bins=bins[0])

        # get bin centers
        zc = ze[1:] - ze[1] / 2

        # complexify
        zk = 0 + 1j * zc  # [K 1]

        # autocorr of histogram, for Toeplitz
        if acorr:
            hk = _corr1d(hk, hk)
            zk = torch.arange(-(bins[0] - 1), bins[0]) * zk[1] - zk[0]

    # flatten histogram values and centers
    hk = hk.flatten()
    zk = zk.flatten()

    # generate time for each segment
    tl = torch.linspace(0, lseg, lseg) / lseg * t[-1]  # time seg centers in [s]

    # complexify histogram and time
    hk = hk.to(dtype=zk.dtype, device=zk.device)
    tl = tl.to(dtype=zk.dtype, device=zk.device)
    t = tl.to(dtype=zk.dtype, device=zk.device)

    # prepare for basis calculation
    ch = torch.exp(-tl.unsqueeze(1) @ zk.unsqueeze(0))
    w = torch.diag(hk**0.5)
    p = torch.linalg.pinv(w @ ch.t()) @ w

    # actual temporal basis calculation
    b = p @ torch.exp(zk.unsqueeze(1) * t.unsqueeze(0))

    # get spatial coeffs
    ct = torch.exp(-tl * zmap[..., None])
    ct = ct[None, ...].swapaxes(0, -1)[..., 0]  # (..., lseg) -> (lseg, ...)

    # clean-up of spatial coeffs
    ct = torch.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)

    return b, ct


# %% utils
def _corr1d(a, b):
    a1 = a.unsqueeze(0).unsqueeze(0)
    b1 = b.unsqueeze(0).unsqueeze(0)
    padsize = b1.shape[-1] - 1
    return torch.nn.functional.conv1d(a1, b1, padding=padsize)[0][0]


def _corr2d(a, b):
    a1 = a.unsqueeze(0).unsqueeze(0)
    b1 = b.unsqueeze(0).unsqueeze(0)
    padsize = (b1.shape[-2] - 1, b1.shape[-1] - 1)
    return torch.nn.functional.conv2d(a1, b1, padding=padsize)[0][0]


def _outer_sum(xx, yy):
    xx = xx.unsqueeze(1)  # Add a singleton dimension at axis 1
    yy = yy.unsqueeze(0)  # Add a singleton dimension at axis 0
    ss = xx + yy  # Compute the outer sum
    return ss
