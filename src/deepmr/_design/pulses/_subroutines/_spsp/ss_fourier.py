"""Fourier transform utils."""

__all__ = ["fftf", "fftr"]

import numpy as np


def fftf(x, N=None, dim=None):
    xsize = x.shape

    if len(xsize) > 2:
        raise ValueError("Only handles 2D matrices")

    if N is None:
        mnval, mndim = np.min(xsize), np.argmin(xsize)
        if mnval > 1:
            dim = 0
            N = xsize[dim]
        else:
            dim = 1 - mndim  # Get opposite of mndim
            N = xsize[dim]

    if dim is None:
        mnval, mndim = np.min(xsize), np.argmin(xsize)
        if mnval > 1:
            dim = 0
            if N < xsize[dim]:
                raise ValueError("N less than number of rows of x")
        else:
            dim = 1 - mndim  # Get opposite of mndim
            if N < xsize[dim]:
                raise ValueError("N less than number of elements of x")

    fsize = list(xsize)
    fsize[dim] = N
    pad = np.zeros(fsize, dtype=x.dtype)

    Nd2 = int(np.ceil((N + 1) / 2))
    nxd2 = int(np.ceil((xsize[dim] + 1) / 2))
    sidx = Nd2 - nxd2
    if dim == 0:
        pad[sidx : sidx + xsize[dim], :] = x
    else:
        pad[:, sidx : sidx + xsize[dim]] = x

    shifted_pad = np.fft.ifftshift(pad, dim)
    xf = np.fft.fftshift(np.fft.fft(shifted_pad, N, dim), dim)

    return xf


def fftr(xf, N=None, dim=None):
    xfsize = xf.shape

    if len(xfsize) > 2:
        raise ValueError("Only handles 2D matrices")

    if N is None:
        mnval, mndim = np.min(xfsize), np.argmin(xfsize)
        if mnval > 1:
            dim = 0
            N = xfsize[dim]
        else:
            dim = 1 - mndim  # Get opposite of mndim
            N = xfsize[dim]

    if dim is None:
        mnval, mndim = np.min(xfsize), np.argmin(xfsize)
        if mnval > 1:
            dim = 0
            if N > xfsize[dim]:
                raise ValueError("N greater than the number of rows of xf")
        else:
            dim = 1 - mndim  # Get opposite of mndim
            if N > xfsize[dim]:
                raise ValueError("N greater than the number of elements of xf")

    # Get transform along dim
    xpad = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(xf, dim), None, dim), dim)

    # Fill in pad array
    Nd2 = int(np.ceil((N + 1) / 2))
    nxfd2 = int(np.ceil((xfsize[dim] + 1) / 2))
    sidx = nxfd2 - Nd2
    if dim == 0:
        x = xpad[sidx : sidx + N, :]
    else:
        x = xpad[:, sidx : sidx + N]

    return x
