"""Pytorch ESPIRIT implementation. Adapted for convenience from https://github.com/mikgroup/espirit-python/tree/master"""

__all__ = ["espirit_cal"]

import numpy as np
import torch

from ... import fft as _fft
from ... import _signal

from . import acs as _acs


def espirit_cal(
    data, coord=None, dcf=None, shape=None, k=6, r=24, t=0.02, c=0.0, nsets=1
):
    """
    Derives the ESPIRiT [1] operator.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Multi channel k-space data.
    coord : np.ndarray | torch.Tensor, optional
        K-space trajectory of ``shape = (ncontrasts, nviews, nsamples, ndim)``.
        The default is ``None`` (Cartesian acquisition).
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation of ``shape = (ncontrasts, nviews, nsamples)``.
        The default is ``None`` (no compensation).
    shape : Iterable[int] | optional
        Shape of the k-space after gridding. If not provided, estimate from
        input data (assumed on a Cartesian grid already).
        The default is ``None`` (Cartesian acquisition).
    k : int, optional
        k-space kernel size. The default is ``6``.
    r : int, optional
        Calibration region size. The default is ``24``.
    t : float, optional
        Rank of the auto-calibration matrix (A).
        The default is ``0.02``.
    c : float, optional
        Crop threshold that determines eigenvalues "=1".
        The defaults is ``0.95``.
    nsets : int, optional
        Number of set of maps to be returned.
        The default is ``1`` (conventional SENSE recon).

    Returns
    -------
    maps : np.ndarray | torch.Tensor
        Output coil sensitivity maps.

    Notes
    -----
    The input k-space ``data`` tensor is assumed to have the following shape:

    * **2Dcart:** ``(nslices, ncoils, ncontrasts, ny, nx)``.
    * **2Dnoncart:** ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    * **3Dcart:** ``(nx, ncoils, ncontrasts, nz, ny)``.
    * **3Dnoncart:** ``(ncoils, ncontrasts, nviews, nsamples)``.

    For multi-contrast acquisitions, calibration is obtained by averaging over
    contrast dimensions.

    The output sensitivity maps are assumed to have the following shape:

    * **2Dcart:** ``(nslices, nsets, ncoils, ny, nx)``.
    * **2Dnoncart:** ``(nslices, nsets, ncoils, ny, nx)``.
    * **3Dcart:** ``(nx, nsets, ncoils, nz, ny)``.
    * **3Dnoncart:** ``(nsets, ncoils, nz, ny, nx)``.

    References
    ----------
    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M.
           ESPIRiT--an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA.
           Magn Reson Med. 2014 Mar;71(3):990-1001. doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.

    """
    if isinstance(data, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    while len(data.shape) < 5:
        data = data[None, ...]

    # keep shape
    if coord is not None:
        ndim = coord.shape[-1]
        if np.isscalar(shape):
            shape = ndim * [shape]
        else:
            shape = list(shape)[-ndim:]
        shape = [int(s) for s in shape]
    else:
        ndim = 2
        shape = list(data.shape[-2:])

    # extract calibration region
    cshape = list(np.asarray(shape, dtype=int) // 2)
    cal_data = _acs.find_acs(data, cshape, coord, dcf)

    # calculate maps
    maps = _espirit(cal_data.clone(), k, r, t, c)

    # select maps
    if nsets == 1:
        maps = maps[[0]]
    else:
        maps = maps[:nsets]

    # resample
    maps = _signal.resample(maps, shape)  # (nsets, ncoils, nz, ny, nx)

    # normalize
    maps_rss = _signal.rss(maps, axis=1, keepdim=True)
    maps = maps / maps_rss[[0]]

    # reformat
    if ndim == 2:  # Cartesian or 2D Non-Cartesian
        maps = maps.swapaxes(1, 2)  # (nsets, nslices, ncoils, ny, nx)
        maps = maps.swapaxes(0, 1)  # (nslices, nsets, ncoils, ny, nx)

    # cast back to numpy if required
    if isnumpy:
        maps = maps.numpy(force=True)

    return maps


# %% local utils
def _espirit(X, k, r, t, c):
    # transpose
    X = X.permute(3, 2, 1, 0)

    # get shape
    sx, sy, sz, nc = X.shape

    sxt = (sx // 2 - r // 2, sx // 2 + r // 2) if (sx > 1) else (0, 1)
    syt = (sy // 2 - r // 2, sy // 2 + r // 2) if (sy > 1) else (0, 1)
    szt = (sz // 2 - r // 2, sz // 2 + r // 2) if (sz > 1) else (0, 1)

    # Extract calibration region.
    C = X[sxt[0] : sxt[1], syt[0] : syt[1], szt[0] : szt[1], :].to(
        dtype=torch.complex64
    )

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = torch.zeros(
        [(r - k + 1) ** p, k**p * nc], dtype=torch.complex64, device=X.device
    )

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
        for ydx in range(max(1, C.shape[1] - k + 1)):
            for zdx in range(max(1, C.shape[2] - k + 1)):
                block = C[xdx : xdx + k, ydx : ydx + k, zdx : zdx + k, :].to(
                    dtype=torch.complex64
                )
                A[idx, :] = block.flatten()
                idx += 1

    # Take the Singular Value Decomposition.
    U, S, VH = torch.linalg.svd(A, full_matrices=True)
    V = VH.conj().t()

    # Select kernels
    n = torch.sum(S >= t * S[0])
    V = V[:, :n]

    kxt = (sx // 2 - k // 2, sx // 2 + k // 2) if (sx > 1) else (0, 1)
    kyt = (sy // 2 - k // 2, sy // 2 + k // 2) if (sy > 1) else (0, 1)
    kzt = (sz // 2 - k // 2, sz // 2 + k // 2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = torch.zeros((sx, sy, sz, nc, n), dtype=torch.complex64, device=X.device)
    kerdims = [
        ((sx > 1) * k + (sx == 1) * 1),
        ((sy > 1) * k + (sy == 1) * 1),
        ((sz > 1) * k + (sz == 1) * 1),
        nc,
    ]
    for idx in range(n):
        kernels[kxt[0] : kxt[1], kyt[0] : kyt[1], kzt[0] : kzt[1], :, idx] = V[
            :, idx
        ].reshape(kerdims)

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = (
        _fft.fft(kernels.flip(0).flip(1).flip(2).conj(), axes)
        * (sx * sy * sz) ** 0.5
        / (k**p) ** 0.5
    )

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    u, s, vh = torch.linalg.svd(
        kerimgs.view(sx, sy, sz, nc, n).reshape(-1, nc, n), full_matrices=True
    )
    mask = s**2 > c

    # mask u (nvoxels, neigen, neigen)
    u = mask[:, None, :] * u

    # Reshape back to the original shape and assign to maps
    maps = u.view(sx, sy, sz, nc, nc)

    # transpose
    maps = maps.permute(4, 3, 2, 1, 0)

    return maps
