"""Utils for calibration region extraction."""

__all__ = ["find_acs"]

import torch
import numpy as np

from ... import fft as _fft


def find_acs(data, shape=None, coord=None, dcf=None):
    """
    Extract k-space calibration region.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input k-space data.
    shape : int, optional
        Size of gridded k-space for Non-Cartesian imaging.
        The default is ``None``.
    coord : np.ndarray | torch.Tensor, optional
        K-space trajectory of ``shape = (ncontrasts, nviews, nsamples, ndim)``.
        The default is ``None`` (Cartesian acquisition).
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation of ``shape = (ncontrasts, nviews, nsamples)``.
        The default is ``None`` (no compensation).

    Returns
    -------
    calib : np.ndarray | torch.Tensor
        Output calibration k-space data of shape ``(ncoils, nz, ny, nx).

    Notes
    -----
    The input k-space ``data`` tensor is assumed to have the following shape:

    * **2Dcart:** ``(nslices, ncoils, ncontrasts, ny, nx)``.
    * **2Dnoncart:** ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    * **3Dcart:** ``(nx, ncoils, ncontrasts, nz, ny)``.
    * **3Dnoncart:** ``(ncoils, ncontrasts, nviews, nsamples)``.

    For multi-contrast acquisitions, calibration is obtained by averaging over
    contrast dimensions.

    """
    if coord is None:
        return _find_cart_acs(data)
    else:
        return _find_noncart_acs(data, shape, coord, dcf)


# %% local utils
def _find_cart_acs(kspace):
    """Adapted from https://github.com/mckib2/pygrappa/blob/main/pygrappa/find_acs.py"""

    # hardcoded
    coil_axis = 1

    # sum over contrasts
    kspace = kspace.sum(axis=-3)  # (nslices, ncoils, ny, nx)

    # convert to numpy
    if isinstance(kspace, torch.Tensor):
        istorch = True
        device = kspace.device
        kspace = kspace.numpy(force=True)
    else:
        istorch = False

    # cast to single precision
    kspace = kspace.astype(np.complex64)

    # move coil axis to the last dim
    kspace = np.moveaxis(kspace, coil_axis, -1)
    mask = np.abs(kspace[..., 0]) > 0

    # Start by finding the largest hypercube
    ctrs = [d // 2 for d in mask.shape]  # assume ACS is at center
    slices = [[c, c + 1] for c in ctrs]  # start with 1 voxel region
    while all(
        l > 0 and r < mask.shape[ii] for ii, (l, r) in enumerate(slices)
    ) and np.all(mask[tuple([slice(l - 1, r + 1) for l, r in slices])]):
        # expand isotropically until we can't no more
        slices = [[l0 - 1, r0 + 1] for l0, r0 in slices]

    # Stretch left/right in each dimension
    for dim in range(mask.ndim):
        # left: only check left condition on the current dimension
        while slices[dim][0] > 0 and np.all(
            mask[tuple([slice(l - (dim == k), r) for k, (l, r) in enumerate(slices)])]
        ):
            slices[dim][0] -= 1
        # right: only check right condition on the current dimension
        while slices[dim][1] < mask.shape[dim] and np.all(
            mask[tuple([slice(l, r + (dim == k)) for k, (l, r) in enumerate(slices)])]
        ):
            slices[dim][1] += 1

    # move back coil axis to original dim
    kspace = np.moveaxis(
        kspace[tuple([slice(l0, r0) for l0, r0 in slices] + [slice(None)])].copy(),
        -1,
        coil_axis,
    )

    # cast back to torch if required
    if istorch:
        kspace = torch.as_tensor(kspace, device=device, dtype=torch.complex64)

    kspace = kspace.swapaxes(0, 1)  # (ncoils, nz, ny, nx)
    kspace = _fft.fft(kspace, axes=(1,))

    return kspace


def _find_noncart_acs(kspace, cal_shape, coord, dcf):
    # get ndim
    ndim = coord.shape[-1]

    # collapse contrast dimension to view
    kspace = kspace.reshape(
        *kspace.shape[:2], 1, -1, kspace.shape[-1]
    )  # (nslices, ncoils, 1, ncontrasts * nviews, nsamples)
    coord = coord.reshape(
        1, -1, *coord.shape[-2:]
    )  # (1, ncontrasts * nviews, nsamples, ndim)
    if dcf is not None:
        dcf = dcf.reshape(1, -1, dcf.shape[-1])

    # find subset of coord along nsamples dim
    cabs = (coord**2).sum(axis=-1) ** 0.5  # sqrt(kx**2 + ky**2 + kz**2)
    ind = (
        cabs[0, 0] <= min(cal_shape) // 2
    )  # from (-mtx /2, mtx / 2) to (-cal / 2, cal / 2)

    # select kspace and coordinates
    cal_ksp = kspace[..., ind]
    cal_coord = coord[..., ind, :]

    if dcf is not None:
        cal_dcf = dcf[..., ind]
        cal_ksp = cal_dcf * cal_ksp

    # grid calibration
    cal_ksp = _fft.nufft_adj(
        cal_ksp, cal_coord, cal_shape, oversamp=2.0
    ).squeeze()  # (nslices, ncoils, ny, nx) or (ncoils, nz, ny, nx)
    if len(cal_ksp.shape) != 4:
        cal_ksp = cal_ksp[None, ...]
    if ndim == 2:
        cal_ksp = cal_ksp.swapaxes(0, 1)
    cal_ksp = _fft.fft(cal_ksp, axes=range(-3, 0))

    return torch.as_tensor(cal_ksp, dtype=torch.complex64)
