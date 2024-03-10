"""Python implementation of the GRAPPA operator formalism. Adapted for convenience from PyGRAPPA"""

__all__ = ["grappaop"]

from types import SimpleNamespace

import torch


def grappaop(
    calib: torch.Tensor, ndim: int = 2, lamda: float = 0.01
) -> SimpleNamespace:
    """
    GRAPPA operator for Cartesian calibration datasets.

    Parameters
    ----------
    calib : torch.Tensor
        Calibration region data of shape (nc, nz, ny, nx) or (nc, ny, nx).
        Usually a small portion from the center of kspace.
    ndim : int, optional
        Number of k-space dimensions. Defaults to 2.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to 0.01.

    Returns
    -------
    GrappaOp : SimpleNamespace
        GRAPPA operator.

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.
    """
    # as Tensor
    calib = torch.as_tensor(calib)

    # expand
    if len(calib.shape) == 3:  # single slice (nc, ny, nx)
        calib = calib[:, None, :, :].clone()

    # compute kernels
    if ndim == 2:
        gy, gx = _grappa_op_2d(calib, lamda)
    elif ndim == 3:
        gz, gy, gx = _grappa_op_3d(calib, lamda)

    # prepare output
    GrappaOp = SimpleNamespace()
    GrappaOp.Gx, GrappaOp.Gy = (gx, gy)

    if ndim == 3:
        GrappaOp.Gz = gz
    else:
        GrappaOp.Gz = None

    return GrappaOp


# %% subroutines
def _grappa_op_2d(calib, lamda):
    """Return a batch of 2D GROG operators (one for each z)."""
    # coil axis in the back
    calib = torch.moveaxis(calib, 0, -1)
    nz, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sy = torch.reshape(calib[:, :-1, :, :], (nz, -1, nc))
    Sx = torch.reshape(calib[:, :, :-1, :], (nz, -1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Ty = torch.reshape(calib[:, 1:, :, :], (nz, -1, nc))
    Tx = torch.reshape(calib[:, :, 1:, :], (nz, -1, nc))

    # train the operators:
    Syh = Sy.conj().permute(0, 2, 1)
    lamda0 = lamda * torch.linalg.norm(Syh, dim=(1, 2)) / Syh.shape[1]
    Gy = torch.linalg.solve(
        _bdot(Syh, Sy) + lamda0[:, None, None] * torch.eye(Syh.shape[1])[None, ...],
        _bdot(Syh, Ty),
    )

    Sxh = Sx.conj().permute(0, 2, 1)
    lamda0 = lamda * torch.linalg.norm(Sxh, dim=(1, 2)) / Sxh.shape[1]
    Gx = torch.linalg.solve(
        _bdot(Sxh, Sx) + lamda0[:, None, None] * torch.eye(Sxh.shape[1])[None, ...],
        _bdot(Sxh, Tx),
    )

    return Gy.clone(), Gx.clone()


def _grappa_op_3d(calib, lamda):
    """Return 3D GROG operator."""
    # coil axis in the back
    calib = torch.moveaxis(calib, 0, -1)
    _, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sz = torch.reshape(calib[:-1, :, :, :], (-1, nc))
    Sy = torch.reshape(calib[:, :-1, :, :], (-1, nc))
    Sx = torch.reshape(calib[:, :, :-1, :], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tz = torch.reshape(calib[1:, :, :, :], (-1, nc))
    Ty = torch.reshape(calib[:, 1:, :, :], (-1, nc))
    Tx = torch.reshape(calib[:, :, 1:, :], (-1, nc))

    # train the operators:
    Szh = Sz.conj().permute(1, 0)
    lamda0 = lamda * torch.linalg.norm(Szh) / Szh.shape[0]
    Gz = torch.linalg.solve(Szh @ Sz + lamda0 * torch.eye(Szh.shape[0]), Szh @ Tz)

    Syh = Sy.conj().permute(1, 0)
    lamda0 = lamda * torch.linalg.norm(Syh) / Syh.shape[0]
    Gy = torch.linalg.solve(Syh @ Sy + lamda0 * torch.eye(Syh.shape[0]), Syh @ Ty)

    Sxh = Sx.conj().permute(1, 0)
    lamda0 = lamda * torch.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = torch.linalg.solve(Sxh @ Sx + lamda0 * torch.eye(Sxh.shape[0]), Sxh @ Tx)

    return Gz.clone(), Gy.clone(), Gx.clone()


def _bdot(a, b):
    return torch.einsum("...ij,...jk->...ik", a, b)
