"""Regularized Linear Least squares optimization."""

__all__ = []

import numpy as np
import torch

from ... import linops as _linops
from ... import optim as _optim
from ... import prox as _prox

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def linear_lstsq(
    input,
    A,
    AHA=None,
    niter=30,
    prior=None,
    lamda=0.0,
    stepsize=1.0,
    tol=1e-4,
    dc_niter=5,
    power_niter=10,
    AHA_offset=None,
    dc_offset=None,
    device=None,
    save_history=False,
    verbose=False,
):
    """
    Linear Least Squares solver for sparse linear operators.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Signal to be reconstructed.
    AHA : deepmr.linop.Linop, optional
        Forward operator input = A(solution).
    AHA : deepmr.linop.Linop, optional
        Normal operator AHA = AH * A. If not provided, compute from A.
        The default is ``None``.
    niter : int, optional
        Number of iterations. If single iteration,
        perform simple adjoin operation. The default is ``30``.
    prior : nn.Module | Iterable[nn.Module], optional
        Signal denoiser for regularization. If no prior is specifiec,
        solve using Conjugate Gradient. If single prior is specified,
        use Proximal Gradient Method. If multiple priors are specified,
        use ADMM.
        The default is ``None`` (no regularizer, i.e., CG).
    lamda : float, optional
        Regularization strength. If 0.0, do not apply regularization.
        The default is ``0.0``.
    stepsize : float, optional
        Iterations step size. If not provided, estimate from Normal
        operator maximum eigenvalue. The default is ``None``.

    Other Parameters
    ----------------
    tol : float, optional
        Stopping condition for CG (either as standalone solver or as
        inner data consistency term for ADMM) or PGD.
        The default is ``1e-4``.
    dc_niter : int, optional
        Number of iterations of CG data consistency step for ADMM. Final
        number of steps will be ``niter``, i.e., number of outer loops in ADMM
        is ``niter // dc_niter``. Ignored for CG (no regularizers) and PGD (single regularizer).
        The default is ``5``.
    power_niter : int, optional
        Number of iterations of power method for Lipschitz constant estimation.
        Ignored for CG (no regularizers).
        The default is ``10``.
    AHA_offset : deepmr.linop.Linop, optional
        Offset to be applied to AHA during computation for regularization.
        If no prior is specified, this is ignored (for CG it is ``Identity``).
        The default is ``Identity``.
    dc_offset : np.ndarray | torch.Tensor, optional
        Offset to be applied to AHy during computation.
        The default is ``None``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    save_history : bool, optional
        Record cost function. The default is ``False``.
    verbose : bool, optional
        Display information. The default is ``False``.

    Returns
    -------
    img np.ndarray | torch.Tensor
        Reconstructed signal.

    """
    # cast to numpy if required
    if isinstance(input, np.ndarray):
        isnumpy = True
        input = torch.as_tensor(input)
    else:
        isnumpy = False

    # default
    if AHA is None:
        AHA = A.H * A

    # parse number of dimensions
    ndim = A.ndim

    # keep original device
    idevice = input.device
    if device is None:
        device = idevice

    # put on device
    input = input.to(device)
    A = A.to(device)
    AHA = AHA.to(device)

    # assume input is AH(y), i.e., adjoint of measurement operator
    # applied on measured data
    AHy = A(input.clone())

    # apply offset to AHy
    if dc_offset is not None and lamda != 0.0:
        dc_offset = torch.as_tensor(dc_offset, dtype=AHy.dtype, device=AHy.device)
        AHy = AHy + lamda * dc_offset

    # if non-iterative, just perform linear recon
    if niter == 1:
        output = AHy
        if isnumpy:
            output = output.numpy(force=True)
        return output

    # rescale for easier handling of Lipschitz constant
    AHy = _intensity_scaling(AHy, ndim=ndim)

    # if no prior is specified, use CG recon
    if prior is None:
        output = _optim.cg_solve(
            AHy,
            AHA,
            niter=niter,
            lamda=lamda,
            ndim=ndim,
            tol=tol,
        )
        if isnumpy:
            output = output.numpy(force=True)
        return output

    # modify EHE
    if lamda != 0.0:
        if AHA_offset is None:
            AHA_offset = _linops.Identity(ndim)
        _AHA = AHA + lamda * AHA_offset
    else:
        _AHA = AHA

    # compute spectral norm
    xhat = torch.rand(AHy.shape, dtype=AHy.dtype, device=AHy.device)
    max_eig = _optim.power_method(
        None, xhat, AHA=_AHA, device=device, niter=power_niter
    )
    if max_eig != 0.0:
        stepsize = stepsize / max_eig

    # if a single prior is specified, use PDG
    if isinstance(prior, (list, tuple)) is False:
        # solve
        output = _optim.pgd_solve(
            AHy, stepsize, _AHA, prior, niter=niter, accelerate=True
        )
    else:
        # solve
        output = _optim.admm_solve(AHy, stepsize, _AHA, prior, niter=niter)
    if isnumpy:
        output = output.numpy(force=True)

    return output


# %% local utils
def _get_prior(ptype, ndim, lamda, device, **params):
    if isinstance(ptype, str):
        if ptype == "L1Wave":
            return _prox.WaveletDenoiser(ndim, ths=lamda, device=device, **params)
        elif ptype == "TV":
            return _prox.TVDenoiser(ndim, ths=lamda, device=device, **params)
        elif ptype == "TGV":
            return _prox.TGVDenoiser(ndim, ths=lamda, device=device, **params)
        elif ptype == "LLR":
            return _prox.LLRDenoiser(ndim, ths=lamda, device=device, **params)
        else:
            raise ValueError(
                f"Prior type = {ptype} not recognized; either specify 'L1Wave', 'TV', 'TGV' or 'LLR', or 'nn.Module' object."
            )
    else:
        raise NotImplementedError("Direct prior object not implemented.")


def _intensity_scaling(input, ndim):
    data = input.clone()
    for n in range(len(input.shape) - ndim):
        data = torch.linalg.norm(data, axis=0)

    # get scaling
    data = torch.nan_to_num(data, posinf=0.0, neginf=0.0, nan=0.0)
    scale = torch.quantile(abs(data.ravel()), 0.95)

    return input / scale
