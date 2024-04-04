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

def linear_lstsq(input, A, AHA=None, niter=1, prior=None, prior_ths=0.01, prior_params=None, solver_params=None, lamda=0.0, stepsize=1.0, max_eig=None, device=None):
    """
    Classical MR reconstruction.

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
        Number of recon iterations. If single iteration,
        perform simple zero-filled recon. The default is ``1``.
    prior : str | Iterable[str] | nn.Module | Iterable[nn.Module], optional
        Prior for image regularization. If string, it must be one of the following:

        * ``"L1Wav"``: L1 Wavelet regularization.
        * ``"TGV"``: Total Variation regularization.
        * ``"TV"``: Total Variation regularization.

        The default is ``None`` (no regularizer).
    prior_ths : float, optional
        Threshold for denoising in regularizer. The default is ``0.01``.
    prior_params : dict, optional
        Parameters for Prior initializations.
        See :func:`deepmr.prox`.
        The defaul it ``None`` (use each regularizer default parameters).
    solver_params : dict, optional
        Parameters for Solver initializations.
        See :func:`deepmr.optim`.
        The defaul it ``None`` (use each solver default parameters).
    lamda : float, optional
        Regularization strength. If 0.0, do not apply regularization.
        The default is ``0.0``.
    stepsize : float, optional
        Iterations step size. If not provided, estimate from Encoding
        operator maximum eigenvalue. The default is ``None``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).

    Returns
    -------
    img np.ndarray | torch.Tensor
        Reconstructed image of shape:

        * 2D Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 2D Non Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 2D Non Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 3D Non Cartesian: ``(ncontrasts, nz, ny, nx).

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

    # if non-iterative, just perform linear recon
    if niter == 1:
        output = AHy
        if isnumpy:
            output = output.numpy(force=True)
        return output

    # default solver params
    if solver_params is None:
        solver_params = {}

    # rescale for easier handling of Lipschitz constant
    AHy = _intensity_scaling(AHy, ndim=ndim)

    # if no prior is specified, use CG recon
    if prior is None:
        output = _optim.cg_solve(
            AHy, AHA, niter=niter, lamda=lamda, ndim=ndim, **solver_params
        )
        if isnumpy:
            output = output.numpy(force=True)
        return output

    # modify EHE
    if lamda != 0.0:
        _AHA = AHA + lamda * _linops.Identity(ndim)
    else:
        _AHA = AHA

    # compute spectral norm
    xhat = torch.rand(AHy.shape, dtype=AHy.dtype, device=AHy.device)
    if max_eig is None:
        max_eig = _optim.power_method(None, xhat, AHA=_AHA, device=device, niter=30)
    if max_eig != 0.0:
        stepsize = stepsize / max_eig

    # if a single prior is specified, use PDG
    if isinstance(prior, (list, tuple)) is False:
        # default prior params
        if prior_params is None:
            prior_params = {}

        # get prior
        D = _get_prior(prior, ndim, lamda, device, **prior_params)

        # solve
        output = _optim.pgd_solve(
            AHy, stepsize, _AHA, D, niter=niter, accelerate=True, **solver_params
        )
    else:
        npriors = len(prior)
        if prior_params is None:
            prior_params = [{} for n in range(npriors)]
        else:
            assert (
                isinstance(prior_params, (list, tuple)) and len(prior_params) == npriors
            ), "Please provide parameters for each regularizer (or leave completely empty to use default)"

        # get priors
        D = []
        for n in range(npriors):
            d = _get_prior(prior[n], ndim, lamda, device, **prior_params[n])
            D.append(d)

        # solve
        output = _optim.admm_solve(AHy, stepsize, _AHA, D, niter=niter, **solver_params)
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


