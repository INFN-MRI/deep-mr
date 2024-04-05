"""Regularized Linear Least squares optimization."""

__all__ = []

import warnings
import time

import numpy as np
import torch

from ... import linops as _linops
from ... import optim as _optim
from ... import prox as _prox

from . import precond as _precond

from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def linear_lstsq(
    input,
    A,
    AHA=None,
    niter=30,
    prior=None,
    lamda=0.0,
    stepsize=1.0,
    use_precond=False,
    precond_degree=4,
    tol=1e-4,
    AHA_niter=5,
    power_niter=10,
    AHA_offset=None,
    AHy_offset=None,
    device=None,
    save_history=False,
    verbose=0,
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
    use_precond : bool, optional
        Use polynomial preconditioning to accelerate convergence.
        The default is ``False``.
    precond_degree : int, optional
        Degree of polynomial preconditioner. Ignored if ``use_precond`` is ``False``
        and for CG / ADMM solvers (no / multi-regularizations), where the degree
        is given by ``niter-1`` and ``AHA_niter-1``, respectively.
        The default is ``4``.
    Other Parameters
    ----------------
    tol : float, optional
        Stopping condition for CG (either as standalone solver or as
        inner data consistency term for ADMM) or PGD.
        The default is ``1e-4``.
    AHA_niter : int, optional
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
    AHy_offset : np.ndarray | torch.Tensor, optional
        Offset to be applied to AHy during computation.
        The default is ``None``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    save_history : bool, optional
        Record cost function. The default is ``False``.
    verbose : int, optional
        Display information (> 1 more, = 1 less, 0 = quiet). The default is ``0``.

    Returns
    -------
    img np.ndarray | torch.Tensor
        Reconstructed signal.

    """
    if verbose:
        tstart = time.time()

    # cast to numpy if required
    if isinstance(input, np.ndarray):
        isnumpy = True
        input = torch.as_tensor(input)
    else:
        isnumpy = False

    # default
    if AHA is None:
        if verbose > 1:
            print("Normal Operator not provided; using AHA = A.H * A")
        AHA = A.H * A

    # parse number of dimension
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
    if verbose > 1:
        print("Computing initial solution AHy = A.H(y)...", end="\t")
        t0 = time.time()
    AHy = A(input.clone())
    if verbose > 1:
        t1 = time.time()
        print(f"done! Elapsed time: {round(t1-t0, 2)} s")

    if verbose > 1:
        print(f"Data shape is {AHy.shape}; Spatial dimension: {AHy.shape[-ndim:]}")

    # if non-iterative, just perform adjoint recon
    if niter == 1:
        output = AHy
        if isnumpy:
            output = output.numpy(force=True)
        if verbose:
            tend = time.time()
            print(
                f"Max # iteration = 1; exiting - total elapsed time: {round(tstart-tend, 2)} s"
            )
        return output, None

    # apply offset to AHy
    if AHy_offset is not None and lamda != 0.0:
        if verbose > 1:
            print("Applying offset to AHy for L2 regularization")
        AHy_offset = torch.as_tensor(AHy_offset, dtype=AHy.dtype, device=AHy.device)
        AHy = AHy + lamda * AHy_offset

    # rescale for easier handling of Lipschitz constant
    AHy, scale = _intensity_scaling(AHy, ndim=ndim)
    if verbose > 1:
        print(f"Rescaling data by the 95% percentile of magnitude = {scale}")

    # if no prior is specified, use CG recon
    if prior is None:
        if use_precond:
            if verbose > 1:
                print("Prior not specified - solving using Polynomial inversion")
            # computing polynomial preconditioner
            P = _precond.create_polynomial_preconditioner("l_inf", niter, AHA)
            # solving
            output, history = P * (AHA(AHy) - (AHy + lamda * AHy)), None
        else:
            if verbose > 1:
                print("Prior not specified - solving using Conjugate Gradient")
            output, history = _optim.cg_solve(
                AHy,
                AHA,
                niter=niter,
                lamda=lamda,
                ndim=ndim,
                tol=tol,
                save_history=save_history,
                verbose=verbose,
            )
        if isnumpy:
            output = output.numpy(force=True)
        if verbose:
            tend = time.time()
            print(f"Exiting - total elapsed time: {round(tstart-tend, 2)} s")
        return output, history

    # compute spectral norm
    xhat = torch.rand(AHy.shape, dtype=AHy.dtype, device=AHy.device)

    # if a single prior is specified, use PDG
    if isinstance(prior, (list, tuple)) is False:
        if verbose > 1:
            print("Single prior - solving using FISTA")
        # modify AHA
        if lamda != 0.0:
            if AHA_offset is None:
                AHA_offset = _linops.Identity(ndim)
            _AHA = AHA + lamda * AHA_offset
        else:
            _AHA = AHA
        # compute norm
        if verbose > 1:
            print("Computing maximum eigenvalue of AHA...", end="\t")
            t0 = time.time()
        max_eig = _optim.power_method(
            None, xhat, AHA=_AHA, device=device, niter=power_niter
        )
        if verbose > 1:
            t1 = time.time()
            print(f"done! Elapsed time: {round(t1-t0, 2)} s")
            print(f"Maximum eigenvalue: {max_eig}")
        if max_eig != 0.0:
            stepsize = stepsize / max_eig
        if verbose > 1:
            print(f"FISTA stepsize: {stepsize}")
        # set prior threshold
        prior.ths = stepsize
        # computing polynomial preconditioner
        if use_precond:
            P = _precond.create_polynomial_preconditioner("l_2", precond_degree, AHA)
        else:
            P = None
        # solve
        output, history = _optim.pgd_solve(
            AHy,
            stepsize,
            _AHA,
            prior,
            niter=niter,
            accelerate=True,
            tol=tol,
            save_history=save_history,
            verbose=verbose,
        )
    else:
        if verbose > 1:
            print("Multiple priors - solving using ADMM")
        # compute norm
        if verbose > 1:
            print("Computing maximum eigenvalue of AHA...", end="\t")
            t0 = time.time()
        max_eig = _optim.power_method(
            None, xhat, AHA=AHA, device=device, niter=power_niter
        )
        if verbose > 1:
            t1 = time.time()
            print(f"done! Elapsed time: {round(t1-t0, 2)} s")
            print(f"Maximum eigenvalue: {max_eig}")
        if max_eig != 0.0:
            lamda = 10.0 * lamda * max_eig
        if verbose > 1:
            print(f"ADMM regularization strength: {lamda}")
        # modify AHA
        if lamda != 0.0:
            if AHA_offset is None:
                AHA_offset = _linops.Identity(ndim)
            _AHA = AHA + lamda * AHA_offset
        else:
            _AHA = AHA
        # set prior threshold
        for p in prior:
            p.ths = lamda / stepsize
        # computing polynomial preconditioner
        if use_precond:
            P = _precond.create_polynomial_preconditioner("l_inf", AHA_niter - 1, AHA)
        else:
            P = None
        # solve
        output, history = _optim.admm_solve(
            AHy,
            stepsize,
            _AHA,
            prior,
            P,
            niter=niter,
            dc_nite=AHA_niter,
            dc_tol=tol,
            dc_ndim=ndim,
            save_history=save_history,
            verbose=verbose,
        )

    # output
    if isnumpy:
        output = output.numpy(force=True)

    if verbose:
        tend = time.time()
        print(f"Exiting - total elapsed time: {round(tstart-tend, 2)} s")

    return output, history


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

    return input / scale, scale
