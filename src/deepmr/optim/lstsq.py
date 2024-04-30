# """Regularized Linear Least squares optimization."""

# __all__ = ["lstsq"]

# import warnings
# import time

# import numpy as np
# import torch

# from .. import linops as _linops
# from .. import optim as _optim

# from .. import precond as _precond

# from numba.core.errors import NumbaPerformanceWarning

# warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


# def lstsq(
#     input,
#     AH,
#     AHA=None,
#     niter=30,
#     prior=None,
#     lamda=0.0,
#     stepsize=1.0,
#     use_precond=False,
#     precond_degree=4,
#     tol=1e-4,
#     AHA_niter=5,
#     power_niter=10,
#     AHA_offset=None,
#     AHy_offset=None,
#     ndim=None,
#     device=None,
#     save_history=False,
#     verbose=0,
# ):
#     """
#     Linear Least Squares solver for sparse linear operators.

#     Parameters
#     ----------
#     input : np.ndarray | torch.Tensor
#         Signal to be reconstructed.
#     AH : deepmr.linop.Linop, optional
#         Adjoint of forward operator AH = A.H.
#     AHA : deepmr.linop.Linop, optional
#         Normal operator AHA = AH * A. If not provided, compute from A.
#         The default is ``None``.
#     niter : int, optional
#         Number of iterations. If single iteration,
#         perform simple adjoin operation. The default is ``30``.
#     prior : nn.Module | Iterable[nn.Module], optional
#         Signal denoiser for regularization. If no prior is specifiec,
#         solve using Conjugate Gradient. If single prior is specified,
#         use Proximal Gradient Method. If multiple priors are specified,
#         use ADMM.
#         The default is ``None`` (no regularizer, i.e., CG).
#     lamda : float, optional
#         Regularization strength. If 0.0, do not apply regularization.
#         The default is ``0.0``.
#     stepsize : float, optional
#         Iterations step size. If not provided, estimate from Normal
#         operator maximum eigenvalue. The default is ``None``.
#     use_precond : bool, optional
#         Use polynomial preconditioning to accelerate convergence. Ignored
#         for CG / ADMM solvers (no / multi-regularizations).
#         The default is ``False``.
#     precond_degree : int, optional
#         Degree of polynomial preconditioner. Ignored if ``use_precond`` is ``False``
#         and for CG / ADMM solvers (no / multi-regularizations).
#         The default is ``4``.

#     Other Parameters
#     ----------------
#     tol : float, optional
#         Stopping condition for CG (either as standalone solver or as
#         inner data consistency term for ADMM) or PGD.
#         The default is ``1e-4``.
#     AHA_niter : int, optional
#         Number of iterations of CG data consistency step for ADMM. Final
#         number of steps will be ``niter``, i.e., number of outer loops in ADMM
#         is ``niter // dc_niter``. Ignored for CG (no regularizers) and PGD (single regularizer).
#         The default is ``5``.
#     power_niter : int, optional
#         Number of iterations of power method for Lipschitz constant estimation.
#         Ignored for CG (no regularizers).
#         The default is ``10``.
#     AHA_offset : deepmr.linop.Linop, optional
#         Offset to be applied to AHA during computation for regularization.
#         If no prior is specified, this is ignored (for CG it is ``Identity``).
#         The default is ``Identity``.
#     AHy_offset : np.ndarray | torch.Tensor, optional
#         Offset to be applied to AHy during computation.
#         The default is ``None``.
#     device : str, optional
#         Computational device. The default is ``None`` (same as ``data``).
#     save_history : bool, optional
#         Record cost function. The default is ``False``.
#     verbose : int, optional
#         Display information (> 1 more, = 1 less, 0 = quiet). The default is ``0``.

#     Returns
#     -------
#     output : np.ndarray | torch.Tensor
#         Reconstructed signal.

#     """
#     if verbose > 0:
#         tstart = time.time()

#     # cast to numpy if required
#     if isinstance(input, np.ndarray):
#         isnumpy = True
#         input = torch.as_tensor(input)
#     else:
#         isnumpy = False

#     # default
#     if AHA is None:
#         if verbose > 1:
#             print("Normal Operator not provided; using AHA = A.H * A")
#         AHA = AH * AH.H

#     # keep original device
#     idevice = input.device
#     if device is None:
#         device = idevice

#     # put on device
#     input = input.to(device)
#     if isinstance(AH, _linops.Linop):
#         AH = AH.to(device)
#     if isinstance(AHA, _linops.Linop):
#         AHA = AHA.to(device)

#     # assume input is AH(y), i.e., adjoint of measurement operator
#     # applied on measured data
#     AHy = _adjoint_recon(AH, input, verbose)

#     # if non-iterative, just perform adjoint recon
#     if niter == 1:
#         output = AHy
#         if isnumpy:
#             output = output.numpy(force=True)
#         if verbose > 0:
#             tend = time.time()
#             print(
#                 f"Max # iteration = 1; exiting - total elapsed time: {round(tstart-tend, 2)} s"
#             )
#         return output, None

#     # rescale for easier handling of Lipschitz constant
#     # AHy, scale = _intensity_scaling(AHy, ndim=ndim)
#     # if verbose > 1:
#     #     print(f"Rescaling data by the 95% percentile of magnitude = {scale}")

#     # if no prior is specified, use CG recon
#     if prior is None:
#         output, history = _CG_recon(
#             AHy, AHA, AHy_offset, AHA_offset, lamda, niter, tol, save_history, verbose
#         )

#         if isnumpy:
#             output = output.numpy(force=True)
#         if verbose > 0:
#             tend = time.time()
#             print(f"Exiting - total elapsed time: {round(tstart-tend, 2)} s")
#         return output, history

#     # if a single prior is specified, use PDG
#     if isinstance(prior, (list, tuple)) is False:
#         output, history = _FISTA_recon(
#             AHy,
#             AHA,
#             AHy_offset,
#             AHA_offset,
#             lamda,
#             niter,
#             tol,
#             save_history,
#             verbose,
#             prior,
#             stepsize,
#             power_niter,
#             use_precond,
#             precond_degree,
#         )

#     # if multiple regularizers are specified, use ADMM
#     else:
#         output, history = _ADMM_recon(
#             AHy,
#             AHA,
#             AHy_offset,
#             AHA_offset,
#             lamda,
#             prior,
#             stepsize,
#             power_niter,
#             niter,
#             tol,
#             save_history,
#             verbose,
#             AHA_niter,
#         )

#     # output
#     if isnumpy:
#         output = output.numpy(force=True)
#     if verbose:
#         tend = time.time()
#         print(f"Exiting - total elapsed time: {round(tstart-tend, 2)} s")

#     return output, history


# # %% local utils
# # def _intensity_scaling(input, ndim):
# #     data = input.clone()
# #     for n in range(len(input.shape) - ndim):
# #         data = torch.linalg.norm(data, axis=0)

# #     # get scaling
# #     data = torch.nan_to_num(data, posinf=0.0, neginf=0.0, nan=0.0)
# #     scale = torch.quantile(abs(data.ravel()), 0.95)

# #     return input / scale, scale


# def _adjoint_recon(AH, input, verbose):
#     if verbose > 1:
#         print("Computing initial solution AHy = A.H(y)...", end="\t")
#         t0 = time.time()
#     AHy = AH(input.clone())
#     if verbose > 1:
#         t1 = time.time()
#         print(f"done! Elapsed time: {round(t1-t0, 2)} s")

#     if verbose > 1:
#         print(f"Data shape is {AHy.shape}")

#     return AHy


# def _CG_recon(
#     AHy, AHA, AHy_offset, AHA_offset, lamda, niter, tol, save_history, verbose
# ):
#     if verbose > 1:
#         print("Prior not specified - solving using Conjugate Gradient")

#     # apply offset to AHy
#     if AHy_offset is not None and lamda != 0.0:
#         if verbose > 1:
#             print("Applying offset to AHy for L2 regularization")
#         AHy_offset = torch.as_tensor(AHy_offset, dtype=AHy.dtype, device=AHy.device)
#         AHy = AHy + lamda * AHy_offset

#     # modify AHA
#     if lamda != 0.0:
#         if AHA_offset is not None:
#             _AHA = AHA + lamda * AHA_offset
#         else:
#             _AHA = AHA
#     else:
#         _AHA = AHA

#     # solve
#     output, history = _optim.cg_solve(
#         AHy,
#         _AHA,
#         niter=niter,
#         lamda=lamda,
#         tol=tol,
#         save_history=save_history,
#         verbose=verbose,
#     )

#     return output, history


# def _FISTA_recon(
#     AHy,
#     AHA,
#     AHy_offset,
#     AHA_offset,
#     lamda,
#     niter,
#     tol,
#     save_history,
#     verbose,
#     prior,
#     stepsize,
#     power_niter,
#     use_precond,
#     precond_degree,
# ):
#     if verbose > 1:
#         print("Single prior - solving using FISTA")

#     # apply offset to AHy
#     if AHy_offset is not None and lamda != 0.0:
#         if verbose > 1:
#             print("Applying offset to AHy for L2 regularization")
#         AHy_offset = torch.as_tensor(AHy_offset, dtype=AHy.dtype, device=AHy.device)
#         AHy = AHy + lamda * AHy_offset

#     # modify AHA
#     if lamda != 0.0:
#         if AHA_offset is None:
#             AHA_offset = _linops.Identity()
#         if isinstance(AHA, _linops.Linop):
#             _AHA = AHA + lamda * AHA_offset
#         else:

#             def _AHA(x):
#                 return AHA(x) + lamda * AHA_offset(x)

#     else:
#         _AHA = AHA

#     # compute norm
#     if verbose > 1:
#         print("Computing maximum eigenvalue of AHA...", end="\t")
#         t0 = time.time()
#     xhat = torch.rand(AHy.shape, dtype=AHy.dtype, device=AHy.device)
#     max_eig = _optim.power_method(None, xhat, AHA=_AHA, niter=power_niter)
#     if verbose > 1:
#         t1 = time.time()
#         print(f"done! Elapsed time: {round(t1-t0, 2)} s")
#         print(f"Maximum eigenvalue: {max_eig}")

#     # rescale stepsize
#     if max_eig != 0.0:
#         stepsize = stepsize / max_eig
#     if verbose > 1:
#         print(f"FISTA stepsize: {stepsize}")

#     # set prior threshold
#     prior.ths = lamda * stepsize

#     # computing polynomial preconditioner
#     if use_precond:
#         P = _precond.create_polynomial_preconditioner("l_2", precond_degree, AHA)
#     else:
#         P = None

#     # solve
#     output, history = _optim.pgd_solve(
#         AHy,
#         stepsize,
#         _AHA,
#         prior,
#         P=P,
#         niter=niter,
#         accelerate=True,
#         tol=tol,
#         save_history=save_history,
#         verbose=verbose,
#     )

#     return output, history


# def _ADMM_recon(
#     AHy,
#     AHA,
#     AHy_offset,
#     AHA_offset,
#     lamda,
#     prior,
#     stepsize,
#     power_niter,
#     niter,
#     tol,
#     save_history,
#     verbose,
#     AHA_niter,
# ):
#     if verbose > 1:
#         print("Multiple priors - solving using ADMM")

#     # apply offset to AHy
#     if AHy_offset is not None and lamda != 0.0:
#         if verbose > 1:
#             print("Applying offset to AHy for L2 regularization")
#         AHy_offset = torch.as_tensor(AHy_offset, dtype=AHy.dtype, device=AHy.device)
#         AHy = AHy + lamda * AHy_offset

#     # modify AHA
#     if lamda != 0.0:
#         if AHA_offset is None:
#             AHA_offset = _linops.Identity()
#         if isinstance(AHA, _linops.Linop):
#             _AHA = AHA + lamda * AHA_offset
#         else:

#             def _AHA(x):
#                 return AHA(x) + lamda * AHA_offset(x)

#     else:
#         _AHA = AHA

#     # compute norm
#     if verbose > 1:
#         print("Computing maximum eigenvalue of AHA...", end="\t")
#         t0 = time.time()
#     xhat = torch.rand(AHy.shape, dtype=AHy.dtype, device=AHy.device)
#     max_eig = _optim.power_method(None, xhat, AHA=AHA, niter=power_niter)
#     if verbose > 1:
#         t1 = time.time()
#         print(f"done! Elapsed time: {round(t1-t0, 2)} s")
#         print(f"Maximum eigenvalue: {max_eig}")

#     # rescale step size
#     if max_eig != 0.0:
#         lamda = 0.1 * lamda * max_eig
#     else:
#         lamda = 0.1 * lamda

#     if verbose > 1:
#         print(f"ADMM regularization strength: {lamda}")

#     # set prior threshold
#     for p in prior:
#         p.ths = lamda / stepsize

#     # solve
#     output, history = _optim.admm_solve(
#         AHy,
#         stepsize,
#         _AHA,
#         prior,
#         niter=niter,
#         dc_niter=AHA_niter,
#         dc_tol=tol,
#         save_history=save_history,
#         verbose=verbose,
#     )

#     return output, history
