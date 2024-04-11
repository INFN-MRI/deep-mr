"""Alternate Direction of Multipliers Method iteration."""

__all__ = ["admm_solve", "ADMMStep"]

import time

import numpy as np
import torch

import torch.nn as nn

from .cg import cg_solve

from .. import linops as _linops


@torch.no_grad()
def admm_solve(
    input,
    step,
    AHA,
    D,
    niter=10,
    device=None,
    dc_niter=10,
    dc_tol=1e-4,
    dc_ndim=None,
    tol=None,
    save_history=False,
    verbose=False,
):
    """
    Solve inverse problem using Alternate Direction of Multipliers Method.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y (i.e., input = AHy).
    step : float
        Gradient step size; should be <= 1 / max(eig(AHA)).
    AHA : Callable | torch.Tensor | np.ndarray
        Normal operator AHA = AH * A.
    D : Callable
        Signal denoiser for plug-n-play restoration.
    niter : int, optional
        Number of iterations. The default is ``10``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).
    dc_niter : int, optional
        Number of iterations of inner data consistency step.
        The default is ``10``.
    dc_tol : float, optional
        Stopping condition for inner data consistency step.
        The default is ``1e-4``.
    dc_ndim : int, optional
        Number of spatial dimensions of the problem for inner data consistency step.
        It is used to infer the batch axes. If ``AHA`` is a ``deepmr.linop.Linop``
        operator, this is inferred from ``AHA.ndim`` and ``ndim`` is ignored.
    atol : float, optional
        Stopping condition for ADMM. If not provided, run until ``niter``.
        The default is ``None``.
    save_history : bool, optional
        Record cost function. The default is ``False``.
    verbose : bool, optional
        Display information. The default is ``False``.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Reconstructed signal.

    """
    # cast to numpy if required
    if isinstance(input, np.ndarray):
        isnumpy = True
        input = torch.as_tensor(input)
    else:
        isnumpy = False
        
    # assert inputs are correct
    if verbose:
        assert save_history is True, "We need to record history to print information."

    # keep original device
    idevice = input.device
    if device is None:
        device = idevice

    # put on device
    input = input.to(device)
    if isinstance(AHA, _linops.Linop):
        AHA = AHA.to(device)
    elif callable(AHA) is False:
        AHA = torch.as_tensor(AHA, dtype=input.dtype, device=device)

    # assume input is AH(y), i.e., adjoint of measurement operator
    # applied on measured data
    AHy = input.clone()

    # initialize algorithm
    ADMM = ADMMStep(step, AHA, AHy, D, niter=dc_niter, tol=dc_tol, atol=tol, ndim=dc_ndim)

    # initialize
    input = 0 * input
    history = []
    
    # start timer
    if verbose:
        t0 = time.time()
        nprint = np.linspace(0, niter, 5)
        nprint = nprint.astype(int).tolist()
        print("================================= ADMM =====================================")
        print("| nsteps | n_AHA | data consistency | regularization | total cost | t-t0 [s]")
        print("============================================================================")

    # run algorithm
    for n in range(niter):
        output = ADMM(input)
        
        # update variable
        input = output.clone()
        
        # if required, compute residual and check if we reached convergence
        if ADMM.check_convergence(output, input, step):
            break
        
        # if required, save history
        if save_history:
            r = output - AHy
            dc = 0.5 * torch.linalg.norm(r).item() ** 2
            reg = np.sum([d.g(output) for d in D])
            history.append(dc+reg)
            if verbose and n in nprint:
                t = time.time()
                print(" {}{}{:.4f}{:.4f}{:.4f}{:.2f}".format(n, n * dc_niter, dc, reg, dc+reg, t-t0))
                
    if verbose:
        t1 = time.time()
        print(f"Exiting ADMM: total elapsed time: {round(t1-t0, 2)} [s]")

    # back to original device
    output = output.to(device)

    # cast back to numpy if requried
    if isnumpy:
        output = output.numpy(force=True)

    return output, history


class ADMMStep(nn.Module):
    """
    Alternate Direction of Multipliers Method step.

    This represents propagation through a single iteration of a
    ADMM algorithm; can be used to build
    unrolled architectures.

    Attributes
    ----------
    step : float
        ADMM step size; should be <= 1 / max(eig(AHA)).
    AHA : Callable | torch.Tensor
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    D : Iterable(Callable)
        Signal denoiser(s) for plug-n-play restoration.
    P : Callable, optional
        Polynomial preconditioner for data consistency.
        The default is ``None`` (standard CG for data consistency).
    trainable : bool, optional
        If ``True``, gradient update step is trainable, otherwise it is not.
        The default is ``False``.
    niter : int, optional
        Number of iterations of inner data consistency step. The default is ``10``.
    tol : float, optional
        Stopping condition for inner data consistency step. The default is ``1e-4``
    atol : float, optional
        Stopping condition for ADMM. If not provided, run until ``niter``.
        The default is ``None``.
    ndim : int, optional
        Number of spatial dimensions of the problem for inner data consistency step.
        It is used to infer the batch axes. If ``AHA`` is a ``deepmr.linop.Linop``
        operator, this is inferred from ``AHA.ndim`` and ``ndim`` is ignored.

    """

    def __init__(
        self, step, AHA, AHy, D, trainable=False, niter=10, tol=1e-4, atol=None, ndim=None
    ):
        super().__init__()
        if trainable:
            self.step = nn.Parameter(step)
        else:
            self.step = step

        # set up problem dims
        try:
            self.ndim = AHA.ndim
        except Exception:
            self.ndim = ndim

        # assign operators
        self.AHA = AHA
        self.AHy = AHy

        # assign denoisers
        if hasattr(D, "__iter__"):
            self.D = list(D)
        else:
            self.D = [D]

        # prepare auxiliary
        self.xi = torch.zeros(
            [1 + len(self.D)] + list(AHy.shape),
            dtype=AHy.dtype,
            device=AHy.device,
        )
        self.ui = torch.zeros_like(self.xi)

        # dc solver settings
        self.niter = niter
        self.tol = tol
        self.atol = atol

    def forward(self, input):
        # data consistency step: zk = (AHA + gamma * I).solve(AHy)
        self.xi[0], _ = cg_solve(
            self.AHy + self.step * (input - self.ui[0]),
            self.AHA,
            niter=self.niter,
            tol=self.tol,
            lamda=self.step,
            ndim=self.ndim,
        )

        # denoise using each regularizator
        for n in range(len(self.D)):
            self.xi[n + 1] = self.D[n](input - self.ui[n + 1])

        # average consensus
        output = self.xi.mean(axis=0)
        self.ui += self.xi - output[None, ...]

        return output
    
    def check_convergence(self, output, input, step):
        if self.atol is not None:
            resid = torch.linalg.norm(output - input).item() / step
            if resid < self.atol:
                return True
            else:
                return False
        else:
            return False
