"""Proximal Gradient Method iteration."""

__all__ = ["pgd_solve", "PGDStep"]

import time

import numpy as np
import torch

import torch.nn as nn

from .. import linops as _linops


@torch.no_grad()
def pgd_solve(
    input,
    step,
    AHA,
    D,
    P=None,
    niter=10,
    accelerate=True,
    device=None,
    tol=None,
    save_history=False,
    verbose=False,
):
    """
    Solve inverse problem using Proximal Gradient Method.

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
    P : Callable, optional
        Polynomial preconditioner.
        The default is ``None``.
    niter : int, optional
        Number of iterations. The default is ``10``.
    accelerate : bool, optional
        Toggle Nesterov acceleration (``True``, i.e., FISTA) or
        not (``False``, ISTA). The default is ``True``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).
    tol : float, optional
        Stopping condition.
        The default is ``None`` (run until ``niter``).
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
        ndim = AHA.ndim
    elif callable(AHA) is False:
        AHA = torch.as_tensor(AHA, dtype=input.dtype, device=device)
        ndim = 2
    else:
        ndim = 2

    # default precondition
    if P is None:
        P = _linops.Identity(ndim)

    # assume input is AH(y), i.e., adjoint of measurement operator
    # applied on measured data
    AHy = input.clone()

    # initialize Nesterov acceleration
    if accelerate:
        q = _get_acceleration(niter)
    else:
        q = [0.0] * niter

    # initialize algorithm
    PGD = PGDStep(step, AHA, AHy, D, P)

    # initialize
    input = 0 * input
    history = []
    
    # start timer
    if verbose:
        t0 = time.time()
        nprint = np.linspace(0, niter, 5)
        nprint = nprint.astype(int).tolist()
        print("============================ FISTA =================================")
        print("| nsteps | data consistency | regularization | total cost | t-t0 [s]")
        print("====================================================================")

    # run algorithm
    for n in range(niter):
        output = PGD(input, q[n])

        # if required, compute residual and check if we reached convergence
        if PGD.check_convergence(output, input, step):
            break

        # update variable
        input = output.clone()
        
        # if required, save history
        if save_history:
            r = output - AHy
            dc = 0.5 * torch.linalg.norm(r).item() ** 2
            reg = D.g(output)
            history.append(dc+reg)
            if verbose and n in nprint:
                t = time.time()
                print(" {}{:.4f}{:.4f}{:.4f}{:.2f}".format(n, dc, reg, dc+reg, t-t0))
                
    if verbose:
        t1 = time.time()
        print(f"Exiting FISTA: total elapsed time: {round(t1-t0, 2)} [s]")
            
    # back to original device
    output = output.to(device)

    # cast back to numpy if requried
    if isnumpy:
        output = output.numpy(force=True)

    return output, history


class PGDStep(nn.Module):
    """
    Proximal Gradient Method step.

    This represents propagation through a single iteration of a
    Proximal Gradient Descent algorithm; can be used to build
    unrolled architectures.

    Attributes
    ----------
    step : float
        Gradient step size; should be <= 1 / max(eig(AHA)).
    AHA : Callable | torch.Tensor
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    D : Callable
        Signal denoiser for plug-n-play restoration.
    P : Callable, optional
        Polynomial preconditioner for data consistency.
        The default is ``None`` (standard CG for data consistency).
    trainable : bool, optional
        If ``True``, gradient update step is trainable, otherwise it is not.
        The default is ``False``.
    tol : float, optional
        Stopping condition.
        The default is ``None`` (run until niter).

    """

    def __init__(self, step, AHA, AHy, D, P, trainable=False, tol=None):
        super().__init__()
        if trainable:
            self.step = nn.Parameter(step)
        else:
            self.step = step

        # assign
        self.AHA = AHA
        self.AHy = AHy
        self.P = P
        self.D = D
        self.s = AHy.clone()
        self.tol = tol

    def forward(self, input, q=0.0):
        # gradient step : zk = xk-1 - gamma * AH(A(xk-1) - y != FISTA (accelerated)
        z = input - self.P(self.step * (self.AHA(input) - self.AHy))

        # denoise: sk = D(zk)
        s = self.D(z)

        # update: xk = sk + [(qk-1 - 1) / qk] * (sk - sk-1)
        if q != 0.0:
            output = s + q * (s - self.s)
            self.s = s.clone()
        else:
            output = s  # q1...qn = 1.0 != ISTA (non-accelerated)

        return output

    def check_convergence(self, output, input, step):
        if self.tol is not None:
            resid = torch.linalg.norm(output - input).item() / step
            if resid < self.tol:
                return True
            else:
                return False
        else:
            return False


# %% local utils
def _get_acceleration(niter):
    t = []
    t_new = 1

    for n in range(niter):
        t_old = t_new
        t_new = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
        t.append((t_old - 1) / t_new)

    return t
