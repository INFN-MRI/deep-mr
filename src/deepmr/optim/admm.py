"""Alternate Direction of Multipliers Method iteration."""

__all__ = ["admm_solve", "ADMMStep"]

import time

import numpy as np
import torch

import torch.nn as nn

from .. import linops as _linops

from . import precond
from .cg import cg_solve


@torch.no_grad()
def admm_solve(
    input,
    step,
    AHA,
    D,
    use_precond=False,
    niter=10,
    lamda=0.0,
    device=None,
    dc_niter=10,
    dc_tol=1e-4,
    tol=None,
    save_history=False,
    verbose=False,
):
    """
    Solve inverse problem using Alternate Direction of Multipliers Method.

    Parameters
    ----------
    input : torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y (i.e., input = AHy).
    step : float
        Gradient step size; should be <= 1 / max(eig(AHA)).
    AHA : deepmr.linops.Linop
        Normal operator AHA = AH * A.
    D : Iterable[torch.nn.Module]
        Signal denoiser for plug-n-play restoration.
    use_precond : bool, optional
        Flag for polynomial inversion in data consistency step.
        If ``True``, use polynomial inversion. If ``False``, use Conjugate Gradient
        update instead. The default is ``False``.
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
    AHA = AHA.to(device)

    # assume input is AH(y), i.e., adjoint of measurement operator
    # applied on measured data
    AHy = input.clone()

    # denoisers
    if hasattr(D, "__iter__"):
        D = list(D)
    else:
        D = [D]

    # initialize algorithm
    ADMM = ADMMStep(
        step,
        AHA,
        AHy,
        D,
        use_precond,
        lamda=lamda,
        niter=dc_niter,
        tol=dc_tol,
        atol=tol,
    )

    # initialize
    input = 0 * input
    history = []

    # start timer
    if verbose:
        t0 = time.time()
        nprint = np.linspace(0, niter, 5)
        nprint = nprint.astype(int).tolist()
        print(
            "================================= ADMM ====================================="
        )
        print(
            "| nsteps | n_AHA | data consistency | regularization | total cost | t-t0 [s]"
        )
        print(
            "============================================================================"
        )

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
            history.append(dc + reg)
            if verbose and n in nprint:
                t = time.time()
                print(
                    " {}{}{:.4f}{:.4f}{:.4f}{:.2f}".format(
                        n, n * dc_niter, dc, reg, dc + reg, t - t0
                    )
                )

    # rescale output
    output *= len(D) + 1

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
    AHA : deepmr.linops.Linop
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    D : Iterable[torch.nn.Module]
        Signal denoiser(s) for plug-n-play restoration.
    use_precond : bool, optional
        Flag for polynomial inversion in data consistency step.
        If ``True``, use polynomial inversion. If ``False``, use Conjugate Gradient
        update instead. The default is ``False``.
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

    """

    def __init__(
        self,
        step,
        AHA,
        AHy,
        D,
        use_precond,
        trainable=False,
        niter=10,
        lamda=0.0,
        tol=1e-4,
        atol=None,
        ndim=None,
    ):
        super().__init__()
        if trainable:
            self.step = nn.Parameter(step)
        else:
            self.step = step

        # add Tikhonov regularization
        _AHA = _linops.Identity() + (self.step + lamda) * AHA
        AHy = (self.step + lamda) * AHy

        # create preconditioner
        if use_precond:
            self.P = precond.create_polynomial_preconditioner(
                "l_inf", niter - 1, _AHA, l=1, L=1 + (self.step + lamda)
            )
        else:
            self.P = None
            self.niter = niter

        # assign operators
        self.AHA = _AHA
        self.AHy = AHy

        # assign denoisers
        self.D = D

        # prepare auxiliary
        self.zi = torch.zeros(
            [1 + len(self.D)] + list(AHy.shape),
            dtype=AHy.dtype,
            device=AHy.device,
        )
        self.ui = torch.zeros_like(self.zi)

        # dc solver settings
        self.niter = niter
        self.tol = tol
        self.atol = atol

    def forward(self, input):  # noqa
        # data consistency step
        if (
            self.P is None
        ):  # z = (rho+lambda * AHA + I).solve(rho+lambda * AHy + x - u) # CG inversion
            self.zi[0], _ = cg_solve(
                self.AHy + (input - self.ui[0]), self.AHA, niter=self.niter
            )  # , tol=self.tol)
        else:  # z = x - u - P((rho+lamda * AHA + I)(x - u) - (rho+lambda * AHy + x - u)) # Polynomial inversion
            self.zi[0] = (
                input
                - self.ui[0]
                - self.P(self.AHA(input - self.ui[0]) - (self.AHy + input - self.ui[0]))
            )

        # denoise using each regularizator
        for n in range(len(self.D)):
            self.zi[n + 1] = self.D[n](input - self.ui[n + 1])

        # average consensus
        output = self.zi.mean(axis=0)
        self.ui += self.zi - output[None, ...]

        return output

    def check_convergence(self, output, input, step):  # noqa
        if self.atol is not None:
            resid = torch.linalg.norm(output - input).item() / step
            if resid < self.atol:
                return True
            else:
                return False
        else:
            return False
