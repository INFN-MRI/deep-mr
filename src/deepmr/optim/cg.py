"""Conjugate Gradient iteration."""

__all__ = ["cg_solve", "CGStep"]

import numpy as np
import torch

import torch.nn as nn

from .. import linops as _linops

@torch.no_grad()
def cg_solve(
    input,
    AHA,
    niter=10,
    device=None,
    tol=1e-4,
    lamda=0.0,
    ndim=None,
):
    """
    Solve inverse problem using Conjugate Gradient method.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y (i.e., input = AHy).
    AHA : Callable
        Normal operator AHA = AH * A.
    niter : int, optional
        Number of iterations. The default is ``10``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).
    tol : float, optional
        Stopping condition. The default is ``1e-4``.
    lamda : float, optional
        Tikhonov regularization strength. The default is ``0.0``.
    ndim : int, optional
        Number of spatial dimensions of the problem.
        It is used to infer the batch axes. If ``AHA`` is a ``deepmr.linop.Linop``
        operator, this is inferred from ``AHA.ndim`` and ``ndim`` is ignored.


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

    # keep original device
    idevice = input.device
    if device is None:
        device = idevice

    # put on device
    input = input.to(device)
    if isinstance(AHA, _linops.Linop):
        AHA = AHA.to(device)

    # assume input is AH(y), i.e., adjoint of measurement operator
    # applied on measured data
    AHy = input.clone()

    # add Tikhonov regularization
    if lamda != 0.0:
        if isinstance(AHA, _linops.Linop):
            _AHA = AHA + lamda * _linops.Identity(AHA.ndim)
        else:
            _AHA = lambda x: AHA(x) + lamda * x
    else:
        _AHA = AHA

    # initialize algorithm
    CG = CGStep(_AHA, AHy, ndim)

    # initialize
    input = 0 * input

    # run algorithm
    for n in range(niter):
        output = CG(input)
        if CG.rsnew.sqrt() < tol:
            break
        input = output.clone()

    # back to original device
    output = output.to(device)

    # cast back to numpy if requried
    if isnumpy:
        output = output.numpy(force=True)

    return output


class CGStep(nn.Module):
    """
    Conjugate Gradient method step.

    This represents propagation through a single iteration of a
    CG algorithm; can be used to build
    unrolled architectures.

    Attributes
    ----------
    AHA : Callable
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    ndim : int
        Number of spatial dimensions of the problem.
        It is used to infer the batch axes. If ``AHA`` is a ``deepmr.linop.Linop``
        operator, this is inferred from ``AHA.ndim`` and ``ndim`` is ignored.

    """

    def __init__(self, AHA, AHy, ndim=None):
        super().__init__()
        # set up problem dims
        try:
            self.ndim = AHA.ndim
        except Exception:
            self.ndim = ndim

        # assign operators
        self.AHA = AHA
        self.AHy = AHy

        # preallocate
        self.r = self.AHy.clone()
        self.p = self.r
        self.rsold = self.dot(self.r, self.r)
        self.rsnew = None

    def dot(self, s1, s2):
        dot = s1.conj() * s2
        dot = dot.reshape(*s1.shape[: -self.ndim], -1).sum(axis=-1)

        return dot

    def forward(self, input):
        AHAp = self.AHA(self.p)
        alpha = self.rsold / self.dot(self.p, AHAp)
        output = input + self.p * alpha
        self.r = self.r + AHAp * (-alpha)
        self.rsnew = torch.real(self.dot(self.r, self.r))
        self.p = self.r + self.p * (self.rsnew / self.rsold)
        self.rsold = self.rsnew

        return output
