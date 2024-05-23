"""Conjugate Gradient algorithm."""

__all__ = ["conjgrad"]

import torch

def conjgrad(AHA, AHy, x0=None, niter=30, tol=None, device=None):
    """
    Solve inverse problem using Conjugate Gradient method.

    Parameters
    ----------
    AHA : deepmr.linop.Linop
        Normal operator AHA = AH * A.
    AHy : torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y.
    x0 : torch.Tensor, optional
        Initial guess for solution. The default is ``None``, (i.e., 0.0).
    niter : int, optional
        Number of iterations. The default is ``10``.
    tol : float, optional
        Stopping condition. The default is ``1e-4``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).

    Returns
    -------
    output : torch.Tensor
        Reconstructed signal.

    """
    return CG.apply(AHy, AHA, x0, niter, tol, device)

# %% local utils
class CG(torch.autograd.Function):
    @staticmethod
    def forward(AHy, AHA, x0=None, niter=10, tol=None, device=None):
        return _cg_solve(AHA, AHy, x0, niter, tol, device)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        AHy, AHA, x0, niter, tol, device = inputs
        ctx.save_for_backward(AHy)
        ctx.AHA = AHA
        ctx.niter = niter
        ctx.tol = tol
        ctx.device = device

    @staticmethod
    def backward(ctx, dx):
        AHy = ctx.saved_tensors[0]
        AHA = ctx.AHA
        niter = ctx.niter 
        tol = ctx.tol 
        device = ctx.device
        return (
            _cg_solve(AHA, dx, AHy, niter, tol, device),
            None,
            None,
            None,
            None,
            None,
        )

def _cg_solve(AHA, AHy, x0=None, niter=10, tol=None, device=None):
    # keep original device
    idevice = AHy.device
    if device is None:
        device = idevice

    # put on device
    AHy = AHy.clone()
    if x0 is not None:
        x0 = x0.to(device)
    AHy = AHy.to(device)
    AHA = AHA.to(device)

    # initialize algorithm
    CG = CGStep(AHA, AHy, x0, tol)

    # initialize
    input = 0 * AHy

    # run algorithm
    for n in range(niter):
        output = CG(input)

        # if required, compute residual and check if we reached convergence
        if CG.check_convergence():
            break

        # update variable
        input = output.clone()

    # back to original device
    output = output.to(device)

    return output

class CGStep:
    """
    Conjugate Gradient method step.

    This represents propagation through a single iteration of a
    CG algorithm.

    Attributes
    ----------
    AHA : Linop
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    x : torch.Tensor
        Initial guess for solution. The default is ``None``, (i.e., 0.0).
    tol : float, optional
        Stopping condition.
        The default is ``None`` (run until niter).

    """

    def __init__(self, AHA, AHy, x0=None, tol=None):

        # assign operators
        self.AHA = AHA
        self.AHy = AHy

        # preallocate
        self.r = self.AHy.clone()
        if x0 is not None:
            self.r -= self.AHA(x0)
                        
        self.p = self.r.clone()
        self.rsold = self.dot(self.r, self.r)
        self.rsnew = None
        self.tol = tol

    def dot(self, s1, s2):  # noqa
        dot = s1.conj() * s2
        dot = dot.sum()

        return dot.real

    def __call__(self, input):  # noqa
        AHAp = self.AHA(self.p)
        alpha = self.rsold / self.dot(self.p, AHAp)
        output = input + self.p * alpha
        self.r = self.r - alpha * AHAp
        self.rsnew = self.dot(self.r, self.r)
        self.p = self.r + self.p * (self.rsnew / self.rsold)
        self.rsold = self.rsnew

        return output

    def check_convergence(self):  # noqa
        if self.tol is not None:
            if self.rsnew.sqrt() < self.tol:
                return True
            else:
                return False
        else:
            return False
