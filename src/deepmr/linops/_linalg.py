"""Linear algorithms for Linops."""

__all__ = ["power_method", "cg_solve", "polynomial_inversion"]

import torch

from .. import _precond


@torch.no_grad()
def power_method(AHA, x, niter=10, device=None):
    r"""
    Use power iteration to calculate the maximum eigenvalue of a Linop.

    From MIRTorch (https://github.com/guanhuaw/MIRTorch/blob/master/mirtorch/alg/spectral.py)

    Parameters
    ----------
    AHA : deepmr.linop.Linop
        Normal operator AHA = AH * A.
    x : torch.Tensor
        Initial guess of singular vector corresponding to max singular value.
    niter : int, optional
        Maximum number of iterations. The default is ``10``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).

    Returns
    -------
    float
        The maximum eigenvalue of the operator ``A``.

    """
    # keep original device
    idevice = x.device
    if device is None:
        device = idevice

    # put on device
    x = x.clone()
    x = x.to(device)
    AHA = AHA.to(device)

    # perform iterations
    max_eig = float("inf")

    for n in range(niter):
        # update eigenvector
        AHAx = AHA(x)
        max_eig = torch.linalg.norm(AHAx)
        x = AHAx / max_eig

    return max_eig.item() ** 0.5


@torch.no_grad()
def cg_solve(AHA, AHy, niter=10, tol=None, device=None):
    """
    Solve inverse problem using Conjugate Gradient method.

    Parameters
    ----------
    AHA : deepmr.linop.Linop
        Normal operator AHA = AH * A.
    AHy : torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y.
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
    # keep original device
    idevice = AHy.device
    if device is None:
        device = idevice

    # put on device
    AHy = AHy.clone()
    AHy = AHy.to(device)
    AHA = AHA.to(device)

    # initialize algorithm
    CG = CGStep(AHA, AHy, tol)

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


@torch.no_grad()
def polynomial_inversion(AHA, AHy, lamda, degree=2, device=None):
    """
    Solve inverse problem using Polynomial Inversion method.

    Parameters
    ----------
    AHA : deepmr.linop.Linop
        Normal operator AHA = AH * A.
    AHy : torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y.
    lamda : float
        Tikhonov regularization strength.
    degree : int, optional
        Degree of the polynomial. The default is ``2``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).

    Returns
    -------
    output : torch.Tensor
        Reconstructed signal.

    """
    # keep original device
    idevice = AHy.device
    if device is None:
        device = idevice

    # put on device
    AHy = AHy.clone()
    AHy = AHy.to(device)
    AHA = AHA.to(device)

    # create preconditioner
    P = _precond.create_polynomial_preconditioner(
        "l_inf", degree - 1, AHA, l=1, L=1 + lamda
    )

    # perform inversion
    output = -P(-AHy)

    # back to original device
    output = output.to(device)

    return output


# %% local utils
class CGStep(torch.nn.Module):
    """
    Conjugate Gradient method step.

    This represents propagation through a single iteration of a
    CG algorithm; can be used to build
    unrolled architectures.

    Attributes
    ----------
    AHA : Callable | torch.Tensor
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    tol : float, optional
        Stopping condition.
        The default is ``None`` (run until niter).

    """

    def __init__(self, AHA, AHy, tol=None):
        super().__init__()

        # assign operators
        self.AHA = AHA
        self.AHy = AHy

        # preallocate
        self.r = self.AHy.clone()
        self.p = self.r
        self.rsold = self.dot(self.r, self.r)
        self.rsnew = None
        self.tol = tol

    def dot(self, s1, s2):  # noqa
        dot = s1.conj() * s2
        dot = dot.sum()

        return dot

    def forward(self, input):  # noqa
        AHAp = self.AHA(self.p)
        alpha = self.rsold / self.dot(self.p, AHAp)
        output = input + self.p * alpha
        self.r = self.r + AHAp * (-alpha)
        self.rsnew = torch.real(self.dot(self.r, self.r))
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
