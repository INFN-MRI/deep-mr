"""Polynomial Inversion routine."""

__all__ = ["polynomial_inversion"]

import time
import torch

from .. import linops as _linops

from . import precond


@torch.no_grad()
def polynomial_inversion(input, AHA, niter=10, lamda=0.0, device=None, verbose=False):
    """
    Solve inverse problem using Polynomial Inversion method.

    Parameters
    ----------
    input : torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y (i.e., input = AHy).
    AHA : deepmr.linop.Linop
        Normal operator AHA = AH * A.
    niter : int, optional
        Number of iterations. The default is ``10``.
    lamda : float, optional
        Tikhonov regularization strength. The default is ``0.0``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).
    verbose : bool, optional
        Display information. The default is ``False``.

    Returns
    -------
    output : torch.Tensor
        Reconstructed signal.

    """
    if verbose:
        t0 = time.time()
        print("Performing Polynomial Inversion...", end="\t")

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

    # add Tikhonov regularization
    if lamda != 0.0:
        _AHA = _linops.Identity() + lamda * AHA
        AHy = lamda * AHy
    else:
        _AHA = AHA

    # create preconditioner
    P = precond.create_polynomial_preconditioner(
        "l_inf", niter - 1, _AHA, l=1, L=1 + lamda
    )

    # perform inversion
    output = -P(-AHy)

    # back to original device
    output = output.to(device)

    if verbose:
        t1 = time.time()
        print(f"done! Total elapsed time: {round(t1-t0, 2)} [s]")

    return output
