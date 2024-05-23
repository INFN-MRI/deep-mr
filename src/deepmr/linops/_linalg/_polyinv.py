"""Polynomial Inversion algorithm."""

__all__ = ["polynomial_inversion"]

from ... import _precond

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

