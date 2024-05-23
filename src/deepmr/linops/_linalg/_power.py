"""Maximum eigenvalue estimation."""

__all__ = ["power_method"]

import torch

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
    max_eig : float
        The maximum eigenvalue of the operator ``A``.
    x : torch.Tensor
        Eigenvector corresponding to maximum eigenvalue.

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

    return max_eig.item() ** 0.5, x

