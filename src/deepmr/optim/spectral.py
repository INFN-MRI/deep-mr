"""Power Method for maximum eigenvalue estimation."""

__all__ = ["power_method"]

import torch

from .. import linops as _linops


@torch.no_grad()
def power_method(A, x, AH=None, AHA=None, device=None, niter=10, tol=None):
    r"""
    Use power iteration to calculate the spectral norm of a Linop.

    From MIRTorch (https://github.com/guanhuaw/MIRTorch/blob/master/mirtorch/alg/spectral.py)

    Parameters
    ----------
    A : Callable | torch.Tensor | np.ndarray
        Linear opeartor
    x : torch.Tensor | np.ndarray
        Initial guess of singular vector corresponding to max singular value
    AH : Callable | torch.Tensor | np.ndarray, optional
        Adjoint operator AH = A.H. If not provided, attempt to estimate from A.
        The default is ``None``.
    AHA : Callable | torch.Tensor | np.ndarray, optional
        Normal operator AHA = AH * A. If not provided, estimate from A and AH.
        The default is ``None``.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).
    niter : int, optional
        Maximum number of iterations. The default is ``10``.
    tol : float, optional
        Stopping criterion. The default is ``None`` (run until ``niter``).

    Returns
    -------
    float
        The spectral norm of the operator ``A``.

    """
    # cast to torch if required
    x = torch.as_tensor(x)

    # get device device
    if device is None:
        device = x.device

    # put on device
    x = x.to(device)
    if A is not None:
        if isinstance(A, _linops.Linop):
            A = A.to(device)
        elif callable(A) is False:
            A = torch.as_tensor(A, dtype=x.dtype, device=device)

    # estimate self-adjoint
    if AHA is None:
        if isinstance(A, _linops.Linop):
            AHA = A.H * A
        elif callable(AHA) is False:
            AHA = lambda x: A.conj().T @ A @ x
        else:
            assert (
                AH is not None
            ), "If AHA is not provided and A is neither a deepr.linops.Linop nor a np.ndarray / torch.Tensor, please provide AH as function handle."
            AHA = lambda x: AH(A(x))

    # perform iterations
    ratio_old = float("inf")
    for iter in range(niter):
        # check for convergence
        if tol is not None:
            if callable(A):
                ratio = torch.linalg.norm(A(x)) / torch.linalg.norm(x)
            else:
                ratio = torch.linalg.norm(A @ x) / torch.linalg.norm(x)

            # actual checking
            if torch.abs(ratio - ratio_old) / ratio < tol:
                break
                ratio_old = ratio

        # update eigenvector
        AHAx = AHA(x)
        max_eig = torch.linalg.norm(AHAx)
        x = AHAx / max_eig

    # compute actual maximum eigenvalue
    if A is not None:
        if callable(A):
            max_eig = torch.linalg.norm(A(x))
        else:
            max_eig = torch.linalg.norm(A @ x)

    return max_eig.item()
