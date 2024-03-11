"""Base linear operator."""

__all__ = ["Linop", "NormalLinop"]

import torch

import deepinv as dinv
from deepinv.utils import zeros_like

class Linop(dinv.physics.LinearPhysics):
    """
    Abstraction class for Linear operators.

    This is an alias for ``deepinv.physics.LinearPhysics``,
    but provides a convenient method for ``A_adjoint`` (i.e., ``Linop.H``)
    """

    def __init__(self, ndim, *args, **kwargs):
        self._ndim = ndim
        super().__init__(*args, **kwargs)
        
    @property
    def H(self):
        A = lambda x: self.A_adjoint(x)
        A_adjoint = lambda x: self.A(x)
        noise = self.noise_model
        sensor = self.sensor_model
        return Linop(
            self._ndim,
            A=A,
            A_adjoint=A_adjoint,
            noise_model=noise,
            sensor_model=sensor,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def __add__(self, other):
        tmp = super().__add__(other)
        return Linop(
            self._ndim,
            A=tmp.A,
            A_adjoint=tmp.A_adjoint,
            noise_model=tmp.noise_model,
            sensor_model=tmp.sensor_model,
            max_iter=tmp.max_iter,
            tol=tmp.tol,
        )

    def __mul__(self, other):
        tmp = super().__mul__(other)
        return Linop(
            self._ndim,
            A=tmp.A,
            A_adjoint=tmp.A_adjoint,
            noise_model=tmp.noise_model,
            sensor_model=tmp.sensor_model,
            max_iter=tmp.max_iter,
            tol=tmp.tol,
        )
    
    def solve(self, b, max_iter=1e2, tol=1e-5, lamda=0.0):
        return conjugate_gradient(self._ndim, self.A, b, max_iter, tol, lamda)
    
    def A_dagger(self, y):
        Aty = self.A_adjoint(y)

        overcomplete = Aty.flatten().shape[0] < y.flatten().shape[0]

        if not overcomplete:
            A = lambda x: self.A(self.A_adjoint(x))
            b = y
        else:
            A = lambda x: self.A_adjoint(self.A(x))
            b = Aty

        x = conjugate_gradient(A=A, b=b, max_iter=self.max_iter, tol=self.tol)

        if not overcomplete:
            x = self.A_adjoint(x)

        return x
    
    def prox_l2(self, z, y, gamma):
        b = self.A_adjoint(y) + 1 / gamma * z
        H = lambda x: self.A_adjoint(self.A(x)) + 1 / gamma * x
        x = conjugate_gradient(H, b, self.max_iter, self.tol)
        return x
    
    
class NormalLinop(Linop):
    """
    Special case of Linop where A.H = A (self-adjoint).

    """
    def __init__(self, ndim, *args, **kwargs):
        super().__init__(ndim, *args, **kwargs)
        self.A_adjoint = self.A
        
    def A_dagger(self, y):
        return self.solve(self._ndim, self.A, y, self.max_iter, self.tol)
    
    def prox_l2(self, z, y, gamma):
        b = y + 1 / gamma * z
        H = lambda x: self.A(x) + 1 / gamma * x
        x = conjugate_gradient(H, b, self.max_iter, self.tol)
        return x
    
    def maxeig(self, input, max_iter=10, tol=1e-6):
        x = torch.randn(input.shape)
        return power_iter(self.A, x, max_iter, tol)
        
    
# %% local utils
def conjugate_gradient(ndim, _A, b, max_iter=1e2, tol=1e-5, lamda=0.0):
    """
    Batched Conjugata Gradient solver.
    
    Supports complex inputs.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
        Each input axis before the last ndim are treated as batch dimensions,
        i.e., prod(b.shape[:-ndim]) subproblems are solved simultaneously.
    _A : Linop
        Linear operator.
    b : torch.Tensor
        Complex valued observation of shape (..., n_ndim, ..., n_0).
    max_iter : int, optional
        Maximum number of iterations. The default is 1e2.
    tol : float, optional
        Convergence threshold. The default is 1e-5.
    lamda : float, optional
        Tikonhov regularization strengh. The default is 0.0.

    Returns
    -------
    output torch.Tensor
        Solution of the problem.

    """
    def dot(s1, s2):
        dot = s1.conj() * s2
        dot = dot.reshape(*s1.shape[:-ndim], -1).sum(axis=-1)
        for n in range(ndim):
            dot = dot[..., None]
        return dot
    
    if lamda != 0:
        def A(x):
            return _A(x) + lamda * x
    else:
        def A(x):
            return _A(x)
        
    x = zeros_like(b)

    r = b
    p = r
    rsold = dot(r, r)

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / dot(p, Ap)
        x = x + p * alpha
        r = r + Ap * (-alpha)
        rsnew = torch.real(dot(r, r))
        if rsnew.sqrt() < tol:
            break
        p = r + p * (rsnew / rsold)
        rsold = rsnew

    return x

@torch.no_grad()
def power_iter(A, x0, max_iter=2, tol=1e-6):
    r"""
    Use power iteration to calculate the spectral norm of a LinearMap.
    
    From MIRTorch (https://github.com/guanhuaw/MIRTorch/blob/master/mirtorch/alg/spectral.py)

    Args:
        A: a LinearMap
        x0: initial guess of singular vector corresponding to max singular value
        max_iter: maximum number of iterations
        tol: stopping tolerance

    Returns:
        The spectral norm (sig1) and the principal right singular vector (x)
        
    """

    x = x0
    max_eig = float("inf")
    for iter in range(max_iter):
        Ax = A(x)
        max_eig = torch.norm(Ax)
        x = x / max_eig

    return max_eig

