"""Base linear operator."""

__all__ = ["Linop"]

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

