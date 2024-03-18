"""Base linear operator."""

__all__ = ["Linop"]

import numpy as np
import torch
import torch.nn as nn


class Linop(nn.Module):
    r"""
    Abstraction of linear operators as matrices :math:`y = A*x`.

    This is adapted from `MIRTorch <https://github.com/guanhuaw/MIRTorch>`_. The following is copied from the corresponding docstring.
    The implementation follow the `SigPy <https://github.com/mikgroup/sigpy>`_ and `LinearmapAA <https://github.com/JeffFessler/LinearMapsAA.jl>`_.

    Common operators, including +, -, *, are overloaded. One may freely compose operators as long as the size matches.

    New linear operators require to implement `_apply` (forward, :math:`A`) and `_adjoint` (conjugate adjoint, :math:`A'`) functions, as well as size.
    Recommendation for efficient backpropagation (but you do not have to do this if the AD is efficient enough):

    .. code-block:: python

        class forward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data_in):
                return forward_func(data_in)
            @staticmethod
            def backward(ctx, grad_data_in):
                return adjoint_func(grad_data_in)
        forward_op = forward.apply

        class adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data_in):
                return forward_func(data_in)
            @staticmethod
            def backward(ctx, grad_data_in):
                return adjoint_func(grad_data_in)
        adjoint_op = adjoint.apply


    """

    def __init__(self, ndim):
        r"""
        Initiate the linear operator.
        """
        super().__init__()
        self.ndim = ndim

    def _adjoint_linop(self):
        raise NotImplementedError

    @property
    def H(self):
        r"""
        Return adjoint linear operator.

        An adjoint linear operator :math:`A^H` for
        a linear operator :math:`A` is defined as:

        .. math:
            \left< A x, y \right> = \left< x, A^H, y \right>

        Returns
        -------
        Linop
            Adjoint linear operator.

        """
        return self._adjoint_linop()

    def __add__(self, other):
        r"""
        Reload the + symbol.
        """
        return Add(self, other)

    def __mul__(self, other):
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, Linop):
            return Compose(self, other)
        elif isinstance(other, torch.Tensor):
            if not other.shape:
                return Multiply(self, other)
            return self.apply(other)
        else:
            raise NotImplementedError(
                f"Only scalers, Linearmaps or Tensors, rather than '{type(other)}' are allowed as arguments for this function."
            )

    def __rmul__(self, other):
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, torch.Tensor) and not other.shape:
            return Multiply(self, other)
        else:
            return NotImplemented

    def __sub__(self, other):
        r"""
        Reload the - symbol.
        """
        return self.__add__(-other)

    def __neg__(self):
        r"""
        Reload the - symbol.
        """
        return -1 * self

    def to(self, *args, **kwargs):
        r"""
        Copy to different devices
        """
        for prop in self.__dict__.keys():
            try:
                self.__dict__[prop] = self.__dict__[prop].to(*args, **kwargs)
            except Exception:
                pass


class Add(Linop):
    r"""
    Addition of linear operators.

    .. math::
         (A+B)*x = A(x) + B(x)

    Attributes
    ----------
    linops : Iterable(Linop)
        List of linear operators to be summed.

    """

    def __init__(self, linops):
        self.linops = linops
        ndim = np.unique([linop.ndim for linop in self.linops])
        assert (
            len(ndim) == 1
        ), "Error! All linops must have the same spatial dimensionality."
        super().__init__(ndim.item())

    def forward(self, input):
        output = 0
        for linop in self.linops:
            output += linop(input)

    def _adjoint_linop(self):
        return Add([linop.H for linop in self.linops])


class Compose(Linop):
    r"""
    Matrix multiplication of linear operators.

    .. math::
        A*B*x = A(B(x))

    """

    def __init__(self, linops):
        self.linops = linops
        ndim = np.unique([linop.ndim for linop in self.linops])
        assert (
            len(ndim) == 1
        ), "Error! All linops must have the same spatial dimensionality."
        super().__init__(ndim.item())

    def forward(self, input):
        output = input
        for linop in self.linops[::-1]:
            output = linop(output)
        return output

    def _adjoint_linop(self):
        return Compose([linop.H for linop in self.linops[::-1]])


class Multiply(Linop):
    r"""
    Scaling linear operators.

    .. math::
        a*A*x = A(ax)

    Attributes
    ----------
    a : float, int
        Scaling factor.
    linop : Linop
        Linear operator A.

    """

    def __init__(self, linop, a):
        self.a = a
        self.linop = linop
        super().__init__(linop.ndim)

    def forward(self, input):
        ax = input * self.a
        return self.linop(ax)

    def _adjoint_linop(self):
        return Multiply(self.linop, self.a)


class Identity(Linop):
    """I
    dentity linear operator.

    Returns input directly.

    """

    def __init__(self, ndim):
        super().__init__(ndim)

    def _apply(self, input):
        return input

    def _adjoint_linop(self):
        return self

    def _normal_linop(self):
        return self


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

    x = torch.zeros_like(b)

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
