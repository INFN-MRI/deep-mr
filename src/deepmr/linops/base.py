"""Base linear operator."""

__all__ = ["Linop", "Identity"]

import numpy as np
import torch
import torch.nn as nn

from . import _linalg


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

    def __init__(self):
        r"""
        Initiate the linear operator.
        """
        super().__init__()

    def _adjoint_linop(self):
        raise NotImplementedError

    def _normal_linop(self):
        return self.H * self

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

    @property
    def N(self):
        r"""
        Return normal linear operator.

        A normal linear operator :math:`A^HA` for
        a linear operator :math:`A`.

        Returns
        -------
        Linop
            Normal linear operator.

        """
        if self.normal is None:
            self.normal = self._normal_linop()
        return self.normal

    def __add__(self, other):
        r"""
        Reload the + symbol.
        """
        return Add([self, other])

    def __mul__(self, other):
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, Linop):
            return Compose([self, other])
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
        return self

    def max_eig(self, x, lamda=0.0, rho=1.0, niter=10, device=None):
        """
        Compute maximum eigenvalue using Power Iteration method.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with the same shape of the expected A input.
        lamda : float, optional
            Tikhonov regularization. The default is ``0.0``.
        rho : float, optional
            Relaxation parameter. The default is ``1.0``.
        niter : int, optional
            Number of Power Method iterations .
            The default is ``10``.
        device : str, optional
            Computational device. The default is ``None`` (use same device as ``y``).

        Returns
        -------
        LL : float
            Maximum eigenvalue of A (i.e., Lipschitz constant).

        """
        # build random array
        x = torch.rand_like(x)

        # add regularization
        _AHA = rho * self.N + lamda * Identity()

        return _linalg.power_method(_AHA, x, niter=niter, device=device)

    def solve(
        self,
        AHy,
        bias=None,
        lamda=0.0,
        rho=1.0,
        niter=None,
        tol=None,
        method="cg",
        device=None,
    ):
        """
        Invert operator using iterative least squares or polynomial inversion.

        Parameters
        ----------
        AHy : torch.Tensor
            Adjoint AH of measurementoperator A applied to the measured data y.
        bias : torch.Tensor, optional
            Bias for L2 regularization. The default is ``None``.
        lamda : float, optional
            Tikhonov regularization. The default is ``0.0``.
        rho : float, optional
            Relaxation parameter. The default is ``1.0``.
        niter : int, optional
            Number of CG iterations / polynomial degree.
            The default is ``10`` (``method == "cg"``)
            or ``2`` (``method == "pi"``).
        tol : float, optional
            Tolerance for convergence. Ignored if ``method == "pi"``.
            The default is ``None`` (run until ``niter``).
        method : str, optional
            Select between Conjugate Gradient (``"cg"``)
            or Polynomial Inversion (``"pi"``). The default is "cg".
        device : str, optional
            Computational device. The default is ``None`` (use same device as ``y``).

        Returns
        -------
        output : torch.Tensor
            Least squares solution.

        """
        # get adjoint
        _AHy = rho * AHy
        if bias is not None and lamda != 0.0:
            _AHy = _AHy + lamda * bias

        # add regularization
        _AHA = rho * self.N + lamda * Identity()

        # solve
        if method == "cg":
            if niter is None:
                niter = 10
            return _linalg.cg_solve(_AHA, _AHy, niter=niter, tol=tol, device=device)
        elif method == "pi":
            if niter is None:
                niter = 2
            return _linalg.polynomial_inversion(
                _AHA, _AHy, lamda, degree=niter, tol=tol, device=device
            )


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
        super().__init__()
        self.linops = linops

    def forward(self, input):
        output = 0
        for linop in self.linops:
            output += linop(input)
        return output

    def to(self, device):
        return Add([linop.to(device) for linop in self.linops])

    def _adjoint_linop(self):
        return Add([linop.H for linop in self.linops])


class Compose(Linop):
    r"""
    Matrix multiplication of linear operators.

    .. math::
        A*B*x = A(B(x))

    """

    def __init__(self, linops):
        super().__init__()
        self.linops = linops

    def forward(self, input):
        output = input
        for linop in self.linops[::-1]:
            output = linop(output)
        return output

    def to(self, device):
        return Compose([linop.to(device) for linop in self.linops])

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
        super().__init__()
        self.a = a
        self.linop = linop

    def forward(self, input):
        ax = input * self.a
        return self.linop(ax)

    def to(self, device):
        return Multiply(self.linop.to(device), self.a)

    def _adjoint_linop(self):
        return Multiply(self.linop, self.a)


class Identity(Linop):
    """I
    dentity linear operator.

    Returns input directly.

    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

    def _adjoint_linop(self):
        return self

    def _normal_linop(self):
        return self
