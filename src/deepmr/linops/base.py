"""Base linear operator."""

__all__ = ["Linop", "Identity"]

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
        ndim = np.unique([linop.ndim for linop in linops])
        assert (
            len(ndim) == 1
        ), "Error! All linops must have the same spatial dimensionality."
        super().__init__(ndim.item())
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
        ndim = np.unique([linop.ndim for linop in linops])
        assert (
            len(ndim) == 1
        ), "Error! All linops must have the same spatial dimensionality."
        super().__init__(ndim.item())
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
        super().__init__(linop.ndim)
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

    def __init__(self, ndim):
        super().__init__(ndim)

    def forward(self, input):
        return input

    def _adjoint_linop(self):
        return self

    def _normal_linop(self):
        return self
