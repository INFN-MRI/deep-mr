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
        """Initiate the linear operator."""
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
    def T(self): # noqa
        return self.H
    
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
        return self._normal_linop()

    def __add__(self, other):
        """Reload the + symbol."""
        return Add([self, other])
    
    def __matmul__(self, other):
        """Reload the @ symbol."""
        if isinstance(other, Linop):
            return Compose([self, other])
        elif isinstance(other, torch.Tensor):
            return self.forward(other)
        else:
            raise NotImplementedError(
                f"Only Linops or Tensors, rather than '{type(other)}' are allowed as arguments for this function."
            )

    def __mul__(self, other):
        """Reload the * symbol."""
        if np.isscalar(other):
            return Multiply(self, other)
        else:
            raise NotImplementedError(
                f"Only scalars, rather than '{type(other)}' are allowed as arguments for this function."
            )

    def __rmul__(self, other):
        """Reload the * symbol."""
        if np.isscalar(other):
            return Multiply(self, other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Reload the - symbol."""
        return self.__add__(-other)

    def __neg__(self):
        """Reload the - symbol."""
        return -1 * self

    def to(self, *args, **kwargs):
        """Copy to different devices."""
        for prop in self.__dict__.keys():
            try:
                self.__dict__[prop] = self.__dict__[prop].to(*args, **kwargs)
            except Exception:
                pass
        return self
    
    def cpu(self):
        """Alternative syntax for '.to("cpu")'."""
        return self.to("cpu")
    
    def cuda(self, device=None):
        """
        Alternative syntax for '.to("cuda")' or '.to("cuda:device")'.
        
        Parameters
        ----------
        device : int, optional
            CUDA device number. By default, move to the current CUDA device.
        
        """
        if device is None:
            return self.to("cuda")
        else:
            return self.to("cuda:" + str(device))

    def maxeig(self, x, lamda=0.0, rho=1.0, niter=10, device=None):
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

    def pinv(self, method="cg", x0=None, lamda=0.0, niter=None, tol=None, bias=None, rho=0.0, device=None):
        """
        Invert operator using iterative least squares.

        Parameters
        ----------
        method : str, optional
            Select between Conjugate Gradient (``"cg"``)
            or LSMR (``"lsmr"``). The default is "cg".
        x0 : torch.Tensor, optional
            Initial estimate for solution. Required for ``lsmr`` only 
            (can be zeroes, in which case is used only to estimate the expected shape).
        bias : torch.Tensor, optional
            Bias for Tikhonov regularization. The default is ``None``.
        lamda : float, optional
            Tikhonov regularization. The default is ``0.0``.
        niter : int, optional
            Number of iterations.
            The default is ``10`` (``method == "cg"``) 
            or ``4`` (``method == "lsmr"``).
        tol : float, optional
            Tolerance for convergence.
            The default is is ``None`` (run until ``niter``, ``method == "cg"``) 
            or ``(atol=1e-6, btol=1e-6, conlim=1e8)`` (``method == "lsmr"``).
        rho : float, optional
            Tikhonov regularization for additional bias, e.g., for ADMM dual variable. 
            The default is ``0.0``.
        device : str, optional
            Computational device. The default is ``None`` (use same device as measured data).

        Returns
        -------
        output : torch.Tensor
            Least squares solution.

        """
        if method == "cg":
            if niter is None:
                niter = 10
            return ConjugateGrad(self, x0, lamda, rho, bias, niter, tol, device)
        elif method == "lsmr":
            if niter is None:
                niter = 4
            if tol is None:
                tol = [1e-6, 1e-6, 1e8]
            return LSMR(self, x0, lamda, rho, bias, niter, tol, device)

# %% base operators
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
    

class Vstack(Linop):
    """
    Vertically stack linear operators.

    Creates a Linop that applies linops independently,
    and concatenates outputs.
    In matrix form, this is equivalant to given matrices {A1, ..., An},
    returns [A1.T, ..., An.T].T.

    Attributes
    ----------
    linops : List[Linopts] 
        List of linops to be vertically stacked.

    """
    
    def __init__(self, linops):
        super().__init__()
        self.linops = linops

    def forward(self, input):
        output = []
        for n in range(len(self.linops)):
            output.append(self.linops[n](input))
        return output

    def to(self, device):
        return Vstack([linop.to(device) for linop in self.linops])

    def _adjoint_linop(self):
        return Hstack([linop.H for linop in self.linops])


class Hstack(Linop):
    """
    Horizontally stack linear operators.

    Creates a Linop that splits the input, applies Linops independently,
    and sums outputs.
    In matrix form, this is equivalant to given matrices {A1, ..., An},
    returns [A1, ..., An].

    Attributes
    ----------
    linops : List[Linopts] 
        List of linops to be horizontally stacked.

    """
    
    def __init__(self, linops):
        super().__init__()
        self.linops = linops

    def forward(self, input):
        output = self.linops[0](input[0])
        for n in range(1, len(self.linops)):
            output = output + self.linops[n](input[n])
        return output

    def to(self, device):
        return Hstack([linop.to(device) for linop in self.linops])

    def _adjoint_linop(self):
        return Vstack([linop.H for linop in self.linops])


class Identity(Linop):
    """
    Identity linear operator.

    Returns input directly.

    """

    def __init__(self):
        super().__init__()

    def forward(self, input): # noqa
        return input

    def _adjoint_linop(self):
        return self

    def _normal_linop(self):
        return self


class ConjugateGrad(torch.nn.Module):
    """Pseudo Inverse of linear operator via conjugate gradient."""
    
    def __init__(self, A, x0, lamda, rho, bias, niter, tol, device):
        super().__init__()
        self.A = A
        self.x0 = x0
        self.rho = rho
        self.lamda = lamda
        self.lamda_p_rho_I = (self.lamda + self.rho) * Identity()
        if bias is not None and self.lamda != 0.0:
            self.bias0 = bias.clone()
        else:
            self.bias0 = None
        self.bias = None
        self.niter = niter
        self.tol = tol
        self.device = device
                
    def __getitem__(self, bias):
        assert self.rho != 0, "if variable bias is specified, rho must be != 0"
        self.bias = bias.clone()
        return self
        
    def forward(self, input):
        # create operator
        AHA = self.A.N
        if self.lamda != 0.0 or self.rho != 0.0:
            AHA = AHA + self.lamda_p_rho_I
        
        # manipulate input
        y = input.clone()
        
        # calculate right hand side
        AHy = self.A.H(y)
        
        # initial solution
        if self.x0 is not None:
            x0 = self.x0
        else:
            x0 = 0 * AHy

        if self.lamda != 0.0 and self.bias0 is not None:
            AHy = AHy + self.lamda * self.bias0
        if self.rho != 0.0:
            assert self.bias is not None, "If rho != 0, variable bias must be specified (e.g., ADMM)."
            AHy = AHy + self.rho * self.bias
                
        return _linalg.conjgrad(AHA, AHy, x0, self.niter, self.tol, self.device)
    
    
class LSMR(torch.nn.Module):
    """Pseudo Inverse of linear operator via LSMR."""
    
    def __init__(self, A, x0, lamda, rho, bias, niter, tol, device):
        super().__init__()
        self.A = A
        self.x0 = x0
        
        # case 0)
        # min || y - A(x) ||_2
        if bias is None and lamda == 0.0 and rho == 0.0:
            self.damp = 0.0
            self.sqrt_lamda_I = None
            self.sqrt_lamda_bias0 = None
            self.sqrt_rho_I = None
            self.sqrt_rho_bias = None
            self.sqrt_rho = 0.0
        
        # case 1)
        # min || y - A(x)            ||
        #     || 0   sqrt(lamda) * I ||_2
        if bias is None and lamda != 0.0 and rho == 0.0:
            self.damp = lamda**0.5
            self.sqrt_lamda_I = None
            self.sqrt_lamda_bias0 = None
            self.sqrt_lamda = 0.0
            self.sqrt_rho_I = None
            self.sqrt_rho_bias = None
            self.sqrt_rho = 0.0

        # case 2)
        # min || y      -      A(x)            ||
        #     || sqrt(lamda) * bias0 sqrt(lamda) * I ||_2
        if bias is not None and lamda != 0.0 and rho == 0.0:
            self.damp = 0.0
            self.sqrt_lamda_I = lamda**0.5 * Identity()
            self.sqrt_lamda = lamda**0.5
            self.sqrt_lamda_bias0 = lamda**0.5 * bias.clone()
            self.sqrt_rho_I = None
            self.sqrt_rho_bias = None
            self.sqrt_rho = 0.0
        
        # case 3)
        # min || y        -        A(x)          ||
        #     || sqrt(rho) * bias  sqrt(rho) * I ||_2
        if bias is None and lamda == 0.0 and rho != 0.0:
            self.damp = 0.0
            self.sqrt_lamda_I = None
            self.sqrt_lamda_bias0 = None
            self.sqrt_lamda = 0.0
            self.sqrt_rho_I = rho**0.5 * Identity()
            self.sqrt_rho_bias = None
            self.sqrt_rho = rho**0.5
        
        # case 4)
        # min || y        -        A(x)            ||
        #     || 0                 sqrt(lamda) * I ||
        #     || sqrt(rho) * bias  sqrt(rho) * I   ||_2
        if bias is None and lamda != 0.0 and rho != 0.0:
            self.damp = 0.0
            self.sqrt_lamda_I = lamda**0.5 * Identity()
            self.sqrt_lamda_bias0 = None
            self.sqrt_lamda = lamda**0.5
            self.sqrt_rho_I = rho**0.5 * Identity()
            self.sqrt_rho_bias = None
            self.sqrt_rho = rho**0.5
        
        # case 5)
        # min || y      -      A(x)                  ||
        #     || sqrt(lamda) * bias0 sqrt(lamda) * I ||
        #     || sqrt(rho)   * bias  sqrt(rho) * I   ||_2
        if bias is not None and lamda != 0.0 and rho != 0.0:
            self.damp = 0.0
            self.sqrt_lamda_I = lamda**0.5 * Identity()
            self.sqrt_lamda_bias0 = lamda**0.5 * bias
            self.sqrt_lamda = lamda**0.5
            self.sqrt_rho_I = rho**0.5 * Identity()
            self.sqrt_rho_bias = None
            self.sqrt_rho = rho**0.5

        self.niter = niter
        self.tol = tol
        self.device = device
                
    def __getitem__(self, bias):
        assert self.rho != 0, "if variable bias is specified, rho must be != 0"
        self.sqrt_rho_bias = self.sqrt_rho * bias
        if self.sqrt_lamda != 0 and self.sqrt_lamda_bias0 is None:
            self.sqrt_lamda_bias0 = 0 * bias
        return self
        
    def forward(self, input):
        # create operator
        ops = [self.A]
        if self.sqrt_lamda_I is not None:
            ops = ops + [self.sqrt_lamda_I]
        if self.sqrt_rho_I is not None:
            ops = ops + [self.sqrt_rho_I]
                
        # create stacked operator
        if len(ops) > 1:
            A = Vstack(ops)
        else:
            A = ops[0]

        # manipulate input
        y = [input.clone()]
        if self.sqrt_lamda_bias0 is not None:
            y = y + [self.sqrt_lamda_bias0]
        if self.sqrt_rho_bias is not None:
            y = y + [self.sqrt_rho_bias]
        
        # create stacked input
        if len(y) == 1:
            y = y[0]
            
        # initial solution
        if self.x0 is not None:
            x0 = self.x0
        else:
            x0 = 0 * A.H(y)
                                    
        return _linalg.lsmr(A, y, x0, self.niter, self.damp, self.tol[0], self.tol[1], self.tol[2], self.device)
    
    
class LambdaOp(Linop):
    """Lambda function linear operator."""
    
    def __init__(self, forw, adj, norm=None):
        """
        Lambda operator constructor.

        Parameters
        ----------
        forw : Callable
            Callable tensor representing
            the forward operator.
        adj : Callable
            Callable tensor representing
            the adjoint operator.
        norm : Callable, optional
            Callable tensor representing
            the normal operator.

        """
        super().__init__()
        self._forw = forw
        self._adj = adj
        
        # define normal operator if not specified
        if norm is None:
            norm = lambda x : adj(forw(x))
        self._norm = norm
        
    def forward(self, input): # noqa
        return self._forw(input)
    
    def _adjoint_linop(self):
        return LambdaOp(self._adj, self._forw, self._norm)
    
    def _normal_linop(self):
        return LambdaOp(self._norm, self._norm, self._norm)
    
    
class MatrixOp(Linop):
    """
    Matrix as a linear operators.
    
    Also supports batch of matrices.
    """
    
    def __init__(self, mat):
        """
        Matrix operator constructor.

        Parameters
        ----------
        mat : torch.Tensor
            Dense or sparse tensor representing
            the operator.

        """
        super().__init__()
        self._mat = mat.clone()
        
    @property
    def T(self):
        transp_mat = self._mat.transpose(-2, -1)
        return MatrixOp(transp_mat)
        
    def forward(self, input):
        return self._mat @ input
    
    def _adjoint_linop(self):
        adj_mat = self._mat.conj().transpose(-2, -1)
        return MatrixOp(adj_mat)
    
    def _normal_linop(self):
        adj_mat = self._mat.conj().transpose(-2, -1)
        norm_mat = adj_mat @ self._mat
        return GramMatrixOp(norm_mat)
        
    
class GramMatrixOp(MatrixOp):
    """Gram Matrix operator."""
    
    def _adjoint_linop(self):
        return self

    def _normal_linop(self):
        return self
    
    