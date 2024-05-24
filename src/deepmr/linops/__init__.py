"""Sub-package containing Linear operators for MR imaging.

Each operator is derived from :func:`torch.nn.Module`.
As a consequence, they can be composed to build neural network architectures.

Currently implemented linear operator include:
    
* centered n-dimensional FFT and iFFT with masking and low rank subspace projection;
* n-dimensional NUFFT with embedded low rank subspace projection;
* coil combination and projection operators.

"""

from . import _base
from . import _coil
from . import _fft
from . import _nufft
from . import _linalg

from ._base import *  # noqa
from ._coil import *  # noqa
from ._fft import *  # noqa
from ._nufft import *  # noqa
from ._linalg import * # noqa

__all__ = ["aslinearoperator", "vstack", "hstack"]
__all__.extend(_base.__all__)
__all__.extend(_coil.__all__)
__all__.extend(_fft.__all__)
__all__.extend(_nufft.__all__)
__all__.extend(_linalg.__all__)

#%% functions
import torch as _torch
import numpy as _np

def aslinearoperator(A, AH=None, AHA=None):
    """
    Create LinearOperator from input tensor or function handle.

    Parameters
    ----------
    A : deepmr.linops.Linop | torch.Tensor | np.ndarray | Callable
        Linear operator, or Tensor / NDArray describing the operator,
        or function handle describing the forward operator.
        If ``type(A) == Linop``, ``A`` is directly returned.
    AH : Callable, optional
        When forward operator is specified as a ``Callable``, the 
        adjoint must be specified as a ``Callable`` as well. 
        The default is ``None`` (ignored for ``type(A) == Linop | torch.Tensor | np.ndarray").
    AHA : Callable, optional
        When forward operator is specified as a ``Callable``,
        normal operator can be specified as a ``Callable`` as well,
        if an efficient implementation is available. The default is ``None`` (i.e., ``A(x) = A.H(A(x))``).

    Returns
    -------
    Op : deepmr.linops.Linop
        Linear operator corresponding to the inputs.
        
    
    Examples
    --------
    
    >>> import deepmr
    >>> import torch
    
    Create a tensor:
        
    >>> A = torch.rand(3, 3)
    >>> x = torch.rand(3)
    
    Transform ``A`` into a linear operator:
        
    >>> Aop = deepmr.linops.aslinearoperator(A)
    
    Apply operator
    
    >>> y = A @ x
    >>> yop = Aop(x)
    
    The results are the same:
        
    >>> torch.equal(y, yop)
    True
            
    """
    if isinstance(A, _base.Linop):
        return A
    if isinstance(A, (_torch.Tensor, _np.ndarray)):
        A = _torch.as_tensor(A)
        return  _base.MatrixOp(A)
    else:
        assert AH is not None, "If A is a function handle, please provide adjoint operation handle"
        return  _base.LambdaOp(A, AH, AHA)
    
    
def vstack(*ops_or_tensors):
    """
    Vertical stack of Linear operators

    Parameters
    ----------
    *ops_or_tensors : torch.Tensor | deepmr.linops.Linop
        operators or tensors to be stacked.

    Returns
    -------
    list | deepmr.linops.Linop
        If input is a list of tensor, return a list of tensors.
        Otherwise, vertically stack operators.

    """
    if isinstance(ops_or_tensors[0], _torch.Tensor):
        return list(ops_or_tensors)
    else:
        return _base.Vstack(ops_or_tensors)


def hstack(ops_or_tensors):
    """
    Horizontal stack of Linear operators

    Parameters
    ----------
    *ops_or_tensors : torch.Tensor | deepmr.linops.Linop
        operators or tensors to be stacked.

    Returns
    -------
    list | deepmr.linops.Linop
        If input is a list of tensor, return a list of tensors.
        Otherwise, horizontally stack operators.

    """
    if isinstance(ops_or_tensors[0], _torch.Tensor):
        raise NotImplementedError
    else:
        return _base.Hstack(ops_or_tensors)    
    