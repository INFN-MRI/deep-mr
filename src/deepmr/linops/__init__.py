"""Sub-package containing Linear operators for MR imaging.

Each operator is derived from :func:`torch.nn.Module`.
As a consequence, they can be composed to build neural network architectures.

Currently implemented linear operator include:
    
* centered n-dimensional FFT and iFFT with masking and low rank subspace projection;
* n-dimensional NUFFT with embedded low rank subspace projection;
* coil combination and projection operators.

"""

from . import base as _base
from . import coil as _coil
from . import fft as _fft
from . import nufft as _nufft

from .base import *  # noqa
from .coil import *  # noqa
from .fft import *  # noqa
from .nufft import *  # noqa

__all__ = []
__all__.extend(_base.__all__)
__all__.extend(_coil.__all__)
__all__.extend(_fft.__all__)
__all__.extend(_nufft.__all__)


#%% functions
import torch as _torch
import numpy as _np

def aslinearoperator(A, AH=None, AHA=None):
    """
    Create LinearOperator from input tensor or function handle.

    Parameters
    ----------
    A : deepmr.Linop | torch.Tensor | np.ndarray | Callable
        Linear operator, or Tensor / NDArray describing the operator,
        or function handle describing the forward operator.
        If ``type(A) == Linop``, ``A`` is directly returned.
    AH : Callable, optional
        When forward operator is specified as a ``Callable``, the 
        adjoint must be specified as a ``Callable`` as well. 
        The default is ``None`` (ignored for ``type(A) == Linop | torch.Tensor | np.ndarray").
    AHA : TYPE, optional
        When forward operator is specified as a ``Callable``,
        normal operator can be specified as a ``Callable`` as well,
        if an efficient implementation is available. The default is ``None`` (i.e., ``A(x) = A.H(A(x))``).

    Returns
    -------
    Op : deepmr.Linop
        Linear operator corresponding to the inputs.
        

    """
    if isinstance(A, _base.Linop):
        return A
    if isinstance(A, (_torch.Tensor, _np.ndarray)):
        A = _torch.as_tensor(A)
        return  _base.MatrixOp(A)
    else:
        assert AH is not None, "If A is a function handle, please provide adjoint operation handle"
        return  _base.LambdaOp(A, AH, AHA)
        
    