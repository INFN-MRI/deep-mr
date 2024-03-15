"""Proximal Gradient Method iteration."""

__all__ = ["pgd_solve", "PGDStep"]

import numpy as np
import torch

import torch.nn as nn

def pgd_solve(input, step, AHA, D, niter=10, accelerate=True, device=None, tol=None):
    """
    Solve inverse problem using Proximal Gradient Method.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Signal to be reconstructed. Assume it is the adjoint AH of measurement
        operator A applied to the measured data y (i.e., input = AHy).
    step : float
        Gradient step size; should be <= 1 / max(eig(AHA)).
    AHA : Callable
        Normal operator AHA = AH * A.
    D : Callable
        Signal denoiser for plug-n-play restoration.
    niter : int, optional
        Number of iterations. The default is 10.
    accelerate : bool, optional
        Toggle Nesterov acceleration (``True``, i.e., FISTA) or
        not (``False``, ISTA). The default is ``True``.
    device : str, optional
        Device on which the wavelet transform is computed. 
        The default is ``None`` (infer from input).
    tol : float, optional
        Stopping condition.
        The default is ``None`` (run until niter).

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Reconstructed signal.

    """
    # cast to numpy if required
    if isinstance(input, np.ndarray):
        isnumpy = True
        input = torch.as_tensor(input)
    else:
        isnumpy = False
        
    # keep original device
    idevice = input.device
    if device is None:
        device = idevice
        
    # put on device
    input = input.to(device)
    
    # assume input is AH(y), i.e., adjoint of measurement operator
    # applied on measured data
    AHy = input.clone()
    
    # initialize Nesterov acceleration
    if accelerate:
        q = _get_acceleration(niter)
    else:
        q = [0.0] * niter
    
    # initialize algorithm
    if tol is None:
        compute_residual = True
    else:
        compute_residual = False
    PGD = PGDStep(step, AHA, AHy, D)
    
    # run algorithm
    for n in range(niter):
        output = PGD(input, q[n])
        
        # if required, compute residual and check if we reached convergence
        if compute_residual:
            resid = torch.linalg.norm(output - input).item() / step
            if resid < tol:
                break
        
        # update variable
        input = output.clone()
        
    # back to original device
    output = output.to(device)
        
    # cast back to numpy if requried
    if isnumpy:
        output = output.numpy(force=True)
        
    return output
        

class PGDStep(nn.Module):
    """
    Proximal Gradient Method step.
    
    This represents propagation through a single iteration of a 
    Proximal Gradient Descent algorithm; can be used to build
    unrolled architectures.

    Attributes
    ----------
    step : float
        Gradient step size; should be <= 1 / max(eig(AHA)).
    AHA : Callable
        Normal operator AHA = AH * A.
    Ahy : torch.Tensor
        Adjoint AH of measurement
        operator A applied to the measured data y.
    D : Callable
        Signal denoiser for plug-n-play restoration.
    trainable : bool, optional
        If ``True``, gradient update step is trainable, otherwise it is not.
        The default is ``False``.
    tol : float, optional
        Stopping condition.
        The default is ``None`` (run until niter).

    """
    def __init__(self, step, AHA, AHy, D, trainable=False):
        super().__init__()
        if trainable:
            self.step = nn.Parameter(step)
        else:
            self.step = step
            
        # assign
        self.AHA = AHA
        self.AHy = AHy
        self.D = D
        self.s = AHy.clone()
                
    def forward(self, input, q=0.0):
        # gradient step : zk = xk-1 - gamma * AH(A(xk-1) - y != FISTA (accelerated)
        z = input - self.step * (self.AHA(input) - self.AHy)
        
        # denoise: sk = D(zk)
        s = self.D(z)
        
        # update: xk = sk + [(qk-1 - 1) / qk] * (sk - sk-1)
        if q != 0.0:
            output = s +  q * (s - self.s)
            self.s = s.clone()
        else:
            output = s # q1...qn = 1.0 != ISTA (non-accelerated)
            
        return output
    
# %% local utils
def _get_acceleration(niter):
    t = []
    t_new = 1

    for n in range(niter):
        t_old = t_new
        t_new = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
        t.append((t_old - 1) / t_new)

    return t
        