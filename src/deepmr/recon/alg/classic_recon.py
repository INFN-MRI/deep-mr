"""Classical iterative reconstruction wrapper."""

__all__ = ["recon_lstsq"]

import copy

import numpy as np
import torch

import deepinv as dinv

from ... import optim as _optim
from ... import prox as _prox
from .. import calib as _calib
from . import linop as _linop

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def recon_lstsq(data, head, mask=None, niter=1, prior=None, prior_ths=0.01, prior_params=None, lamda=0.0, stepsize=None, basis=None, nsets=1, device=None, cal_data=None, toeplitz=True):
    """
    Classical MR reconstruction.    

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input k-space data of shape ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    head : deepmr.Header
        DeepMR acquisition header, containing ``traj``, ``shape`` and ``dcf``.  
    mask : np.ndarray | torch.Tensor, optional
        Sampling mask for Cartesian imaging. 
        Expected shape is ``(ncontrasts, nviews, nsamples)``.
        The default is ``None``.       
    niter : int, optional
        Number of recon iterations. If single iteration,
        perform simple zero-filled recon. The default is ``1``.
    prior : str | deepinv.optim.Prior, optional
        Prior for image regularization. If string, it must be one of the following:
        
        * ``"L1Wav"``: L1 Wavelet regularization.
        * ``"TV"``: Total Variation regularization.

        The default is ``None`` (no regularizer).
    prior_ths : float, optional
        Threshold for denoising in regularizer. The default is ``0.01``.
    prior_params : dict, optional
        Parameters for Prior initializations.
        See :func:`deepmr.prox`.
        The defaul it ``None`` (use each regularizer default parameters).
    lamda : float, optional
        Tikonhov regularization strength. If 0.0, do not apply
        Tikonhov regularization. The default is ``0.0``.
    stepsize : float, optional
        Iterations step size. If not provided, estimate from Encoding
        operator maximum eigenvalue. The default is ``None``.
    basis : np.ndarray | torch.Tensor, optional
        Low rank subspace basis of shape ``(ncontrasts, ncoeffs)``. The default is ``None``.
    nsets : int, optional
        Number of coil sensitivity sets of maps. The default is ``1.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    cal_data : np.ndarray | torch.Tensor, optional
        Calibration dataset for coil sensitivity estimation. 
        The default is ``None`` (use center region of ``data``).
    toeplitz : bool, optional
        Use Toeplitz approach for normal equation. The default is ``False``.

    Returns
    -------
    img np.ndarray | torch.Tensor
        Reconstructed image of shape:
            
        * 2D Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 2D Non Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 2D Non Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 3D Non Cartesian: ``(ncontrasts, nz, ny, nx).

    """
    if isinstance(data, np.ndarray):
        data = torch.as_tensor(data)
        isnumpy = True
    else:
        isnumpy = False
        
    if device is None:
        device = data.device
    data = data.to(device)
    
    if head.dcf is not None:
        head.dcf = head.dcf.to(device)
    
    # toggle off topelitz for non-iterative
    if niter == 1:
        toeplitz = False
        
    # get ndim
    if head.traj is not None:
        ndim = head.traj.shape[-1]
    else:
        ndim = 2 # assume 3D data already decoupled along readout
        
    # build encoding operator
    E, EHE = _linop.EncodingOp(data, mask, head.traj, head.dcf, head.shape, nsets, basis, device, cal_data, toeplitz)
    
    # perform zero-filled reconstruction
    img = E.H(head.dcf**0.5 * data[:, None, ...])
    
    # if non-iterative, just perform linear recon
    if niter == 1:
        output = img
        if isnumpy:
            output = output.numpy(force=True)
        return output
    
    # rescale
    img = _calib.intensity_scaling(img, ndim=ndim)
    
    # if no prior is specified, use CG recon
    if prior is None:
        output = EHE.solve(img, max_iter=niter, lamda=lamda)
        if isnumpy:
            output = output.numpy(force=True)
        return output
    
    # if a single prior is specified, use PDG
    if isinstance(prior, (list, tuple)) is False:
        
        # default prior params
        if prior_params is None:
            prior_params = {}
        
        # modify EHE
        if lamda != 0.0:
            img = img / lamda
            prior_ths = prior_ths / lamda
            tmp = copy.deepcopy(EHE)
            f = lambda x : tmp.A(x) + lamda * x
            EHE.A = f
            EHE.A_adjoint = f
        else:
            lamda = 1.0
        
        # compute spectral norm
        if stepsize is None:
            max_eig = EHE.maxeig(img, max_iter=1)
            if max_eig == 0.0:
                stepsize = 1.0
            else:
                stepsize = 1.0 / float(max_eig)
        
        # solver parameters
        params_algo = {"stepsize": stepsize, "g_param": prior_ths, "lambda": lamda}
        
        # select the data fidelity term
        data_fidelity = _optim.L2()
        
        # Get Wavelet Prior
        prior = _get_prior(prior, ndim, device, **prior_params)
        
        # instantiate the algorithm class to solve the IP problem.
        solver = dinv.optim.optim_builder(
            iteration="PGD",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=True,
            max_iter=niter,
            params_algo=params_algo,
        )
        
        output = solver(img, EHE) * lamda
        if isnumpy:
            output = output.numpy(force=True)
        return output
        

# %% local utils   
def _get_prior(ptype, ndim, device, **params):
    if isinstance(ptype, str):
        if ptype == "L1Wave":
            return _prox.WaveletPrior(ndim, device=device, **params)
        elif ptype == "TV":
            return _prox.TVPrior(ndim, device=device, **params)
        else:
            raise ValueError(f"Prior type = {ptype} not recognized; either specify 'L1Wave', 'TV' or 'deepinv.optim.Prior' object.")
    else:
        raise NotImplementedError("Direct prior object not implemented.")
