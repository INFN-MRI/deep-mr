"""Sub-package containing MR Encoding Operator builder."""

__all__ = ["EncodingOp"]

from ... import linops as _linops
from .. import calib as _calib

def EncodingOp(data, mask=None, traj=None, dcf=None, shape=None, nsets=1, basis=None, device=None, cal_data=None, toeplitz=False):
    """
    Prepare MR encoding operator for Cartesian / Non-Cartesian imaging.    

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input k-space data of shape ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    mask : np.ndarray | torch.Tensor, optional
        Sampling mask for Cartesian imaging. 
        Expected shape is ``(ncontrasts, nviews, nsamples)``.
        The default is ``None``.
    traj : np.ndarray | torch.Tensor, optional
        K-space trajectory for Non Cartesian imaging. 
        Expected shape is ``(ncontrasts, nviews, nsamples, ndims)``.
        The default is ``None``.
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation. 
        Expected shape is ``(ncontrasts, nviews, nsamples)``.
        The default is ``None``.
    shape : Iterable[int], optional
        Cartesian grid size of shape ``(nz, ny, nx)``. 
        The default is ``None``.
    nsets : int, optional
        Number of coil sensitivity sets of maps. The default is ``1.
    basis : np.ndarray | torch.Tensor, optional
        Low rank subspace basis of shape ``(ncontrasts, ncoeffs)``. The default is ``None``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    cal_data : np.ndarray | torch.Tensor, optional
        Calibration dataset for coil sensitivity estimation. 
        The default is ``None`` (use center region of ``data``).
    toeplitz : bool, optional
        Use Toeplitz approach for normal equation. The default is ``False``.

    Returns
    -------
    E : deepmr.linops.Linop
        MR encoding operator (i.e., ksp = E(img)).
    EHE : deepmr.linops.NormalLinop
        MR normal operator (i.e., img_n = EHE(img_n-1)).

    """
    # parse number of coils
    ncoils = data.shape[-4]
    
    # get device
    if device is None:
        device = data.device
    
    if mask is not None and traj is not None:
        raise ValueError("Please provide either mask or traj, not both.")
    if mask is not None: # Cartesian
        # Fourier
        F = _linops.FFTOp(mask, basis, device)
        
        # Normal operator
        if toeplitz:
            FHF = _linops.FFTGramOp(mask, basis, device)
        else:
            FHF = F.H * F
                  
        # Sensititivy
        if ncoils == 1:
            return F, _linops.NormalLinop(FHF._ndim, A=FHF.A)
        else:
            if cal_data is not None:
                sensmap = _calib.espirit_cal(cal_data.to(device), nsets=nsets)
            else:
                sensmap = _calib.espirit_cal(data.to(device), nsets=nsets)
            C = _linops.CoilOp(2, sensmap)
            EHE = C.H * FHF * C
            return F * C, _linops.NormalLinop(EHE._ndim, A=EHE.A)
                
    if traj is not None:
        assert shape is not None, "Please provide shape for Non-Cartesian imaging."
        ndim = traj.shape[-1]
        
        # Fourier
        F = _linops.NUFFTOp(traj, shape[-ndim:], basis, dcf, device=device)
        
        # Normal operator
        if toeplitz:
            FHF = _linops.NUFFTGramOp(traj, shape[-ndim:], basis, dcf, device=device)
        else:
            FHF = F.H * F
        
        # Sensititivy
        if ncoils == 1:
            return F, _linops.NormalLinop(FHF._ndim, A=FHF.A)
        else:
            if cal_data is not None:
                sensmap = _calib.espirit_cal(cal_data.to(device), nsets=nsets, coord=traj, shape=shape, dcf=dcf)
            else:
                sensmap = _calib.espirit_cal(data.to(device), nsets=nsets, coord=traj, shape=shape, dcf=dcf)
            C = _linops.CoilOp(2, sensmap)
            EHE = C.H * FHF * C
            return F * C, _linops.NormalLinop(EHE._ndim, A=EHE.A)


