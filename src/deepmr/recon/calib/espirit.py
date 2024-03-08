"""Pytorch ESPIRIT implementation. Adapted for convenience from https://github.com/mikgroup/espirit-python/tree/master"""

__all__ = ["espirit_cal"]

import numpy as np
import torch

# from ... import fft as _fft
from ... import _signal

from . import acs as _acs

def espirit_cal(data, coord=None, dcf=None, shape=None, k=6, r=24, t=0.02, c=0.95, nsets=1):
    """
    Derives the ESPIRiT [1] operator.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Multi channel k-space data.
    coord : np.ndarray | torch.Tensor, optional
        K-space trajectory of ``shape = (ncontrasts, nviews, nsamples, ndim)``. 
        The default is ``None`` (Cartesian acquisition).
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation of ``shape = (ncontrasts, nviews, nsamples)``. 
        The default is ``None`` (no compensation).
    shape : Iterable[int] | optional
        Shape of the k-space after gridding. If not provided, estimate from
        input data (assumed on a Cartesian grid already).
        The default is ``None`` (Cartesian acquisition).
    k : int, optional
        k-space kernel size. The default is ``6``.
    r : int, optional
        Calibration region size. The default is ``24``.
    t : float, optional
        Rank of the auto-calibration matrix (A). 
        The default is ``0.02``.
    c : float, optional
        Crop threshold that determines eigenvalues "=1". 
        The defaults is ``0.95``.
    nsets : int, optional
        Number of set of maps to be returned.
        The default is ``1`` (conventional SENSE recon).
        
    Returns
    -------
    maps : np.ndarray | torch.Tensor
        Output coil sensitivity maps.
        
    Notes
    -----
    The input k-space ``data`` tensor is assumed to have the following shape:

    * **2Dcart:** ``(nslices, ncoils, ncontrasts, ny, nx)``.
    * **2Dnoncart:** ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    * **3Dcart:** ``(nx, ncoils, ncontrasts, nz, ny)``.
    * **3Dnoncart:** ``(ncoils, ncontrasts, nviews, nsamples)``.
    
    For multi-contrast acquisitions, calibration is obtained by averaging over
    contrast dimensions.
    
    The output sensitivity maps are assumed to have the following shape:
        
    * **2Dcart:** ``(nslices, nsets, ncoils, ny, nx)``.
    * **2Dnoncart:** ``(nslices, nsets, ncoils, ny, nx)``.
    * **3Dcart:** ``(nx, nsets, ncoils, nz, ny)``.
    * **3Dnoncart:** ``(nsets, ncoils, nz, ny, nx)``.
        
    References
    ----------
    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. 
           ESPIRiT--an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. 
           Magn Reson Med. 2014 Mar;71(3):990-1001. doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.

    """
    if isinstance(data, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False
        
    while len(data.shape) < 5:
        data = data[None, ...]
        
    # keep shape
    if coord is not None:
        ndim = coord.shape[-1]
        if np.isscalar(shape):
            shape = ndim * [shape]
        else:
            shape = list(shape)[-ndim:]
        shape = [int(s) for s in shape]
    else:
        ndim = 2
        shape = list(data.shape[-2:])
        
    # extract calibration region
    cal_data = _acs.find_acs(data, r, coord, dcf)
    
    # calculate maps
    maps = _espirit(cal_data, k, r, t, c)
    
    # select maps
    if nsets == 1:
        maps = maps[[0]]
    else:
        maps = maps[:nsets]
                    
    # resample 
    maps = _signal.resample(maps, shape) # (nsets, ncoils, nz, ny, nx)
    
    # normalize
    maps_rss = _signal.rss(maps, axis=1, keepdim=True)
    maps = maps / maps_rss
    
    # reformat
    if ndim == 2: # Cartesian or 2D Non-Cartesian
        maps = maps.swapaxes(1, 2) # (nsets, nslices, ncoils, ny, nx)
        maps = maps.swapaxes(0, 1) # (nslices, nsets, ncoils, ny, nx)
                
    # cast back to numpy if required
    if isnumpy:
        maps = maps.numpy(force=True)
    
    return maps
        
# %% local utils
fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

def _espirit(X, k, r, t, c):
    """Adapted for convenience from https://github.com/mikgroup/espirit-python/tree/master"""
    
    # MC
    device = X.device
    X = X.clone().numpy(force=True).T
    # MC

    sx = np.shape(X)[0]
    sy = np.shape(X)[1]
    sz = np.shape(X)[2]
    nc = np.shape(X)[3]

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
    szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)

    # Extract calibration region.    
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].astype(np.complex64)

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = np.zeros([(r-k+1)**p, k**p * nc]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
      for ydx in range(max(1, C.shape[1] - k + 1)):
        for zdx in range(max(1, C.shape[2] - k + 1)):
          # numpy handles when the indices are too big
          block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].astype(np.complex64) 
          A[idx, :] = block.flatten()
          idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    n = np.sum(S >= t * S[0])
    V = V[:, 0:n]

    kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
    kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
    kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
            kerimgs[:,:,:,jdx,idx] = fft(ker, axes) * np.sqrt(sx * sy * sz) / np.sqrt(k**p)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):

                Gq = kerimgs[idx,jdx,kdx,:,:]

                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if (s[ldx]**2 > c):
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]
                        
    # MC
    maps = torch.as_tensor(maps.T.copy(), device=device, dtype=torch.complex64)
    # MC

    return maps