
__all__ = ["espirit_cal"]

import numpy as np
import torch

from ... import fft as _fft
from ... import _signal

from ... import linops as _linops
from ... import optim as _optim
from ... import prox as _prox

from . import acs as _acs

def nlinv(
    data, mask=None, coord=None, shape=None, nsets=1
):
    """
    Derives the NLINV/ENLIVE [1] operator.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Multi channel k-space data.
    coord : np.ndarray | torch.Tensor, optional
        K-space trajectory of ``shape = (ncontrasts, nviews, nsamples, ndim)``.
        The default is ``None`` (Cartesian acquisition).
    shape : Iterable[int] | optional
        Shape of the k-space after gridding. If not provided, estimate from
        input data (assumed on a Cartesian grid already).
        The default is ``None`` (Cartesian acquisition).
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

    * **2Dcart:** ``(nslices, ncoils, ..., ny, nx)``.
    * **2Dnoncart:** ``(nslices, ncoils, ..., nviews, nsamples)``.
    * **3Dcart:** ``(nx, ncoils, ..., nz, ny)``.
    * **3Dnoncart:** ``(ncoils, ..., nviews, nsamples)``.

    For multi-contrast acquisitions, calibration is obtained by averaging over
    contrast dimensions.

    The output sensitivity maps are assumed to have the following shape:

    * **2Dcart:** ``(nsets, nslices, ncoils, ny, nx)``.
    * **2Dnoncart:** ``(nsets, nslices, ncoils, ny, nx)``.
    * **3Dcart:** ``(nsets, nx, ncoils, nz, ny)``.
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
    if np.max(shape) >= 1024:
        cshape = list(np.asarray(shape, dtype=int) // 16)
    elif np.max(shape) >= 512:
        cshape = list(np.asarray(shape, dtype=int) // 8)
    elif np.max(shape) >= 256:
        cshape = list(np.asarray(shape, dtype=int) // 4)
    elif np.max(shape) >= 128:
        cshape = list(np.asarray(shape, dtype=int) // 2)
    else:
        cshape = [64] * ndim
    
    # get calibration region
    cal_data = _acs.find_acs(data, cshape, coord)
    cmask = _signal.resize(mask, ndim * [r])
    
    # calculate maps
    maps = _enlive(cal_data.clone(), cmask, cshape, coord, toeplitz)
    
    # select maps
    if nsets == 1:
        maps = maps[[0]]
    else:
        maps = maps[:nsets]

    # resample
    maps = _signal.resample(maps, shape)  # (nsets, ncoils, nz, ny, nx)

    # normalize
    maps_rss = _signal.rss(maps, axis=1, keepdim=True)
    maps = maps / maps_rss[[0]]

    # reformat
    if ndim == 2:  # Cartesian or 2D Non-Cartesian
        maps = maps.swapaxes(
            1, 2
        )  # (nsets, nslices, ncoils, ny, nx) / (nsets, nx, ncoils, nz, ny)

    # cast back to numpy if required
    if isnumpy:
        maps = maps.numpy(force=True)

    return maps, _signal.resize(cal_data, ndim * [r])


# %% local utils
def _enlive(data, mask, shape, coord, toeplitz):
    
    # build encoding operator
    F, FHF = _get_linop(data, mask, coord, shape, device, toeplitz)
    
    # scale data
    img = _fft.ifft(data, axes=range(-ndim, 0))
    for n in range(len(img.shape) - ndim):
        img = torch.linalg.norm(img, axis=0)

    # get scaling
    img = torch.nan_to_num(img, posinf=0.0, neginf=0.0, nan=0.0)
    scale = torch.quantile(abs(img.ravel()), 0.95)
    
    # scale data
    data /= scale

    
    
def _get_linop(data, mask, coord, shape, device, toeplitz):
    # get device
    if device is None:
        device = data.device

    if mask is not None and coord is not None:
        raise ValueError("Please provide either mask or traj, not both.")
        
    if mask is not None:  # Cartesian
        # Fourier
        F = _linops.FFTOp(mask, device=device)

        # Normal operator
        if toeplitz:
            FHF = _linops.FFTGramOp(mask, device=device)
        else:
            FHF = F.H * F
            
    if coord is not None: # Non Cartesian
        assert shape is not None, "Please provide shape for Non-Cartesian imaging."
        ndim = coord.shape[-1]

        # Fourier
        F = _linops.NUFFTOp(coord, shape[-ndim:], device=device)

        # Normal operator
        if toeplitz:
            FHF = _linops.NUFFTGramOp(coord, shape[-ndim:], device=device)
        else:
            FHF = F.H * F
            
    return F, FHF


def _irgn_fista(data, ndim, F, FHF, niter, D, step):
    
    # input shape is 
    # 2D Cart / NonCart: (nslices, nc, ny, nx) 
    # 3D Cart: (nx, nc, nz, ny) 
    # 3D NonCart : (nc, nz, ny, nx)
    # in general (..., nc, *mtx), with len(mtx) = ndim
    
    # shape of dx is (..., 1+nc, *mtx), with dx[..., 0, ...] = M0, dx[..., 1:, ...] = smap[0], smap[1], ... smap[nc-1]
    if ndim == 2:
        while len(data.shape) < 4:
            data = data[None, ...]
        data = data.swapaxes(0, 1)
        
    # build initial guess
    nc, nz, ny, nx = data.shape
    x0 = torch.zeros((nc+1, nz, ny, nx), dtype=data.dtype, device=data.device)
    x0[0, ...] = 1.0
    
    # initialize operator
    C = _linops.SenseOp(3, x0[1:])
    E = F * C
    EHE = C.H * FHF * C
    
    # initialize step
    dx = torch.cat(((E.H * data)[None, ...], F.H * data), axis=0)
    
    for n in range(niter):
        
        # compute update
        dx = _optim.pgd_solve(dx, step, EHE, D)
        
        # update variable
        
        # update operator
        
        
    
    
    
    
    
