"""B1+ and sensitivity maps generation routines."""

__all__ = ["sensmap", "b1field"]

import math
import numpy as np
import torch

def sensmap(shape, coil_cent=None, coil_width=1.5, n_rings=None):
    """
    Simulate birdcage coils.
    
    Adapted from SigPy [1].

    Parameters
    ----------
    shape : Iterable[int]
        Size of the image ``(ncoils, ny, nx)`` (2D) or ``(ncoils, nz, ny, nx)`` (3D) for the sensitivity coils.
    coil_cent : Iterable[int], optional
        Displacement of the coil center with respect to matrix center. 
        The default is ``(0, 0)`` / ``(0, 0, 0)``.
    coil_width : float, optional
        Parameter governing the width of the
        coil, multiplied by actual image dimension.
        The default is ``1.5``.
    n_rings : int, optional
        Number of rings for a cylindrical hardware set-up. 
        The default is ``ncoils // 4``.

    Returns
    -------
    smap : torch,Tensor
        Complex spatially varying sensitivity maps of shape ``(nmodes, ny, nx)`` (2D)
        or ``(nmodes, nz, ny, nx)`` (3D). If ``nmodes = 1``, the first dimension is squeezed.
        
    Example
    -------
        
    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    smap = _birdcage(shape, coil_cent, coil_width, n_rings)
    
    # normalize
    rss = sum(abs(smap) ** 2, 0) ** 0.5
    smap /= rss
    
    return smap

def b1field(shape, nmodes=1, b1range=(0.8, 1.2), coil_cent=None, ncoils=16, coil_width=1.5, n_rings=None):
    """
    Simulate birdcage coils.
    
    Adapted from SigPy [1].

    Parameters
    ----------
    shape : Iterable[int]
        Size of the image ``(ncoils, ny, nx)`` (2D) or 
        ``(ncoils, nz, ny, nx)`` (3D) for the sensitivity coils.
    nmodes : int, optional
        Number of B1+ modes. First mode is ``CP`` mode, second
        is ``gradient`` mode, and so on. The default is ``1``.
    b1range : Iterable[float]
        Range of B1+ magnitude. The default is ``(0.8, 1.2)``.
    coil_cent : Iterable[int], optional
        Displacement of the coil center with respect to matrix center. 
        The default is ``(0, 0)`` / ``(0, 0, 0)``.
    ncoils : int, optional
        Number of transmit coil elements. Standard coils have ``2`` channels
        operating in quadrature. To support multiple modes (i.e., PTX), increase this
        number. The default is ``2``.
    coil_width : float, optional
        Parameter governing the width of the
        coil, multiplied by actual image dimension.
        The default is ``1.5``.
    n_rings : int, optional
        Number of rings for a cylindrical hardware set-up. 
        The default is ``ncoils // 4``.

    Returns
    -------
    smap : torch,Tensor
        Complex spatially varying sensitivity maps of shape ``(ncoils, ny, nx)`` (2D)
        or ``(ncoils, nz, ny, nx)`` (3D).
        
    Example
    -------
        
    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    # check we can do quadrature
    assert ncoils >= 2, f"We support circular polarization only - found {ncoils} transmit elements."
    assert ncoils >= nmodes, f"Need ncoils (={ncoils}) to be >= nmodes (={nmodes})."
    
    # generate coils
    smap = _birdcage([ncoils] + list(shape), coil_cent, coil_width, n_rings).numpy()
    
    # normalize
    rss = sum(abs(smap) ** 2, 0) ** 0.5
    smap /= rss

    # combine
    dalpha = math.pi / ncoils
    alpha = np.arange(ncoils) * dalpha
    mode = np.arange(nmodes) + 1
    phafu = np.exp(1j * mode[:, None] * alpha[None, :]) # (nmodes, nchannels)
    
    # get modes
    smap = smap.T # (nc, ...) -> (..., nc)
    smap = [(smap * phafu[n]).sum(axis=-1) for n in range(nmodes)]
    smap = np.stack(smap, axis=-1) # (..., nmodes)
    smap = smap.T # (..., nmodes) -> (nmodes, ...)
        
    # rescale
    smap = smap - abs(smap).min() # (min, max) -> (0, max - min)
    smap = smap / abs(smap).max() # (0, max - min) -> (0, 1)
    smap = smap * (b1range[1] - b1range[0]) + b1range[0] # (0, 1) -> (b1range[0], b1range[1])
    
    # convert to tensor
    if nmodes == 1:
        smap = torch.as_tensor(abs(smap[0]), dtype=torch.float32)
    else:
        smap = torch.as_tensor(smap, dtype=torch.complex64)
    
    return smap

    
def _birdcage(shape, coil_cent=None, coil_width=1.5, n_rings=None):
    # default
    if coil_cent is None:
        coil_cent = [ax / 2.0 for ax in shape[1:]]
        
    if n_rings is None:
        n_rings = np.max((shape[0] // 4, 1))
        
    if len(shape) == 3:
        nc, ny, nx = shape
        c, y, x = np.mgrid[:nc, :ny, :nx]

        coilx = coil_width * np.cos(c * (2 * math.pi / nc))
        coily = coil_width * np.sin(c * (2 * math.pi / nc))
        
        coil_phs = -c * (2 * np.pi / nc)

        x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
        y_co = (y - ny / 2.0) / (ny / 2.0) - coily
        
        rr = np.sqrt(x_co**2 + y_co**2)

    elif len(shape) == 4:
        nc, nz, ny, nx = shape
        c, z, y, x = np.mgrid[:nc, :nz, :ny, :nx]

        coilx = coil_width * np.cos(c * (2 * math.pi / n_rings))
        coily = coil_width * np.sin(c * (2 * math.pi / n_rings))
        coilz = np.floor(c / n_rings) - 0.5 * (np.ceil(nc / n_rings) - 1)
        
        coil_phs = -(c + np.floor(c / n_rings)) * (2 * math.pi / n_rings)

        x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
        y_co = (y - ny / 2.0) / (ny / 2.0) - coily
        z_co = (z - nz / 2.0) / (nz / 2.0) - coilz
        
        rr = (x_co**2 + y_co**2 + z_co**2) ** 0.5
    else:
        raise ValueError("Can only generate shape with length 3 or 4")
        
    # build coils
    phi = np.arctan2(x_co, -y_co) + coil_phs
    smap = (1.0 / rr) * np.exp(1j * phi)
        
    return torch.as_tensor(smap, dtype=torch.complex64)
