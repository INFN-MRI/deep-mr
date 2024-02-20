"""B1+ and sensitivity maps generation routines."""

__all__ = ["sensmap", "b1field"]

import math
import numpy as np
import torch

def sensmap(shape ,coil_width=2.0,  shift=None, dphi=0.0, nrings=None):
    """
    Simulate birdcage coils.
    
    Adapted from SigPy [1].

    Parameters
    ----------
    shape : Iterable[int]
        Size of the image ``(ncoils, ny, nx)`` (2D) or ``(ncoils, nz, ny, nx)`` (3D) for the sensitivity coils.
    shift : Iterable[int], optional
        Displacement of the coil center with respect to matrix center. 
        The default is ``(0, 0)`` / ``(0, 0, 0)``.
    dphi : float
        Bulk coil angle in ``[deg]``.
        The default is ``0.0°``.
    coil_width : float, optional
        Width of the coil, with respect to image dimension.
        The default is ``2.0``.
    nrings : int, optional
        Number of rings for a cylindrical hardware set-up. 
        The default is ``ncoils // 4``.

    Returns
    -------
    smap : torch,Tensor
        Complex spatially varying sensitivity maps of shape ``(nmodes, ny, nx)`` (2D)
        or ``(nmodes, nz, ny, nx)`` (3D). If ``nmodes = 1``, the first dimension is squeezed.
        
    Example
    -------
    >>> import deepmr
    
    We can generate a set of ``nchannels=8`` 2D sensitivity maps of shape ``(ny=128, nx=128)`` by:
        
    >>> smap = deepmr.sensmap((8, 128, 128))
    
    Coil center and rotation can be modified by ``shift`` and ``dphi`` arguments:
        
    >>> smap = deepmr.sensmap((8, 128, 128), shift=(-3, 5), dphi=30.0) # center shifted by (dy, dx) = (-3, 5) pixels and rotated by 30.0 degrees.
    
    Similarly, ``nchannels=8`` 3D sensitivity maps can be generated as:
        
    >>> smap = deepmr.sensmap((8, 128, 128, 128))
    
    Beware that this will require more memory.
        
    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    smap = _birdcage(shape, coil_width, nrings, shift, np.deg2rad(dphi))
    
    # normalize
    rss = sum(abs(smap) ** 2, 0) ** 0.5
    smap /= rss
    
    return smap


def b1field(shape, nmodes=1, b1range=(0.5, 2.0), shift=None, dphi=0.0, coil_width=1.1, ncoils=8, nrings=None):
    """
    Simulate inhomogeneous B1+ fields.
    
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
        Range of B1+ magnitude. The default is ``(0.5, 2.0)``.
    shift : Iterable[int], optional
        Displacement of the coil center with respect to matrix center. 
        The default is ``(0, 0)`` / ``(0, 0, 0)``.
    dphi : float
        Bulk coil angle in ``[deg]``.
        The default is ``0.0°``.
    coil_width : float, optional
        Width of the coil, with respect to image dimension.
        The default is ``1.1``.
    ncoils : int, optional
        Number of transmit coil elements. Standard coils have ``2`` channels
        operating in quadrature. To support multiple modes (i.e., PTX), increase this
        number. The default is ``8``.
    nrings : int, optional
        Number of rings for a cylindrical hardware set-up. 
        The default is ``ncoils // 4``.

    Returns
    -------
    smap : torch,Tensor
        Complex spatially varying b1+ maps of shape ``(nmodes, ny, nx)`` (2D)
        or ``(nmodes, nz, ny, nx)`` (3D). Magnitude of the map represents
        the relative flip angle scaling (wrt to the nominal).
        
    Example
    -------
    >>> import deepmr
    
    We can generate a 2D B1+ field map of shape ``(ny=128, nx=128)`` by:
        
    >>> b1map = deepmr.b1field((128, 128))
    
    Field center and rotation can be modified by ``shift`` and ``dphi`` arguments:
        
    >>> b1map = deepmr.b1field((8, 128, 128), shift=(-3, 5), dphi=30.0) # center shifted by (dy, dx) = (-3, 5) pixels and rotated by 30.0 degrees.
    
    B1+ values range and steepness of variation can be specified using ``b1range`` and ``coil_width`` arguments:
        
    >>> # transmit coil is 4 times bigger than FOV (e.g., body coil) and 
    >>> # B1+ scalings are between (0.8, 1.2) the nominal flip angle (e.g., 3T scanner)
    >>> b1map3T = deepmr.b1field((128, 128), b1range=(0.8, 1.2), coil_width=4.0) 
    >>>
    >>> # transmit coil is 1.1 times bigger than FOV (e.g., head coil) and 
    >>> # B1+ scalings are between (0.5, 2.0) the nominal flip angle (e.g., 7T scanner)
    >>> b1map3T = deepmr.b1field((128, 128), b1range=(0.5, 2.0), coil_width=1.1) 
    
    Multiple orthogonal modes can be simulated by ``nmodes`` argument.
    For example, ``CP`` mode and ``gradient mode`` can be obtained as:
        
    >>> b1map = deepmr.b1field((128, 128), nmodes=2) # b1map[0] is CP, b1map[1] is gradient mode.
    
    Three dimensional sensitivity maps of shape ``(nz, ny, nx)`` can be obtained as:
        
    >>> b1map = deepmr.b1field((128, 128, 128))
    
    Beware that this will require more memory.
        
    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    # check we can do quadrature
    assert ncoils >= 2, f"We support circular polarization only - found {ncoils} transmit elements."
    assert ncoils >= nmodes, f"Need ncoils (={ncoils}) to be >= nmodes (={nmodes})."
    
    # generate coils
    smap = _birdcage([ncoils] + list(shape), coil_width, nrings, shift, np.deg2rad(dphi)).numpy()
    
    # normalize
    rss = sum(abs(smap) ** 2, 0) ** 0.5
    smap /= rss

    # # combine
    dalpha = 2 * math.pi / ncoils
    alpha = np.arange(ncoils) * dalpha
    mode = np.arange(nmodes)
    phafu = np.exp(1j * mode[:, None] * alpha[None, :]) # (nmodes, nchannels)
    
    # # get modes
    smap = smap.T # (nc, ...) -> (..., nc)
    smap = [(abs(smap) * phafu[n]).sum(axis=-1) for n in range(nmodes)]
    smap = np.stack(smap, axis=-1) # (..., nmodes)
    smap = smap.T # (..., nmodes) -> (nmodes, ...)
        
    # # rescale
    phase = smap / abs(smap)
    smap = abs(smap)
    smap = smap - smap.min() # (min, max) -> (0, max - min)
    smap = smap / smap.max() # (0, max - min) -> (0, 1)
    smap = smap * (b1range[1] - b1range[0]) + b1range[0] # (0, 1) -> (b1range[0], b1range[1])
    smap = smap * phase
    
    # convert to tensor
    if nmodes == 1:
        smap = torch.as_tensor(abs(smap[0]), dtype=torch.float32)
    else:
        smap = torch.as_tensor(smap, dtype=torch.complex64)
    
    return smap

    
def _birdcage(shape, coil_width=1.5, nrings=None, shift=None, dphi=0.0):
    
    # default
    if shift is None:
        shift = [0.0 for ax in range(len(shape) -1 )]    
    if nrings is None:
        nrings = np.max((shape[0] // 4, 1))
        
    # coil width and radius
    c_width = coil_width * min(shape[-2:])
    c_rad = 0.5 * min(shape[-2:])
    
    if len(shape) == 3:
        nc, ny, nx = shape
        phi = np.arange(nc) * (2 * math.pi / nc) + dphi
        y, x = np.mgrid[:ny, :nx]
        
        x0 = c_rad * np.cos(phi) + shape[-1] / 2.0 + shift[-1]
        y0 = c_rad * np.sin(phi) + shape[-2] / 2.0 + shift[-2]
        
        x_co = x[None, ...] - x0[:, None, None]
        y_co = y[None, ...] - y0[:, None, None]
        
        # coil magnitude
        rr = np.sqrt(x_co**2 + y_co**2) / (2 * c_width)
                
        # coil phase
        phi = np.arctan2(x_co, -y_co) - phi[:, None, None]

    elif len(shape) == 4:
        nc, nz, ny, nx = shape
        phi = np.arange(nc) * (2 * math.pi / (nc + nrings)) + dphi
        z, y, x = np.mgrid[:nz, :ny, :nx]

        x0 = c_rad * np.cos(phi) + shape[-1] / 2.0 + shift[-1]
        y0 = c_rad * np.sin(phi) + shape[-2] / 2.0 + shift[-2]
        z0 = np.floor(np.arange(nc) / nrings) - 0.5 * (np.ceil(np.arange(nc) / nrings) - 1) + shape[-3] / 2.0 + shift[-3]
        
        x_co = x[None, ...] - x0[:, None, None, None]
        y_co = y[None, ...] - y0[:, None, None, None]
        z_co = z[None, ...] - z0[:, None, None, None]
        
        # coil magnitude
        rr = np.sqrt(x_co**2 + y_co**2 + z_co**2) / (2 * c_width)
        
        # coil phase
        phi = np.arctan2(x_co, -y_co) - phi[:, None, None, None]
    else:
        raise ValueError("Can only generate shape with length 3 or 4")
        
    # build coils
    rr[rr == 0.0] = 1.0
    smap = (1.0 / rr) * np.exp(1j * phi)
        
    return torch.as_tensor(smap, dtype=torch.complex64)
