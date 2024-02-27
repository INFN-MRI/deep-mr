"""Two- and three-dimensional Cartesian sampling."""

__all__ = ["cartesian2D", "cartesian3D"]

import numpy as np
import torch

# this is for stupid Sphinx
try:
    from ... import _design
except Exception:
    pass

from ..._types import Header


def cartesian2D(shape, accel=(1, 1), acs_shape=None):
    r"""
    Design a 2D (+t) cartesian encoding scheme.
    
    This function only support simple regular undersampling along phase
    encoding direction, with Parallel Imaging and Partial Fourier acceleration
    and rectangular FOV. For multi-echo acquisitions, sampling pattern is assumed
    constant along the echo train.
    
    Parameters
    ----------
    shape : Iterable[int]
        Matrix shape ``(x, y, echoes=1)``.
    accel : Iterable[int], optional
        Acceleration ``(Ry, Pf)``. 
        Ranges from ``(1, 1)`` (fully sampled) to ``(ny, 0.75)``.
        The default is ``(1, 1)``.
    acs_shape : int, optional
        Matrix size for calibration regions ``ACSy``.
        The default is ``None``.

    Returns
    -------
    mask : torch.Tensor
        Binary mask indicating the sampled k-space locations of shape ``(nz, ny)``.
    head : Header
        Acquisition header corresponding to the generated sampling pattern.

    Example
    -------
    >>> import deepmr

    We can create a 2D Cartesian sampling mask of ``(128, 128)`` pixels with Parallel Imaging factor ``Ry = 2`` by:

    >>> mask, head = deepmr.cartesian2D(128, accel=2)

    Partial Fourier acceleration can be enabled by passing a ``tuple`` as the ``accel`` argument:

    >>> mask, head = deepmr.cartesian2D(128, accel=(1, 0.8))

    A rectangular matrix can be specified by passing a ``tuple`` as the ``shape`` argument:

    >>> mask, head = deepmr.cartesian2D((128, 96), accel=2)
    >>> mask.shape
    torch.Size([96, 128])

    Autocalibration region width can be specified via the ``acs_shape`` argument:

    >>> mask, head = deepmr.cartesian2D(128, accel=2, acs_shape=32)

    The generated mask will have an inner ``(32, 128)`` fully sampled k-space stripe
    for coil sensitivity estimation.

    Multiple echoes with the same sampling (e.g., for QSM and T2* mapping) can be obtained by providing
    a 3-element tuple of ints as the ``shape`` argument:

    >>> mask, head = deepmr.cartesian2D((128, 128, 8), accel=2)
    >>> head.TE.shape
    torch.Size([8])

    corresponding to a 8-echoes undersampled k-spaces.

    Notes
    -----
    The returned ``head`` (:func:`deepmr.Header`) is a structure with the following fields:

    * shape (torch.Tensor):
        This is the expected image size of shape ``(nz, ny, nx)``.
    * t (torch.Tensor):
        This is the readout sampling time ``(0, t_read)`` in ``ms``.
        with shape ``(nx,)``. K-space raster time of ``1 us`` is assumed.
    * TE (torch.Tensor):
        This is the Echo Times array. Assumes a k-space raster time of ``1 us`` 
        and minimal echo spacing.

    """
    # expand shape if needed
    if np.isscalar(shape):
        shape = [shape, shape]
    else:
        shape = list(shape)

    while len(shape) < 3:
        shape = shape + [1]
    
    shape = shape[:2] + [1] + [shape[-1]]

    # assume 1mm iso
    fov = [float(shape[0]), float(shape[1])]
    
    # get nechoes
    nechoes = shape[-1]
    shape[-1] = 1

    # design mask
    tmp, _ = _design.cartesian2D(fov, shape[:2], accel, acs_shape=acs_shape)
    
    # get shape
    shape = shape[::-1]

    # get time
    t = tmp["t"]

    # calculate TE
    min_te = float(tmp["te"][0])
    TE = np.arange(nechoes, dtype=np.float32) * t[-1] + min_te

    # get indexes
    head = Header(shape, t=t, TE=TE)
    head.torch()
    
    # build mask
    mask = tmp["mask"]
    mask = np.repeat(mask, shape[-1], axis=-1)
    mask = torch.as_tensor(mask, dtype=int)
    mask = mask[0, 0]
    
    return mask, head


def cartesian3D(shape, accel_type="PI", accel=(1, 1, 1), shift=0, acs_shape=None):
    r"""
    Design a 3D (+t) cartesian encoding scheme.
    
    This function regular undersampling along both phase encoding directions, 
    with Parallel Imaging (including CAIPIRINHA shift), Partial Fourier acceleration
    and rectangular FOV. In addition, variable density Poisson disk sampling for Compressed Sensing 
    is supported. In the former case, sampling pattern is assumed constant for each contrast
    in multi-contrast acquisitions; in the latter, sampling pattern is unique for each contrast.
    
    For multi-echo acquisitions, sampling pattern is assumed constant along the echo train for
    both Parallel Imaging and Poisson Disk sampling.

    Parameters
    ----------
    shape : Iterable[int]
        Matrix shape ``(y, z, contrast=1, echoes=1)``.
    accel_type : str, optional
        Acceleration type. Can be either ``PI`` (Parallel Imaging)
        or ``CS`` (Compressed Sensing). In the former case, undersampling
        is regular and equal for each contrast. In the latter, build unique variable density Poisson-disk
        sampling for each contrast. The default is ``PI``.
    accel : Iterable[int], optional
        Acceleration factor. For ``accel_type = PI``, it is defined as ``(Ry, Rz, Pf)``, 
        ranging from ``(1, 1, 1)`` (fully sampled) to ``(ny, nz, 0.75)``. For ``accel_type = CS``,
        ranges from ``1`` (fully sampled) to ``ny * nz``.
        The default is ``(1, 1, 1)``.
    shift : int, optional
        CAIPIRINHA shift between ``ky`` and ``kz``.
        The default is ``0`` (standard Parallel Imaging).
    acs_shape : int, optional
        Matrix size for calibration regions ``ACSy``.
        The default is ``None``.

    Returns
    -------
    mask : torch.Tensor
        Binary mask indicating the sampled k-space locations of shape ``(ncontrasts, nz, ny)``.
    head : Header
        Acquisition header corresponding to the generated sampling pattern.

    Example
    -------
    >>> import deepmr

    We can create a 3D Cartesian sampling mask of ``(128, 128)`` pixels with Parallel Imaging factor ``(Ry, Rz) = (2, 2)`` by:

    >>> mask, head = deepmr.cartesian3D(128, accel=(2, 2))
    
    The undersampling along ``ky`` and ``kz`` can be shifted as in a CAIPIRINHA sampling by specifying the ``shift`` argument:
        
    >>> mask, head = deepmr.cartesian3D(128, accel=(1, 3), shift=2)

    Partial Fourier acceleration can be enabled by passing a 3-element ``tuple`` as the ``accel`` argument:

    >>> mask, head = deepmr.cartesian3D(128, accel=(2, 2, 0.8))
    
    Instead of regular undersampling, variable density Poisson disk sampling can be obtained by passing ``accel_type = CS``:
        
    >>> mask, head = deepmr.cartesian3D(128, accel_type="CS", accel=4) # 4 is the overall acceleration factor.

    A rectangular matrix can be specified by passing a ``tuple`` as the ``shape`` argument:

    >>> mask, head = deepmr.cartesian3D((128, 96), accel=(2, 2))
    >>> mask.shape
    torch.Size([96, 128])

    Autocalibration region width can be specified via the ``acs_shape`` argument:

    >>> mask, head = deepmr.cartesian2D(128, accel=2, acs_shape=32)

    The generated mask will have an inner ``(32, 32)`` fully sampled ``(kz, ky)`` k-space square
    for coil sensitivity estimation.

    Multiple contrasts with different sampling (e.g., for T1-T2 Shuffling) can be achieved by providing
    a 3-element tuple of ints as the ``shape`` argument:

    >>> mask, head = deepmr.cartesian3D((128, 128, 96), accel_type="CS", accel=4)
    >>> mask.shape
    torch.Size([96, 128, 128])

    corresponding to 96 different contrasts, each sampled with a different undersampled ``(kz, ky)`` pattern.
    Similarly, multiple echoes (with fixed sampling) can be specified using a 4-element tuple as:

    >>> mask, head = deepmr.cartesian3D((128, 128, 1, 8), accel=(2, 2))
    >>> mask.shape
    torch.Size([128, 128])
    >>> head.TE.shape
    torch.Size([8])

    corresponding to a 8-echoes undersampled k-spaces.

    Notes
    -----
    The returned ``head`` (:func:`deepmr.Header`) is a structure with the following fields:

    * shape (torch.Tensor):
        This is the expected image size of shape ``(nz, ny, nx)``. Matrix is assumed squared (i.e., ``nx = ny``).
    * t (torch.Tensor):
        This is the readout sampling time ``(0, t_read)`` in ``ms``.
        with shape ``(nx,)``. K-space raster time of ``1 us`` is assumed.
    * TE (torch.Tensor):
        This is the Echo Times array. Assumes a k-space raster time of ``1 us`` 
        and minimal echo spacing.

    """
    # expand shape if needed
    if np.isscalar(shape):
        shape = [shape, shape]
    else:
        shape = list(shape)
        
    while len(shape) < 4:
        shape = shape + [1]
        
    # assume 1mm iso
    fov = [float(shape[0]), float(shape[0]), float(shape[1])]
    
    # add x 
    shape = [shape[0]] + shape
    
    # get ncontrasts
    ncontrasts = shape[-2]
    shape[-2] = 1
    
    # get nechoes
    nechoes = shape[-1]
    shape[-1] = 1
    
    # fix acs_shape
    if acs_shape is not None:
        acs_shape = list(acs_shape)
    
    # design mask
    if accel_type == "PI":
        tmp, _ = _design.cartesian3D(fov, shape, accel, accel_type=accel_type, shift=shift, acs_shape=acs_shape)
    elif accel_type == "CS":
        if accel == 1:
            tmp, _ = _design.cartesian3D(fov, shape, accel_type="PI")
        else:
            tmp, _ = _design.cartesian3D(fov, shape, accel, accel_type=accel_type, acs_shape=acs_shape)
        if ncontrasts > 1:
            mask = np.zeros([ncontrasts-1] + list(tmp["mask"].shape[1:]), dtype=tmp["mask"].dtype)
            mask = np.concatenate((tmp["mask"], mask), axis=0)
            idx = np.random.rand(*mask.shape).argsort(axis=0)
            mask = np.take_along_axis(mask, idx, axis=0)
            tmp["mask"] = mask
    else:
        raise ValueError(f"accel_type = {accel_type} not recognized; must be either 'PI' or 'CS'.")
        
    # get shape
    shape = shape[::-1]

    # get time
    t = tmp["t"]

    # calculate TE
    min_te = float(tmp["te"][0])
    TE = np.arange(nechoes, dtype=np.float32) * t[-1] + min_te

    # get indexes
    head = Header(shape, t=t, TE=TE)
    head.torch()
    
    # build mask
    mask = tmp["mask"]
    mask = mask[..., 0]
    mask = torch.as_tensor(mask, dtype=int)
    
    return mask, head
