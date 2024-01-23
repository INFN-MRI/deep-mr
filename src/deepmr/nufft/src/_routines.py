# -*- coding: utf-8 -*-
"""
Main FFT/NUFFT/Toeplitz routines.

@author: Matteo Cencini
"""
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=redefined-builtin
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=protected-access
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments

import gc
from typing import List, Tuple, Dict, Union

import numpy as np
import torch

from torch import Tensor

import time

from nufftorch.src._subroutines import (
    Apodize,
    Crop,
    DeGrid,
    DeviceDispatch,
    FFT,
    Grid,
    IFFT,
    Toeplitz,
    ZeroPad,
)

from nufftorch.src._factory import NUFFTFactory, NonCartesianToeplitzFactory


def prepare_nufft(
    coord: Tensor,
    shape: Union[int, List[int], Tuple[int]],
    oversamp: Union[float, List[float], Tuple[float]] = 1.125,
    width: Union[int, List[int], Tuple[int]] = 3,
    basis: Union[None, Tensor] = None,
    device: Union[str, torch.device] = "cpu",
    threadsperblock: int = 512,
) -> Dict:
    """Precompute NUFFT coefficients."""
    return NUFFTFactory()(coord, shape, width, oversamp, basis, device, threadsperblock)


def nufft(image: Tensor, interpolator: Dict) -> Tensor:
    """Non-uniform Fast Fourier Transform."""
    # Unpack interpolator
    ndim = interpolator["ndim"]
    oversamp = interpolator["oversamp"]
    width = interpolator["width"]
    beta = interpolator["beta"]
    kernel_dict = interpolator["kernel_dict"]
    scale = interpolator["scale"]
    device_dict = interpolator["device_dict"]
    device = device_dict["device"]

    # Collect garbage
    gc.collect()

    # Copy input to avoid original data modification
    image = image.clone()

    # Offload to computational device
    dispatcher = DeviceDispatch(computation_device=device, data_device=image.device)
    image = dispatcher.dispatch(image)

    # Apodize
    Apodize(image.shape[-ndim:], oversamp, width, beta, device)(image)

    # Zero-pad
    image = ZeroPad(oversamp, image.shape[-ndim:])(image)

    # FFT
    kdata = FFT(image)(image, axes=range(-ndim, 0), norm=None)

    # Interpolate
    kdata = DeGrid(device_dict)(kdata, kernel_dict) / scale

    # Bring back to original device
    kdata, image = dispatcher.gather(kdata, image)

    # Collect garbage
    gc.collect()

    return kdata


def nufft_adjoint(kdata: Tensor, interpolator: Dict) -> Tensor:
    """Adjoint Non-uniform Fast Fourier Transform."""
    # Unpack interpolator
    ndim = interpolator["ndim"]
    oversamp = interpolator["oversamp"]
    shape = interpolator["shape"]
    width = interpolator["width"]
    beta = interpolator["beta"]
    kernel_dict = interpolator["kernel_dict"]
    scale = interpolator["scale"]
    device_dict = interpolator["device_dict"]
    device = device_dict["device"]

    # Collect garbage
    gc.collect()

    # Offload to computational device
    dispatcher = DeviceDispatch(computation_device=device, data_device=kdata.device)
    kdata = dispatcher.dispatch(kdata)

    # Gridding
    kdata = Grid(device_dict)(kdata, kernel_dict) / scale

    # IFFT
    image = IFFT(kdata)(kdata, axes=range(-ndim, 0), norm=None)

    # Crop
    image = Crop(shape[-ndim:])(image)

    # Apodize
    Apodize(shape[-ndim:], oversamp, width, beta, device)(image)

    # Bring back to original device
    image, kdata = dispatcher.gather(image, kdata)

    # Collect garbage
    gc.collect()

    return image * (oversamp**ndim)


def prepare_noncartesian_toeplitz(
    coord: Tensor,
    shape: Union[int, List[int], Tuple[int]],
    prep_oversamp: Union[float, List[float], Tuple[float]] = 1.125,
    comp_oversamp: Union[float, List[float], Tuple[float]] = 1.0,
    width: Union[int, List[int], Tuple[int]] = 5,
    basis: Union[Tensor, None] = None,
    device: Union[str, torch.device] = "cpu",
    threadsperblock: int = 512,
    dcf: Union[Tensor, None] = None,
) -> Dict:
    """Prepare Toeplitz matrix for Non-Cartesian sampling."""
    return NonCartesianToeplitzFactory()(
        coord,
        shape,
        prep_oversamp,
        comp_oversamp,
        width,
        basis,
        device,
        threadsperblock,
        dcf,
    )


def toeplitz_convolution(image: Tensor, toeplitz_dict: Dict) -> Tensor:
    """Perform in-place fast self-adjoint by convolution with spatio-temporal kernel matrix.

    Args:
        data_in (tensor): Input image.
        toeplitz (dict): Pre-computed spatio-temporal kernel.

    Returns
        data_out (tensor): Output image.

    """
    # Unpack input
    mtf = toeplitz_dict["mtf"]
    islowrank = toeplitz_dict["islowrank"]
    device_dict = toeplitz_dict["device_dict"]
    device = device_dict["device"]
    ndim = toeplitz_dict["ndim"]
    oversamp = toeplitz_dict["oversamp"]

    # Collect garbage
    gc.collect()

    # Offload to computational device
    dispatcher = DeviceDispatch(computation_device=device, data_device=image.device)
    image = dispatcher.dispatch(image)

    # Reshape for computation
    ishape = list(image.shape)
    image = image.reshape(ishape[0], np.prod(ishape[1:-ndim]), *ishape[-ndim:])

    # Zero-pad
    image = ZeroPad(oversamp, ishape[-ndim:])(image)

    # FFT
    kdata_in = FFT(image)(image, axes=range(-ndim, 0), norm=None, center=False)

    # Perform convolution
    if islowrank:
        os_shape = kdata_in.shape
        kdata_in = kdata_in.reshape(
            *kdata_in.shape[:2], int(np.prod(kdata_in.shape[-ndim:]))
        ).T.contiguous()
        kdata_out = torch.zeros(
            kdata_in.shape, dtype=kdata_in.dtype, device=kdata_in.device
        )
        Toeplitz(kdata_in.numel(), device_dict)(kdata_out, kdata_in, mtf)
        kdata_out = kdata_out.T.reshape(os_shape).contiguous()
    else:
        kdata_out = mtf * kdata_in

    # IFFT
    image = IFFT(kdata_out)(kdata_out, axes=range(-ndim, 0), norm=None, center=False)

    # Crop
    image = Crop(ishape[-ndim:])(image)

    # Reshape back to original shape
    image = image.reshape(ishape)

    # Bring back to original device
    image = dispatcher.gather(image)

    # Collect garbage
    gc.collect()

    return image
