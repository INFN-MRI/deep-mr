# -*- coding: utf-8 -*-
"""
Functional wrappers for Non-uniform Fourier Transform routines.


Adapted from SigPy [1]. Compared to [1], we use a reduced memory
radix tensor decomposition of pre-computed interpolators, to speed-up
computation with respect to precomputation-free NUFFT [2,3] while
mantaining a reasonably light memory footprint in high-dimensional
problems (non-separable 3D+t data). We include interpolation
routines with embedded low-rank subspace projection operators for
model-based MRI reconstructions.

[1]: Ong, F., and M. Lustig. "SigPy: a python package for high performance iterative reconstruction."
         Proceedings of the ISMRM 27th Annual Meeting, Montreal, Quebec, Canada. Vol. 4819. 2019.
[2]: Lin, Jyh-Miin. "Python non-uniform fast Fourier transform (PyNUFFT):
     An accelerated non-Cartesian MRI package on a heterogeneous platform (CPU/GPU)."
     Journal of Imaging 4.3 (2018): 51.
[3]: Lin, Jyh-Miin, et al.
     "Memory reduced non-Cartesian MRI encoding using the mixed-radix tensor product on CPU and GPU."
     arXiv preprint arXiv:1903.08365 (2019).
[4]: McGivney DF, Pierre E, Ma D, et al.
     SVD compression for magnetic resonance fingerprinting in the time domain.
     IEEE Trans Med Imaging. 2014;33(12):2311-2322. doi:10.1109/TMI.2014.2337321

"""
# pylint: disable=no-member
# pylint: disable=protected-access
# pylint: disable=too-many-arguments

from typing import List, Tuple, Union

import torch
from torch import Tensor

from nufftorch.src import _routines


def nufft(image: Tensor,
          coord: Tensor,
          oversamp: Union[float, List[float], Tuple[float]] = 2.0,
          width: Union[int, List[int], Tuple[int]] = 4,
          basis: Union[None, Tensor] = None,
          device: Union[str, torch.device] = 'cpu',
          threadsperblock: int = 512) -> Tensor:
    """Non-uniform Fast Fourier Transform.
    
    Args:
        image (tensor): Input data in image space of shape [n, ..., nz, ny, nx],
                        where n can be number of frames or low-rank subspace
                        coefficients and ... is a set of batches dimensions
                        (coil, te, ...).
        coord (tensor): Coordinate array of shape [nframes, ..., ndim]
        oversamp (float): Grid oversampling factor.
        width (int or tuple of int): Interpolation kernel full-width.
        basis (tensor): Low-rank temporal subspace basis.

    Returns:
        kdata (tensor): Fourier domain data of shape [nframes, ..., coord_shape],
                        where  ... is a set f batches dimensions
                        (coil, te, ...) and coord_shape must match
                        coord.shape[:-1] used in prepare_nufft.

    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.
    """
    # Get image dimension
    ndim = coord.shape[-1]
    shape = image.shape[-ndim:]

    # Prepare interpolator object
    interpolator = _routines.prepare_nufft(
        coord, shape, oversamp, width, basis, device, threadsperblock)

    # Calculate k-space data
    kdata = _routines.nufft(image, interpolator)

    return kdata


def nufft_adjoint(kdata: Tensor,
                  coord: Tensor,
                  shape: Union[int, List[int], Tuple[int]],
                  oversamp: Union[float, List[float], Tuple[float]] = 2.0,
                  width: Union[int, List[int], Tuple[int]] = 4,
                  basis: Union[None, Tensor] = None,
                  device: Union[str, torch.device] = 'cpu',
                  threadsperblock: int = 512) -> Tensor:
    """Adjoint Non-uniform Fast Fourier Transform.
    
    Args:
        kdata (tensor): Input data in  Fourier space of shape [nframes, ..., coord_shape],
                        where  ... is a set f batches dimensions
                        (coil, te, ...) and coord_shape must match
                        coord.shape[:-1] used in prepare_nufft.
        coord (tensor): Coordinate array of shape [nframes, ..., ndim]
        shape (int or tuple of ints): Cartesian grid size.
        oversamp (float): Grid oversampling factor.
        width (int or tuple of int): Interpolation kernel full-width.
        basis (tensor): Low-rank temporal subspace basis.

    Returns:
        image (tensor): Image domain data of shape [n, ..., nz, ny, nx],
                        where n can be number of frames or low-rank subspace
                        coefficients and ... is a set of batches dimensions
                        (coil, te, ...).


    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.
    """
    # Prepare interpolator object
    interpolator = _routines.prepare_nufft(
        coord, shape, oversamp, width, basis, device, threadsperblock)

    # Calculate k-space data
    image = _routines.nufft_adjoint(kdata, interpolator)

    return image
