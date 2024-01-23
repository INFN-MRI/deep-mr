# -*- coding: utf-8 -*-
"""
Object-oriented wrappers for Non-uniform Fourier Transform routines.


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
from typing import List, Tuple, Dict, Union

import torch
from torch import Tensor

from nufftorch import _autograd
from nufftorch.src import _routines


class AbstractNUFFT(torch.nn.Module):  # pylint: disable=abstract-method
    """common class for forward and adjoint NUFFT."""

    def __init__(
        self,
        coord: Union[None, Tensor] = None,
        shape: Union[None, int, List[int], Tuple[int]] = None,
        oversamp: Union[float, List[float], Tuple[float]] = 2.0,
        width: Union[int, List[int], Tuple[int]] = 4,
        basis: Union[None, Tensor] = None,
        device: str = "cpu",
        interpolator: Union[None, Dict] = None,
    ):
        super().__init__()

        # if provided, re-use precomputed interpolator.
        if interpolator is None:
            self.interpolator = _routines.prepare_nufft(
                coord, shape, oversamp, width, basis, device
            )
        else:
            self.interpolator = interpolator

    def _adjoint_linop(self):
        raise NotImplementedError

    @property
    def H(self):
        r"""Return adjoint linear operator.

        An adjoint linear operator :math:`A^H` for
        a linear operator :math:`A` is defined as:

        .. math:
            \left< A x, y \right> = \left< x, A^H, y \right>

        Returns:
            Linop: adjoint linear operator.

        """
        return self._adjoint_linop()


class NUFFT(AbstractNUFFT):
    """Non-Uniform Fourier Transform operator with embedded low-rank projection.

    Args:
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

    def forward(self, image):
        """Perform forward NUFFT operation.

        Args:
            image (tensor): Input data in image space of shape [n, ..., nz, ny, nx],
                            where n can be number of frames or low-rank subspace
                            coefficients and ... is a set of batches dimensions
                            (coil, te, ...).

        Returns:
            kdata (tensor): Fourier domain data of shape [nframes, ..., coord_shape],
                            where  ... is a set f batches dimensions
                            (coil, te, ...) and coord_shape must match
                            coord.shape[:-1] used in prepare_nufft.
        """
        return _autograd._nufft.apply(image, self.interpolator)

    def _adjoint_linop(self):
        return NUFFTAdjoint(interpolator=self.interpolator)


class NUFFTAdjoint(AbstractNUFFT):
    """Adjoint Non-Uniform Fourier Transform operator with embedded low-rank projection.

    Args:
        coord (tensor): Coordinate array of shape [nframes, ..., ndim]
        shape (int or tuple of ints): Cartesian grid size.
        oversamp (float): Grid oversampling factor.
        width (int or tuple of int): Interpolation kernel full-width.
        basis (tensor): Low-rank temporal subspace basis.

    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.
    """

    def forward(self, kdata):
        """Perform adjoint NUFFT operation.

        Args:
            kdata (tensor): Fourier domain data of shape [nframes, ..., coord_shape],
                            where  ... is a set f batches dimensions
                            (coil, te, ...) and coord_shape must match
                            coord.shape[:-1] used in prepare_nufft.

        Returns:
            image (tensor): Image domain data of shape [n, ..., nz, ny, nx],
                            where n can be number of frames or low-rank subspace
                            coefficients and ... is a set of batches dimensions
                            (coil, te, ...).
        """
        return _autograd._nufft_adjoint.apply(kdata, self.interpolator)

    def _adjoint_linop(self):
        return NUFFT(interpolator=self.interpolator)


class NUFFTSelfadjoint(torch.nn.Module):
    """Self-adjoint Non-Uniform Fourier Transform operator."""

    def __init__(
        self,
        coord: Tensor,
        shape: Union[int, List[int], Tuple[int]],
        prep_oversamp: Union[float, List[float], Tuple[float]] = 2.0,
        comp_oversamp: Union[float, List[float], Tuple[float]] = 2.0,
        width: Union[int, List[int], Tuple[int]] = 4,
        basis: Union[Tensor, None] = None,
        device: Union[str, torch.device] = "cpu",
        threadsperblock: int = 512,
        dcf: Union[Tensor, None] = None,
        toeplitz_kernel: Union[None, Dict] = None,
    ):
        super().__init__()

        # if provided, re-use precomputed interpolator.
        if toeplitz_kernel is None:
            self.toeplitz_kernel = _routines.prepare_noncartesian_toeplitz(
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
        else:
            self.toeplitz_kernel = toeplitz_kernel

    def forward(self, image):
        """Perform self-adjoint NUFFT operation.

        Args:
            image (tensor): Input data in image space of shape [n, ..., nz, ny, nx],
                            where n can be number of frames or low-rank subspace
                            coefficients and ... is a set of batches dimensions
                            (coil, te, ...).

        Returns:
            image (tensor): Image domain data of shape [n, ..., nz, ny, nx],
                            where n can be number of frames or low-rank subspace
                            coefficients and ... is a set of batches dimensions
                            (coil, te, ...).
        """
        return _autograd._nufft_selfadjoint.apply(image, self.toeplitz_kernel)

    def _adjoint_linop(self):
        return self
