# -*- coding: utf-8 -*-
"""
Factory routines to build NUFFT/Toeplitz objects.

@author: Matteo Cencini
"""
# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
# pylint: disable=line-too-long

import numpy as np
import torch

from nufftorch.src import _cpu
from nufftorch.src._subroutines import (
    Apodize,
    BackendBridge,
    Crop,
    FFT,
    Grid,
    IFFT,
    Utils,
    ZeroPad,
)


class AbstractFactory:
    """Base class for NUFFT and Toeplitz factory classes."""

    def _parse_coordinate_size(self, coord):
        # inspect input coordinates
        self.nframes = coord.shape[0]
        pts_shape = coord.shape[1:-1]
        self.ndim = coord.shape[-1]

        # calculate total number of point per frame and total number of points
        self.npts = np.prod(pts_shape)
        self.coord_length = self.nframes * self.npts

    def _reformat_coordinates(self, coord):
        return coord.reshape([self.coord_length, self.ndim]).T


class NUFFTFactory(AbstractFactory):
    """Functions to prepare NUFFT operator."""

    def __call__(self, coord, shape, width, oversamp, basis, device, threadsperblock):
        """Actual pre-computation routine."""
        # get parameters
        ndim = coord.shape[-1]

        # get arrays
        width, shape = Utils._scalars2arrays(ndim, width, shape)

        # get kernel shape parameter
        beta = Utils._beatty_parameter(width, oversamp)

        # get grid shape
        grid_shape = Utils._get_oversampled_shape(oversamp, shape)

        # adjust coordinates
        coord = Utils._scale_coord(coord, shape, oversamp)

        # compute interpolator
        kernel_dict = self._prepare_kernel(
            coord, grid_shape, width, beta, basis, device
        )

        # get scaling
        scale = Utils._get_kernel_scaling(beta, width)

        # device_dict
        device_dict = {"device": device, "threadsperblock": threadsperblock}

        # create interpolator dict
        interpolator = {
            "ndim": ndim,
            "shape": shape,
            "oversamp": oversamp,
            "width": width,
            "beta": beta,
            "kernel_dict": kernel_dict,
            "scale": scale,
            "device_dict": device_dict,
        }

        return interpolator

    def _prepare_kernel(self, coord, shape, width, beta, basis, device):
        # parse size
        self._parse_coordinate_size(coord)

        # reformat coordinates
        coord_shape = coord.shape
        coord = self._reformat_coordinates(coord)

        # calculate interpolator
        kernel_tuple = self._prepare_sparse_coefficient_matrix(
            coord, shape, width, beta, device
        )

        # adjust basis
        if basis is not None:
            # build basis adjoint
            basis_adjoint = basis.conj().T

            # convert to numpy or numba cuda array
            basis = BackendBridge.pytorch2numba(basis.to(device))
            basis_adjoint = BackendBridge.pytorch2numba(basis_adjoint.to(device))
        else:
            basis_adjoint = None

        # prepare kernel dictionary
        kernel_dict = {
            "sparse_coefficients": kernel_tuple,
            "coord_shape": coord_shape,
            "grid_shape": shape,
            "width": width,
            "basis": basis,
            "basis_adjoint": basis_adjoint,
        }

        return kernel_dict

    def _preallocate_sparse_coefficient_matrix(self, width):
        index = []
        value = []

        for i in range(self.ndim):
            # kernel value
            empty_value = torch.zeros(
                (self.coord_length, width[i]), dtype=torch.float32
            )
            value.append(empty_value)

            # kernel index
            empty_index = torch.zeros((self.coord_length, width[i]), dtype=torch.int32)
            index.append(empty_index)

        return value, index

    def _reformat_sparse_coefficients(self, value, index, width, device):
        for i in range(self.ndim):
            # reshape
            value[i] = value[i].reshape([self.nframes, self.npts, width[i]]).to(device)
            index[i] = index[i].reshape([self.nframes, self.npts, width[i]]).to(device)

        # stack (support isotropic kernel only)
        value = torch.stack(value, dim=0)
        index = torch.stack(index, dim=0)

        # convert to numpy / numba array
        value, index = BackendBridge.pytorch2numba(value, index)

        return value, index

    def _prepare_sparse_coefficient_matrix(self, coord, shape, width, beta, device):
        # preallocate interpolator
        value, index = self._preallocate_sparse_coefficient_matrix(width)

        # actual computation
        for i in range(self.ndim):
            val, ind, coo = BackendBridge.pytorch2numba(value[i], index[i], coord[i])
            _cpu._prepare_sparse_coefficient_matrix(val, ind, coo, beta[i], shape[i])

        # reformat coefficients
        value, index = self._reformat_sparse_coefficients(value, index, width, device)

        # get grid offset and kernel width
        offset = torch.tensor(shape, device=device).cumprod(dim=0)[:-1]
        width = torch.tensor(width, device=device)
        offset, width = BackendBridge.pytorch2numba(offset, width)

        # pack kernel tuple
        kernel_tuple = (value, index, width, offset)

        return kernel_tuple


class NonCartesianToeplitzFactory(AbstractFactory):
    """Functions to prepare Toeplitz kernel for Non-Cartesian sampling."""

    def __call__(
        self,
        coord,
        shape,
        prep_osf,
        comp_osf,
        width,
        basis,
        device,
        threadsperblock,
        dcf,
    ):
        """Actual pre-computation routine."""
        # clone coordinates and dcf to avoid modifications
        coord = coord.clone()

        # kernel precomputation
        mtf = self._initialize_mtf(
            coord, shape, width, prep_osf, basis, device, threadsperblock, dcf
        )

        # kernel resampling
        mtf = self._resample_mtf(mtf, shape, prep_osf, comp_osf, width, device)

        # post process
        mtf = self._post_process_mtf(mtf)

        # device_dict
        device_dict = {"device": device, "threadsperblock": threadsperblock}

        return {
            "mtf": mtf,
            "islowrank": self.islowrank,
            "device_dict": device_dict,
            "ndim": self.ndim,
            "oversamp": comp_osf,
        }

    def _initialize_mtf(
        self, coord, shape, width, prep_osf, basis, device, threadsperblock, dcf
    ):
        """Core kernel computation sub-routine."""
        # parse size
        self._parse_coordinate_size(coord)

        # process basis
        basis, basis_adjoint = self._process_basis(basis, device)

        # prepare dcf for MTF/PSF estimation
        dcf = self._prepare_dcf(dcf, basis_adjoint, device)

        # prepare Interpolator object
        interpolator = NUFFTFactory()(
            coord, shape, width, prep_osf, basis, device, threadsperblock
        )

        # get scaling
        scale = (prep_osf**self.ndim) / interpolator["scale"]

        return (
            Grid(interpolator["device_dict"])(dcf, interpolator["kernel_dict"]) * scale
        )

    def _resample_mtf(self, mtf, shape, prep_osf, comp_osf, width, device):
        # get arrays
        width, shape = Utils._scalars2arrays(self.ndim, width, shape)

        # get kernel shape parameter
        beta = Utils._beatty_parameter(width, prep_osf)

        # IFFT
        psf = IFFT(mtf)(mtf, axes=range(-self.ndim, 0), norm=None)

        # Crop
        psf = Crop(shape[-self.ndim :])(psf)

        # Apodize
        Apodize(shape[-self.ndim :], prep_osf, width, beta, device)(psf)

        # Zero-pad
        psf = ZeroPad(comp_osf, shape[-self.ndim :])(psf)

        # FFT
        mtf = FFT(psf)(psf, axes=range(-self.ndim, 0), norm=None)

        return mtf

    def _post_process_mtf(self, mtf):
        # fftshift kernel to accelerate computation
        mtf = torch.fft.fftshift(mtf, dim=list(range(-self.ndim, 0)))

        if self.islowrank:
            mtf = mtf.reshape(
                *mtf.shape[:2], np.prod(mtf.shape[-self.ndim :])
            ).T.contiguous()

        # remove NaN
        if self.isreal:
            mtf = mtf.real
            mtf = torch.nan_to_num(mtf)
        else:
            mtf = torch.nan_to_num(mtf.real) + 1j * torch.nan_to_num(mtf.imag)

        # transform to numba
        if self.islowrank:
            mtf = BackendBridge.pytorch2numba(mtf)

        return mtf

    # utils
    def _prepare_dcf(self, dcf, basis_adjoint, device):
        # if dcf are not provided, assume uniform sampling density
        if dcf is None:
            dcf = torch.ones(
                (self.nframes, 1, self.npts), torch.complex64, device=device
            )

        else:
            dcf = dcf.clone().squeeze()
            dcf = dcf.to(torch.complex64).to(device)

            if len(dcf.shape) == 2:
                dcf = dcf[:, None, :]

            if len(dcf.shape) == 1:
                dcf = dcf[None, None, :]

        # multiply by basis
        if basis_adjoint is not None:
            dcf = basis_adjoint[..., None] * dcf

        return dcf

    def _process_basis(self, basis, device):
        # inspect basis to determine data type
        if basis is not None:
            self.islowrank = True
            self.isreal = not torch.is_complex(basis)
            self.ncoeff = basis.shape[0]
        else:
            self.islowrank = False
            self.isreal = False
            self.ncoeff = self.nframes

        if self.isreal:
            self.dtype = torch.float32
        else:
            self.dtype = torch.complex64

        # if spatio-temporal basis is provided, check reality and offload to device
        if self.islowrank:
            basis = basis.to(self.dtype).to(device)
            basis_adjoint = basis.conj().T
        else:
            basis_adjoint = None

        return basis, basis_adjoint
