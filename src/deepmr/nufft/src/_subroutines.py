# -*- coding: utf-8 -*-
"""
Private subroutines for FFT and NUFFT.

@author: Matteo Cencini
"""
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=redefined-builtin
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long

from typing import List, Tuple, Dict, Union

import numpy as np
import numba as nb

import torch
from torch import Tensor

from nufftorch.src import _cpu

if torch.cuda.is_available():
    from nufftorch.src import _cuda


class Utils:
    """Miscellaneous utility functions."""

    @staticmethod
    def _get_oversampled_shape(oversamp: float, shape:  Union[List[int], Tuple[int]]) -> Tuple[int]:
        return [np.ceil(oversamp * axis).astype(int) for axis in shape]

    @staticmethod
    def _scalars2arrays(ndim, *inputs):
        arrays = []

        for input in inputs:
            if np.isscalar(input):
                arrays.append(np.array([input] * ndim, dtype=np.int16))
            else:
                arrays.append(np.array(input, dtype=np.int16))

        return arrays

    @staticmethod
    def _beatty_parameter(width, oversamp):
        
        # fix for width = 1
        width = width.copy()
        width[width < 2] = 2
        
        return np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    @staticmethod
    def _scale_coord(coord, shape, oversamp):
        ndim = coord.shape[-1]
        coord = coord.clone()

        for i in range(-ndim, 0):
            scale = np.ceil(oversamp * shape[i]) / shape[i]
            shift = np.ceil(oversamp * shape[i]) // 2
            coord[..., i] *= scale
            coord[..., i] += shift

        return coord
    
    @staticmethod
    def _get_kernel_scaling(beta, width):
        
        # init kernel centered on k-space node
        value = []
        
        # fill the three axes
        for ax in range(len(width)):
            start = np.ceil(-width[ax] / 2)
            value.append(np.array([_cpu._kernel._function((start + el) / (width[ax] / 2), beta[ax]) for el in range(width[ax])]))
                        
        value = np.stack(np.meshgrid(*value), axis=0).prod(axis=0)
        
        return value.sum()


class DeviceDispatch:
    """Manage computational devices."""

    computation_device: str
    data_device: str

    def __init__(self, computation_device: str, data_device: str):
        """DeviceDispatch object constructor.

        Args:
            computation_device: target device to perform the computation.
            data_device: original device hosting the data (could be the same)
                         as computation device.

        """
        self.computation_device = computation_device
        self.data_device = data_device

    def dispatch(self, *tensors):
        """Dispatch input to computational device."""
        for tensor in tensors:
            if tensor is not None:
                tensor.to(self.computation_device)
                
        if len(tensors) == 1:
            tensors = tensors[0]

        return tensors

    def gather(self, *tensors):
        """Gather output to original device"""
        for tensor in tensors:
            if tensor is not None:
                tensor.to(self.data_device)
                
        if len(tensors) == 1:
            tensors = tensors[0]

        return tensors


class DataReshape:
    """Ravel and unravel multi-channel/-echo/-slice sparse k-space data."""
    ndim: int
    batch_shape: Union[List[int], Tuple[int]]
    batch_size: int
    grid_shape:  Union[List[int], Tuple[int]]
    grid_size: int
    batch_axis_shape: Union[None, List[int], Tuple[int]]
    nbatches: int
    nframes: int
    device: Union[str, torch.device]
    dtype: torch.dtype

    def __init__(self,
                 coord_shape: Union[List[int], Tuple[int]],
                 grid_shape: Union[List[int], Tuple[int]]):
        """DataReshape object constructor.

        Args:
            coord_shape: shape of k-space coordinates tensor.

        """
        self.ndim = coord_shape[-1]
        self.batch_shape = coord_shape[1:-1]
        self.batch_size = int(np.prod(self.batch_shape))
        self.grid_shape = grid_shape
        self.grid_size = int(np.prod(self.grid_shape))
        self.nframes = coord_shape[0]

    def ravel_data(self, input: Tensor) -> Tensor:
        """Ravel multi-channel/-echo/-slice data.

        Args:
            input (tensor): input data tensor of size [*(batch_axis_shape), *(batch_shape)]
                            where batch_axis are one or more dimensions corresponding
                            to different batches (e.g. channels/echoes/slices/...)
                            with same spectral locations
                            and batch_shape are one or more dimension describing
                            different spectral locations
                            (e.g. (n_readouts, n_point_per readout)).

        Returns:
            tensor: output 3D raveled data tensor of shape
                    (n_frames, n_batches, n_spectral_locations).
        """
        # get input device and data type
        self.device = input.device
        self.dtype = input.dtype

        # keep original batch shape
        self.batch_axis_shape = input.shape[1:-len(self.batch_shape)]
        self.nbatches = int(np.prod(self.batch_axis_shape))
 
        return input.reshape(self.nframes, self.nbatches, self.batch_size)

    def ravel_grid(self, input: Tensor) -> Tensor:
        """Ravel multi-channel/-echo/-slice data.

        Args:
            input (tensor): input data tensor of size [*(batch_axis_shape), *(grid_shape)]
                            where batch_axis are one or more dimensions corresponding
                            to different batches (e.g. channels/echoes/slices/...)
                            with same spectral locations
                            and batch_shape are one or more dimension describing
                            different spectral locations
                            (e.g. ((nz), ny, nx)).

        Returns:
            tensor: output 3D raveled data tensor of shape
                    (n_frames, n_batches, n_spectral_locations).
        """
        # get input device and data type
        self.device = input.device
        self.dtype = input.dtype
        
        # keep original batch shape
        self.batch_axis_shape = input.shape[1:-self.ndim]
        self.nbatches = int(np.prod(self.batch_axis_shape))

        return input.reshape(input.shape[0], self.nbatches, self.grid_size)

    def unravel_data(self, input: Tensor) -> Tensor:
        """Unavel raveled data, restoring original shape.

        Args:
            input (tensor): input 3D raveled data tensor  of shape
                            (n_frames, n_batches, n_spectral_locations).

        Returns:
            tensor: output original-shape data.
        """
        return input.reshape(self.nframes, *self.batch_axis_shape, *self.batch_shape)

    def unravel_grid(self, input: Tensor) -> Tensor:
        """Unavel raveled data, restoring original shape.

        Args:
            input (tensor): input 3D raveled data tensor  of shape
                            (n_frames, n_batches, n_spectral_locations).

        Returns:
            tensor: output original-shape data.
        """
        return input.reshape(input.shape[0], *self.batch_axis_shape, *self.grid_shape)

    def generate_empty_data(self):
        """Generate empty Non-Cartesian data matrix."""
        return torch.zeros((self.nframes, self.nbatches, self.batch_size),
                           dtype=self.dtype, device=self.device)

    def generate_empty_grid(self, basis: Union[None, Tensor] = None):
        """Generate empty Cartesian data matrix."""
        if basis is None:
            return torch.zeros((self.nframes, self.nbatches, self.grid_size),
                               dtype=self.dtype, device=self.device)

        ncoeff = basis.shape[0]
        return torch.zeros((ncoeff, self.nbatches, self.grid_size),
                           dtype=self.dtype, device=self.device)


class Apodize:
    """Image-domain apodization operator to correct effect of convolution."""
    ndim: int
    apod: List[float]

    def __init__(self,
                 grid_shape: Union[List[int], Tuple[int]],
                 oversamp: float,
                 width: Union[List[int], Tuple[int]],
                 beta: Union[List[float], Tuple[float]],
                 device: Union[str, torch.device]):

        # get number of spatial dimensions
        self.ndim = len(grid_shape)
        self.apod = []
        
        # fix for width = 1
        width = width.copy()
        width[width < 2] = 2

        for axis in range(-self.ndim, 0):
            i = grid_shape[axis]
            os_i = np.ceil(oversamp * i)
            idx = torch.arange(i, dtype=torch.float32, device=device)

            # Calculate apodization
            apod = (beta[axis]**2 - (np.pi * width[axis] * (idx - i // 2) / os_i)**2)**0.5
            apod /= torch.sinh(apod)
            
            # normalize by DC
            apod = apod / apod[int(i // 2)]
            
            # avoid NaN
            apod = torch.nan_to_num(apod, nan=1.0)
                        
            self.apod.append(apod.reshape([i] + [1] * (-axis - 1)))

    def __call__(self, input: Tensor):
        """Apply apodization step in-place.

        Args:
            input (tensor): Input image.
        """
        for axis in range(-self.ndim, 0):
            input *= self.apod[axis]


class ZeroPad:
    """ Image-domain padding operator to interpolate k-space on oversampled grid."""
    padsize: Tuple[int]

    def __init__(self, oversamp: float, shape: Union[List[int], Tuple[int]]):
        # get oversampled shape
        self.oshape = Utils._get_oversampled_shape(oversamp, shape)

    def __call__(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.

        Returns:
            tensor: Output zero-padded image.
        """
        # unpack
        oshape = np.array(self.oshape)
        ishape = np.array(input.shape)
        
        # get number of dimensions
        ndim = len(oshape)
        
        
        # get center
        center = oshape // 2
        
        # calculate start

        start = center - ishape[-ndim:] // 2
        end = start + ishape[-ndim:]
        
        # pad
        output = torch.zeros([*ishape[:-ndim], *oshape], dtype=input.dtype, device=input.device)
                
        if ndim == 2:
            output[..., start[0]:end[0], start[1]:end[1]] = input                         
        elif ndim == 3:
            output[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]] = input
        else:
            raise NotImplementedError('Only 2D or 3D images supported so far')
        
        return output
    

class Crop:
    """Image-domain cropping operator to select targeted FOV."""
    def __init__(self, shape: Union[List[int], Tuple[int]]):
        self.oshape = shape

    def __call__(self, input: Tensor) -> Tensor:
        """Apply zero padding step.

        Args:
            input (tensor): Input image.

        Returns:
            tensor: Output zero-padded image.
        """
        # unpack
        oshape = np.array(self.oshape)
        
        # get number of dimensions
        ndim = len(oshape)
        
        # get center
        center = np.array(input.shape[-ndim:]) // 2
        
        # calculate start
        start = center - oshape // 2
        end = start + oshape
        
        # crop
        if ndim == 2:
            output = input[..., start[0]:end[0], start[1]:end[1]]                         
        elif ndim == 3:
            output = input[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        else:
            raise NotImplementedError('Only 2D or 3D images supported so far')
        
        return output

    
class BaseFFT:
    """Cartesian (Inverse) Fourier Transform."""
    def __init__(self, input):
        self.isreal = not(torch.is_complex(input))

    @staticmethod
    def _fftc(input, s=None, dim=None, norm='ortho'):
        # actual fft
        tmp = torch.fft.ifftshift(input, dim=dim)
        tmp = torch.fft.fftn(tmp, s=s, dim=dim, norm=norm)
        output = torch.fft.fftshift(tmp, dim=dim)

        return output

    @staticmethod
    def _ifftc(input, s=None, dim=None, norm='ortho'):
        # actual fft
        tmp = torch.fft.ifftshift(input, dim=dim)
        tmp = torch.fft.ifftn(tmp, s=s, dim=dim, norm=norm)
        output = torch.fft.fftshift(tmp, dim=dim)

        return output


class FFT(BaseFFT):
    """Cartesian Fourier Transform."""

    def __call__(self, input, oshape=None, axes=-1, center=True, norm='ortho'):
        # allow for single scalar axis
        if np.isscalar(axes):
            axes = [axes]

        # transform to list to allow range object
        if isinstance(axes, list) is False:
            axes = list(axes)   
            
        # process axes arg
        ndim = len(axes)

        # adapt output shape
        if oshape is None:
            oshape = input.shape[-ndim:]
        else:
            oshape = oshape[-ndim:]

        if center:
            output = BaseFFT._fftc(input, s=oshape, dim=axes, norm=norm)
        else:
            output = torch.fft.fftn(input, s=oshape, dim=axes, norm=norm)

        if torch.is_complex(input) and input.dtype != output.dtype:
            output = output.to(input.dtype, copy=False)
            
        # if required discard imaginary
        if self.isreal:
            output = output.real

        return output


class IFFT(BaseFFT):
    """Cartesian Inverse Fourier Transform."""

    def __call__(self, input, oshape=None, axes=-1, center=True, norm='ortho'):
        # allow for single scalar axis
        if np.isscalar(axes):
            axes = [axes]

        # transform to list to allow range object
        if isinstance(axes, list) is False:
            axes = list(axes)
        
        # process axes arg
        ndim = len(axes)

        # adapt output shape
        if oshape is None:
            oshape = input.shape[-ndim:]
        else:
            oshape = oshape[-ndim:]

        if center:
            output = BaseFFT._ifftc(input, s=oshape, dim=axes, norm=norm)
        else:
            output = torch.fft.ifftn(input, s=oshape, dim=axes, norm=norm)

        # if required discard imaginary
        if self.isreal:
            output = output.real

        return output


class BackendBridge:
    """Helper class to convert between pytorch and numba."""
    @staticmethod
    def numba2pytorch(*arrays, requires_grad=True):  # pragma: no cover
        """Zero-copy conversion from Numpy/Numba CUDA array to PyTorch tensor.

        Args:
            arrays (numpy/cupy array): list or tuple of input.
            requires_grad(bool): Set .requires_grad output tensor

        Returns:
            PyTorch tensor.
        """
        try:
            is_cuda = nb.cuda.is_cuda_array(arrays[0])
        except:
            is_cuda = False

        if is_cuda:
            tensors = [torch.as_tensor(array, device="cuda")
                       for array in arrays]
        else:
            tensors = [torch.from_numpy(array) for array in arrays]

        for tensor in tensors:
            tensor.requires_grad = requires_grad
            tensor = tensor.contiguous()
        
        if len(tensors) == 1:
            tensors = tensors[0]

        return tensors

    @staticmethod
    def pytorch2numba(*tensors):  # pragma: no cover
        """Zero-copy conversion from PyTorch tensor to Numpy/Numba CUDA array.

        Args:
            tensor (PyTorch tensor): input.

        Returns:
            Numpy/Numba CUDA array.

        """
        device = tensors[0].device

        if device.type == 'cpu':
            arrays = [tensor.detach().contiguous().cpu().numpy()
                      for tensor in tensors]
        else:
            arrays = [nb.cuda.as_cuda_array(tensor.contiguous())
                      for tensor in tensors]
            
        if len(arrays) == 1:
            arrays = arrays[0]

        return arrays


class DeGrid:
    """K-space data de-gridding operator."""
    device: Union[str, torch.device]
    threadsperblock: int

    def __init__(self, device_dict: Dict):
        self.device = device_dict['device']
        self.threadsperblock = device_dict['threadsperblock']

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.module = _cpu
        else:
            self.module = _cuda

    def __call__(self, input: Tensor, kernel_dict: Dict) -> Tensor:
        """Interpolate Cartesian k-space data to given coordinates.

        Args:
            input (tensor): Cartesian sparse k-space data.

        Returns:
            tensor: Output Non-Cartesian k-space data.
        """
        # unpack input
        sparse_coeff = kernel_dict['sparse_coefficients']
        coord_shape = kernel_dict['coord_shape']
        grid_shape = kernel_dict['grid_shape']
        basis_adjoint = kernel_dict['basis_adjoint']

        # reformat data for computation
        reformat = DataReshape(coord_shape, grid_shape)
        input = reformat.ravel_grid(input)

        # preallocate output data
        output = reformat.generate_empty_data()

        # do actual interpolation
        output, input = BackendBridge.pytorch2numba(output, input)
        self.module._DeGridding(output.size, sparse_coeff, basis_adjoint, self.threadsperblock)(output, input)
        output, input = BackendBridge.numba2pytorch(output, input)

        # reformat for output
        output = reformat.unravel_data(output)

        # restore original shape
        input = reformat.unravel_grid(input)

        return output


class Grid:
    """K-space data gridding operator."""
    device: Union[str, torch.device]
    threadsperblock: int

    def __init__(self, device_dict: Dict):
        self.device = device_dict['device']
        self.threadsperblock = device_dict['threadsperblock']

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.module = _cpu
        else:
            self.module = _cuda

    def __call__(self, input: Tensor, kernel_dict: Dict) -> Tensor:
        """Grid sparse k-space data on a Cartesian Grid.

        Args:
            input (tensor): Non-Cartesian k-space data.

        Returns:
            tensor: Cartesian sparse k-space data.
        """
        # unpack input
        sparse_coeff = kernel_dict['sparse_coefficients']
        coord_shape = kernel_dict['coord_shape']
        grid_shape = kernel_dict['grid_shape']
        basis = kernel_dict['basis']

        # reformat data for computation
        reformat = DataReshape(coord_shape, grid_shape)
        input = reformat.ravel_data(input)

        # preallocate output data
        output = reformat.generate_empty_grid(basis)

        # do actual interpolation
        output, input = BackendBridge.pytorch2numba(output, input)
        self.module._Gridding(input.size, sparse_coeff, basis, self.threadsperblock)(output, input)
        output, input = BackendBridge.numba2pytorch(output, input)

        # reformat for output
        output = reformat.unravel_grid(output)

        # restore original shape
        input = reformat.unravel_data(input)

        return output


class Toeplitz:
    """Perform Toeplitz convolution for fast self-adjoint."""

    def __init__(self, data_size: int, device_dict: Dict):
        device = device_dict['device']
        threadsperblock = device_dict['threadsperblock']

        # calculate blocks per grid
        blockspergrid = int((data_size + (threadsperblock - 1)) // threadsperblock)

        if device == 'cpu' or device == torch.device('cpu'):
            self._apply = _cpu._batched_dot_product
        else:
            self._apply = _cuda._batched_dot_product[blockspergrid, threadsperblock]

    def __call__(self, kdata_out, kdata_in, mtf):
        kdata_out, kdata_in = BackendBridge.pytorch2numba(kdata_out, kdata_in)
        self._apply(kdata_out, kdata_in, mtf)
        kdata_out, kdata_in = BackendBridge.numba2pytorch(kdata_out, kdata_in)
