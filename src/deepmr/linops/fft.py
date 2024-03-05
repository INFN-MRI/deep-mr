"""Fast Fourier Transform linear operator (dense and sparse)."""

__all__ = ["FFTOp", "FFTGramOp", "SparseFFTOp", "SparseFFTGramOp"]

import numpy as np
import torch

import deepinv as dinv

from .. import fft as _fft


class FFTOp(dinv.physics.LinearPhysics):
    """
    Fast Fourier Transform operator.
    
    K-space sampling mask, if provided, is expected to to have the following dimensions:
        
    * 2D MRI: ``(ncontrasts, ny, nx)``
    * 3D MRI: ``(ncontrasts, nz, ny)``
    
    Input images are expected to have the following dimensions:
        
    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
    
    Similarly, output k-space data are expected to have the following dimensions:
        
    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny)``
    
    """
    
    def __init__(self, mask, basis=None, device=None, **kwargs):
        super().__init__(**kwargs)
        self._device = device
        if device is None:
            device = "cpu"
        if mask is not None:
            self._mask = torch.as_tensor(mask, device=device)
        else:
            self._mask = None
        if basis is not None:
            self._basis = torch.as_tensor(basis, device=device)
        else:
            self._basis = None
                       
    def A(self, x):
        """
        Apply Sparse Fast Fourier Transform.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny)`` (3D).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

        """
        if isinstance(x, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False
            
        # convert to tensor
        x = torch.as_tensor(x)
        
        if self._device is None:
            self._device = x.device
            
        # cast
        x = x.to(self._device)
        if self._basis is not None:
            self._basis = self._basis.to(self._device)         
        if self._mask is not None:
            self._mask = self._mask.to(self._device)
        
        # get adjoint basis
        if self._basis is not None:
            basis_adjoint = self._basis.conj().T
        else:
            basis_adjoint = None
                        
        # apply Fourier transform
        y = _fft.fft(x, axes=(-1, -2), norm="ortho")
        
        # project
        if basis_adjoint is not None:
            y = y[..., None]
            y = y.swapaxes(-4, -1)
            yshape = list(y.shape)
            y = y.reshape(-1, y.shape[-1]) # (prod(y.shape[:-1]), ncoeff)
            y = y @ basis_adjoint # (prod(y.shape[:-1]), ncontrasts)
            y = y.reshape(*yshape[:-1], y.shape[-1])
            y = y.swapaxes(-4, -1)
            y = y[..., 0]
        
        # mask if required
        if self._bask is not None:
            y = self._mask * y
            
        # cast back to numpy if required
        if isnumpy:
            y = y.numpy(force=True)
            
        return y
                        
    def A_adjoint(self, y):
        """
        Apply adjoint Non-Uniform Fast Fourier Transform.

        Parameters
        ----------
        y : torch.Tensor
            Input sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny)`` (3D).

        """
        if isinstance(y, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False
            
        # convert to tensor
        y = torch.as_tensor(y)
        
        if self._device is None:
            self._device = y.device
            
        # cast
        y = y.to(self._device)
        if self._mask is not None:
            self._mask = self._mask.to(self._device)
        if self._basis is not None:
            self._basis = self._basis.to(self._device)
                        
        # mask if required
        if self._bask is not None:
            y = self._mask * y
            
        # project
        if self._basis is not None:
            y = y[..., None]
            y = y.swapaxes(-4, -1)
            yshape = list(y.shape)
            y = y.reshape(-1, y.shape[-1]) # (prod(y.shape[:-1]), ncoeff)
            y = y @ self._basis # (prod(y.shape[:-1]), ncontrasts)
            y = y.reshape(*yshape[:-1], y.shape[-1])
            y = y.swapaxes(-4, -1)
            y = y[..., 0]
            
        # apply Fourier transform
        x = _fft.ifft(y, axes=(-1, -2), norm="ortho")
        
        # cast back to numpy if required
        if isnumpy:
            x = x.numpy(force=True)
                    
        return x


class FFTGramOp(dinv.physics.LinearPhysics):
    """
    Self-adjoint Sparse Fast Fourier Transform operator.
    
    K-space sampling mask, if provided, is expected to to have the following dimensions:
        
    * 2D MRI: ``(ncontrasts, ny, nx)``
    * 3D MRI: ``(ncontrasts, nz, ny)``
    
    Input and output images are expected to have the following dimensions:
        
    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
        
    """
    
    def __init__(self, mask, basis=None, device=None, **kwargs):
        super().__init__(**kwargs)
        self._device = device
        if device is None:
            device = "cpu"
        if basis is not None:
            basis = torch.as_tensor(basis, device=device)
        else:
            basis = None
        if mask is not None:
            mask = torch.as_tensor(mask, device=device)
        else:
            mask = None
            
        # calculate space-time kernel
        if basis is not None and mask is not None:            
            T, K = basis.shape
            nt, nz, ny = mask.shape  # or (nt, ny, nx) for 2D
            assert(nt == T)
            tmp = mask.permute(2, 1, 0).reshape((ny, nz, T, 1, 1)) * basis.reshape((1, 1, nt, 1, K)) # (ny, nz, T, 1, K) / (nx, ny, T, 1, K)
            toeplitz_kern = (tmp * basis.reshape(1, 1, T, K, 1)).sum(axis=2) # (ny, nz, K, K) / (nx, ny, K, K)
            toeplitz_kern = torch.fft.fftshift(torch.fft.fftshift(toeplitz_kern, axis=0), axis=1)
            self._toeplitz_kern = toeplitz_kern.swapaxes(0, 1).reshape(-1, K, K).contiguous() # (nz*ny, K, K) / (ny*nx, K, K)
        else:
            self._toeplitz_kern = None
            
    def A(self, x):
        """
        Apply Toeplitz convolution (``SparseFFT.H * SparseFFT``).
        
        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny)`` (3D).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny)`` (3D).

        """
        if isinstance(x, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False
            
        # convert to tensor
        x = torch.as_tensor(x)
        
        if self._device is None:
            self._device = x.device
            
        # cast
        x = x.to(self._device)
        if self._toeplitz_kern is not None:
            self._toeplitz_kern = self._toeplitz_kern.to(self._device)
            
        # fourier transform
        y = _fft.fft(x, axes=(-1, -2), center=False)
                                    
        # project if required
        if self._toeplitz_kern is not None:
            y = y[..., None] # (..., ncoeff, nz, ny, 1) / (..., ncoeff, ny, nx, 1)
            y = y.swapaxes(-4, -1) # (..., 1, nz, ny, ncoeff) / (..., 1, ny, nx, ncoeff)
            yshape = list(y.shape)
            y = y.reshape(int(np.prod(yshape[:-4])), -1, y.shape[-1]) # (prod(y.shape[:-4]), nz*ny, ncoeff) / (prod(y.shape[:-4]), ny*nx, ncoeff)
            y = torch.einsum("...bi,bij->...bj", y, self._toeplitz_kern)
            y = y.reshape(*yshape) # (..., 1, nz, ny, ncoeff) / # (..., 1, ny, nx, ncoeff)
            y = y.swapaxes(-4, -1) # (..., ncoeff, nz, ny, 1) / # (..., ncoeff, ny, nx, 1)
            y = y[..., 0] # (..., ncoeff, nz, ny) / # (..., ncoeff, ny, nx)
            
        # apply Fourier transform
        x = _fft.ifft(y, axes=(-1, -2), norm="ortho")
        
        # cast back to numpy if required
        if isnumpy:
            x = x.numpy(force=True)
                    
        return x
    
    def A_adjoint(self, y):
        """
        Apply Toeplitz convolution (``SparseFFT.H * SparseFFT``).
        
        This is the same as the forward operator (i.e., self-adjoint).

        Parameters
        ----------
        y : torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny)`` (3D).

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny)`` (3D).

        """
        if isinstance(y, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False
            
        # convert to tensor
        y = torch.as_tensor(y)
        
        if self._device is None:
            self._device = y.device
            
        # cast
        y = y.to(self._device)
        if self._toeplitz_kern is not None:
            self._toeplitz_kern = self._toeplitz_kern.to(self._device)
            
        # fourier transform
        x = _fft.fft(y, axes=(-1, -2), center=False)
                                    
        # project if required
        if self._toeplitz_kern is not None:
            x = x[..., None] # (..., ncoeff, nz, ny, 1) / (..., ncoeff, ny, nx, 1)
            x = x.swapaxes(-4, -1) # (..., 1, nz, ny, ncoeff) / (..., 1, ny, nx, ncoeff)
            xshape = list(x.shape)
            x = x.reshape(int(np.prod(xshape[:-4])), -1, x.shape[-1]) # (prod(y.shape[:-4]), nz*ny, ncoeff) / (prod(y.shape[:-4]), ny*nx, ncoeff)
            x = torch.einsum("...bi,bij->...bj", x, self._toeplitz_kern)
            x = x.reshape(*xshape) # (..., 1, nz, ny, ncoeff) / # (..., 1, ny, nx, ncoeff)
            x = x.swapaxes(-4, -1) # (..., ncoeff, nz, ny, 1) / # (..., ncoeff, ny, nx, 1)
            x = x[..., 0] # (..., ncoeff, nz, ny) / # (..., ncoeff, ny, nx)
            
        # apply Fourier transform
        y = _fft.ifft(x, axes=(-1, -2), norm="ortho")
        
        # cast back to numpy if required
        if isnumpy:
            y = y.numpy(force=True)
                    
        return y
    
    
class SparseFFTOp(dinv.physics.LinearPhysics):
    """
    Sparse Fast Fourier Transform operator.
    
    K-space sampling locations are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.
    
    Input images are expected to have the following dimensions:
        
    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
    
    Similarly, output k-space data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``.
    
    """
    
    def __init__(self, indexes, shape, basis=None, weight=None, device="cpu", threadsperblock=128, **kwargs):
        super().__init__(**kwargs)
        self._nufft_plan = _fft.prepare_sampling(indexes, shape, device)
        if weight is not None:
            self._weight = torch.as_tensor(weight**0.5, device=device)
        else:
            self._weight = None
        if basis is not None:
            self._basis = torch.as_tensor(basis, device=device)
        else:
            self._basis = None
        self._threadsperblock = threadsperblock
    
    def A(self, x):
        """
        Apply Sparse Fast Fourier Transform.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

        """
        if self._basis is not None:
            basis_adjoint = self._basis.conj().T
        else:
            basis_adjoint = None
        return _fft.apply_nufft(x, self._nufft_plan, basis_adjoint, self._weight, threadsperblock=self._threadsperblock)
    
    def A_adjoint(self, y):
        """
        Apply adjoint Non-Uniform Fast Fourier Transform.

        Parameters
        ----------
        y : torch.Tensor
            Input sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        """
        return _fft.apply_nufft_adj(y, self._nufft_plan, self._basis, self._weight, threadsperblock=self._threadsperblock)
    

class SparseFFTGramOp(dinv.physics.LinearPhysics):
    """
    Self-adjoint Sparse Fast Fourier Transform operator.
    
    K-space sampling locations are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.
    
    Input and output data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``,
    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
        
    """
    
    def __init__(self, coord, shape, basis=None, weight=None, device="cpu", threadsperblock=128, **kwargs):
        super().__init__(**kwargs)
        self._toeplitz_kern = _fft.plan_toeplitz_fft(coord, shape, basis, weight, device)
        self._threadsperblock = threadsperblock
    
    def A(self, x):
        """
        Apply Toeplitz convolution (``SparseFFT.H * SparseFFT``).
        
        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        """
        return _fft.apply_sparse_fft_selfadj(x, self._toeplitz_kern, threadsperblock=self._threadsperblock)
    
    def A_adjoint(self, y):
        """
        Apply Toeplitz convolution (``SparseFFT.H * SparseFFT``).
        
        This is the same as the forward operator (i.e., self-adjoint).

        Parameters
        ----------
        y : torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        """
        return _fft.apply_sparse_fft_selfadj(y, self._toeplitz_kern, threadsperblock=self._threadsperblock)
