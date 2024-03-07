"""Non-Uniform Fast Fourier Transform linear operator."""

__all__ = ["NUFFTOp", "NUFFTGramOp"]

import torch

from .. import fft as _fft

from . import base

class NUFFTOp(base.Linop):
    """
    Non-Uniform Fast Fourier Transform operator.
    
    K-space sampling trajectory are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.
    
    Input images are expected to have the following dimensions:
        
    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
    
    Similarly, output k-space data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``.
    
    """
    
    def __init__(self, coord, shape, basis=None, weight=None, device="cpu", threadsperblock=128, width=3, oversamp=1.125, **kwargs):
        super().__init__(**kwargs)
        self._nufft_plan = _fft.plan_nufft(coord, shape, width, oversamp, device)
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
        Apply Non-Uniform Fast Fourier Transform.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

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
            Input Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
            or ``(..., ncontrasts, nz, ny, nx)`` (3D).

        """
        return _fft.apply_nufft_adj(y, self._nufft_plan, self._basis, self._weight, threadsperblock=self._threadsperblock)
    

class NUFFTGramOp(base.Linop):
    """
    Self-adjoint Non-Uniform Fast Fourier Transform operator.
    
    K-space sampling trajectory are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.
    
    Input and output data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``,
    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
        
    """
    
    def __init__(self, coord, shape, basis=None, weight=None, device="cpu", threadsperblock=128, width=3, **kwargs):
        super().__init__(**kwargs)
        self._toeplitz_kern = _fft.plan_toeplitz_nufft(coord, shape, basis, weight, width, device)
        self._threadsperblock = threadsperblock
    
    def A(self, x):
        """
        Apply Toeplitz convolution (``NUFFT.H * NUFFT``).
        
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
        return _fft.apply_nufft_selfadj(x, self._toeplitz_kern, threadsperblock=self._threadsperblock)
    
    def A_adjoint(self, y):
        """
        Apply Toeplitz convolution (``NUFFT.H * NUFFT``).
        
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
        return _fft.apply_nufft_selfadj(y, self._toeplitz_kern, threadsperblock=self._threadsperblock)
