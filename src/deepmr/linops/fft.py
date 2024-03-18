"""Fast Fourier Transform linear operator."""

__all__ = ["FFTOp", "IFFTOp", "FFTGramOp"]

import numpy as np
import torch

from .. import fft as _fft

from . import base


class FFTOp(base.Linop):
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

    def __init__(self, mask=None, basis_adjoint=None, device=None):
        super().__init__(ndim=2)
        if device is None:
            device = "cpu"
        self.device = device

        if mask is not None:
            self.mask = torch.as_tensor(mask, device=device)
        else:
            self.mask = None
        if basis_adjoint is not None:
            self.basis_adjoint = torch.as_tensor(basis_adjoint, device=device)
        else:
            self.basis_adjoint = None

    def forward(self, x):
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

        if self.device is None:
            self.device = x.device

        # cast
        x = x.to(self._device)
        if self.basis is not None:
            self.basis = self.basis.to(self._device)
        if self.mask is not None:
            self.mask = self.mask.to(self._device)

        # get adjoint basis
        if self.basis_adjoint is not None:
            basis_adjoint = self.basis_adjoint
        else:
            basis_adjoint = None

        # apply Fourier transform
        y = _fft.fft(x, axes=(-1, -2), norm="ortho")

        # project
        if basis_adjoint is not None:
            y = y[..., None]
            y = y.swapaxes(-4, -1)
            yshape = list(y.shape)
            y = y.reshape(-1, y.shape[-1])  # (prod(y.shape[:-1]), ncoeff)
            y = y @ basis_adjoint  # (prod(y.shape[:-1]), ncontrasts)
            y = y.reshape(*yshape[:-1], y.shape[-1])
            y = y.swapaxes(-4, -1)
            y = y[..., 0]

        # mask if required
        if self.mask is not None:
            y = self.mask * y

        # cast back to numpy if required
        if isnumpy:
            y = y.numpy(force=True)

        return y

    def _adjoint_linop(self):
        # get adjoint basis
        if self.basis_adjoint is not None:
            basis = self.basis_adjoint.conj().t()
        else:
            basis = None
        return IFFTOp(self.mask, basis, self.device)


class IFFTOp(base.Linop):
    """
    Inverse Fast Fourier Transform operator.

    K-space sampling mask, if provided, is expected to to have the following dimensions:

    * 2D MRI: ``(ncontrasts, ny, nx)``
    * 3D MRI: ``(ncontrasts, nz, ny)``

    Input k-space data are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    Similarly, output images are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny)``

    """

    def __init__(self, mask=None, basis=None, device=None, **kwargs):
        super().__init__(ndim=2, **kwargs)
        if device is None:
            device = "cpu"
        self.device = device

        if mask is not None:
            self.mask = torch.as_tensor(mask, device=device)
        else:
            self.mask = None
        if basis is not None:
            self.basis = torch.as_tensor(basis, device=device)
        else:
            self.basis = None

    def forward(self, y):
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

        if self.device is None:
            self.device = y.device

        # cast
        y = y.to(self.device)
        if self.mask is not None:
            self.mask = self.mask.to(self.device)
        if self.basis is not None:
            self.basis = self.basis.to(self.device)

        # mask if required
        if self.mask is not None:
            y = self.mask * y

        # project
        if self.basis is not None:
            y = y[..., None]
            y = y.swapaxes(-4, -1)
            yshape = list(y.shape)
            y = y.reshape(-1, y.shape[-1])  # (prod(y.shape[:-1]), ncoeff)
            y = y @ self.basis  # (prod(y.shape[:-1]), ncontrasts)
            y = y.reshape(*yshape[:-1], y.shape[-1])
            y = y.swapaxes(-4, -1)
            y = y[..., 0]

        # apply Fourier transform
        x = _fft.ifft(y, axes=(-1, -2), norm="ortho")

        # cast back to numpy if required
        if isnumpy:
            x = x.numpy(force=True)

        return x

    def _adjoint_linop(self):
        # get adjoint basis
        if self.basis is not None:
            basis_adjoint = self.basis.conj().t()
        else:
            basis_adjoint = None
        return FFTOp(self.mask, basis_adjoint, self.device)


class FFTGramOp(base.Linop):
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

    def __init__(self, mask=None, basis=None, device=None, **kwargs):
        super().__init__(ndim=2, **kwargs)
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
            assert nt == T
            tmp = mask.permute(2, 1, 0).reshape((ny, nz, T, 1, 1)) * basis.reshape(
                (1, 1, nt, 1, K)
            )  # (ny, nz, T, 1, K) / (nx, ny, T, 1, K)
            toeplitz_kern = (tmp * basis.reshape(1, 1, T, K, 1)).sum(
                axis=2
            )  # (ny, nz, K, K) / (nx, ny, K, K)
            toeplitz_kern = torch.fft.fftshift(
                torch.fft.fftshift(toeplitz_kern, axis=0), axis=1
            )
            self._toeplitz_kern = (
                toeplitz_kern.swapaxes(0, 1).reshape(-1, K, K).contiguous()
            )  # (nz*ny, K, K) / (ny*nx, K, K)
        else:
            self._toeplitz_kern = None

    def forward(self, x):
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
        y = _fft.fft(x, axes=(-1, -2), norm="ortho", centered=False)

        # project if required
        if self._toeplitz_kern is not None:
            y = y[..., None]  # (..., ncoeff, nz, ny, 1) / (..., ncoeff, ny, nx, 1)
            y = y.swapaxes(
                -4, -1
            )  # (..., 1, nz, ny, ncoeff) / (..., 1, ny, nx, ncoeff)
            yshape = list(y.shape)
            y = y.reshape(
                int(np.prod(yshape[:-4])), -1, y.shape[-1]
            )  # (prod(y.shape[:-4]), nz*ny, ncoeff) / (prod(y.shape[:-4]), ny*nx, ncoeff)
            y = torch.einsum("...bi,bij->...bj", y, self._toeplitz_kern)
            y = y.reshape(
                *yshape
            )  # (..., 1, nz, ny, ncoeff) / # (..., 1, ny, nx, ncoeff)
            y = y.swapaxes(
                -4, -1
            )  # (..., ncoeff, nz, ny, 1) / # (..., ncoeff, ny, nx, 1)
            y = y[..., 0]  # (..., ncoeff, nz, ny) / # (..., ncoeff, ny, nx)

        # apply Fourier transform
        x = _fft.ifft(y, axes=(-1, -2), norm="ortho", centered=False)

        # cast back to numpy if required
        if isnumpy:
            x = x.numpy(force=True)

        return x

    def _adjoint_linop(self):
        return self
