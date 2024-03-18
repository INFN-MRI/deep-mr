"""Sparse Fast Fourier Transform linear operator."""

__all__ = ["SparseFFTOp", "SparseIFFTOp", "SparseFFTGramOp"]

import torch

from .. import fft as _fft

from . import base


class SparseFFTOp(base.Linop):
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

    def __init__(
        self,
        indexes=None,
        shape=None,
        basis_adjoint=None,
        weight=None,
        device="cpu",
        threadsperblock=128,
    ):
        if indexes is not None and shape is not None:
            super().__init__(ndim=indexes.shape[-1])
            self.sampling = _fft.prepare_sampling(indexes, shape, device)
        else:
            super().__init__(ndim=None)
            self.sampling = None
        if weight is not None:
            self.weight = torch.as_tensor(weight**0.5, device=device)
        else:
            self.weight = None
        if basis_adjoint is not None:
            self.basis_adjoint = torch.as_tensor(basis_adjoint, device=device)
        else:
            self.basis_adjoint = None
        self.threadsperblock = threadsperblock

    def forward(self, x):
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
        return _fft.apply_sparse_fft(
            x,
            self.sampling,
            self.basis_adjoint,
            self.weight,
            threadsperblock=self.threadsperblock,
        )

    def _adjoint_linop(self):
        if self.basis_adjoint is not None:
            basis = self.basis_adjoint.conj().t()
        else:
            basis = None
        adjOp = SparseIFFTOp(
            basis=basis, weight=self.weight, threadsperblock=self.threadsperblock
        )
        adjOp.ndim = self.ndim
        adjOp.sampling = self.sampling
        return adjOp


class SparseIFFTOp(base.Linop):
    """
    Inverse sparse Fast Fourier Transform operator.

    K-space sampling locations are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.

    Input k-space data are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    Similarly, output images are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``.

    """

    def __init__(
        self,
        indexes=None,
        shape=None,
        basis=None,
        weight=None,
        device="cpu",
        threadsperblock=128,
    ):
        if indexes is not None and shape is not None:
            super().__init__(ndim=indexes.shape[-1])
            self.sampling = _fft.prepare_sampling(indexes, shape, device)
        else:
            super().__init__(ndim=None)
            self.sampling = None
        if weight is not None:
            self.weight = torch.as_tensor(weight**0.5, device=device)
        else:
            self.weight = None
        if basis is not None:
            self.basis = torch.as_tensor(basis, device=device)
        else:
            self.basis = None
        self.threadsperblock = threadsperblock

    def forward(self, y):
        """
        Apply inverse Sparse Fast Fourier Transform.

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
        return _fft.apply_sparse_ifft(
            y,
            self.sampling,
            self.basis,
            self.weight,
            threadsperblock=self.threadsperblock,
        )

    def _adjoint_linop(self):
        if self.basis is not None:
            basis_adjoint = self.basis.conj().t()
        else:
            basis_adjoint = None
        adjOp = SparseFFTOp(
            basis_adjoint=basis_adjoint,
            weight=self.weight,
            threadsperblock=self.threadsperblock,
        )
        adjOp.ndim = self.ndim
        adjOp.sampling = self.sampling
        return adjOp


class SparseFFTGramOp(base.Linop):
    """
    Self-adjoint Sparse Fast Fourier Transform operator.

    K-space sampling locations are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.

    Input and output data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``,
    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(
        self,
        indexes,
        shape,
        basis=None,
        weight=None,
        device="cpu",
        threadsperblock=128,
        **kwargs
    ):
        super().__init__(ndim=indexes.shape[-1], **kwargs)
        self.toeplitz_kern = _fft.plan_toeplitz_fft(
            indexes, shape, basis, weight, device
        )
        self.threadsperblock = threadsperblock

    def forward(self, x):
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
        return _fft.apply_sparse_fft_selfadj(
            x, self.toeplitz_kern, threadsperblock=self.threadsperblock
        )

    def _adjoint_linop(self):
        return self
