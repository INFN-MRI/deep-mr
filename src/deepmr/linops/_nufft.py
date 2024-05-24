"""Non-Uniform Fast Fourier Transform linear operator."""

__all__ = ["NUFFTOp", "NUFFTAdjointOp", "NUFFTGramOp"]

import torch

from .. import fft as _fft

from . import _base as base


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

    def __init__(
        self,
        coord=None,
        shape=None,
        basis_adjoint=None,
        weight=None,
        device="cpu",
        threadsperblock=128,
        width=4,
        oversamp=1.25,
    ):
        if coord is not None and shape is not None:
            super().__init__()
            self.nufft_plan = _fft.plan_nufft(coord, shape, width, oversamp, device)
        else:
            super().__init__()
            self.nufft_plan = None
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
        return _fft.apply_nufft(
            x,
            self.nufft_plan,
            self.basis_adjoint,
            self.weight,
            threadsperblock=self.threadsperblock,
            norm="ortho",
        )

    def _adjoint_linop(self):
        if self.basis_adjoint is not None:
            basis = self.basis_adjoint.conj().T
        else:
            basis = None
        if self.weight is not None:
            weight = self.weight**2
        else:
            weight = None
        adjOp = NUFFTAdjointOp(
            basis=basis, weight=weight, threadsperblock=self.threadsperblock
        )
        adjOp.nufft_plan = self.nufft_plan
        return adjOp


class NUFFTAdjointOp(base.Linop):
    """
    Adjoint Non-Uniform Fast Fourier Transform operator.

    K-space sampling trajectory are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.

    Input k-psace data are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ncontrasts, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ncontrasts, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    Similarly, output images data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``.

    """

    def __init__(
        self,
        coord=None,
        shape=None,
        basis=None,
        weight=None,
        device="cpu",
        threadsperblock=128,
        width=4,
        oversamp=1.25,
    ):
        if coord is not None and shape is not None:
            super().__init__()
            self.nufft_plan = _fft.plan_nufft(coord, shape, width, oversamp, device)
        else:
            super().__init__()
            self.nufft_plan = None
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
        return _fft.apply_nufft_adj(
            y,
            self.nufft_plan,
            self.basis,
            self.weight,
            threadsperblock=self.threadsperblock,
            norm="ortho",
        )

    def _adjoint_linop(self):
        if self.basis is not None:
            basis_adjoint = self.basis.conj().T
        else:
            basis_adjoint = None
        if self.weight is not None:
            weight = self.weight**2
        else:
            weight = None
        adjOp = NUFFTOp(
            basis_adjoint=basis_adjoint,
            weight=weight,
            threadsperblock=self.threadsperblock,
        )
        adjOp.nufft_plan = self.nufft_plan
        return adjOp


class NUFFTGramOp(base.Linop):
    """
    Self-adjoint Non-Uniform Fast Fourier Transform operator.

    K-space sampling trajectory are expected to be shaped ``(ncontrasts, nviews, nsamples, ndims)``.

    Input and output data are expected to be shaped ``(nslices, nsets, ncoils, ncontrasts, nviews, nsamples)``,
    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(
        self,
        coord,
        shape,
        basis=None,
        weight=None,
        device="cpu",
        threadsperblock=128,
        width=6,
    ):
        super().__init__()
        self.toeplitz_kern = _fft.plan_toeplitz_nufft(
            coord, shape, basis, weight, width, device
        )
        self.threadsperblock = threadsperblock

    def forward(self, x):
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
        return _fft.apply_nufft_selfadj(
            x, self.toeplitz_kern, threadsperblock=self.threadsperblock
        )

    def _adjoint_linop(self):
        return self
