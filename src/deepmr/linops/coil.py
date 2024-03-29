"""Sensitivity coil linear operator."""

__all__ = ["SenseOp", "SenseAdjointOp"]

import numpy as np
import torch

from . import base


class SenseOp(base.Linop):
    """
    Multiply input image by coil sensitivity profile.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nsets, nslices, ncoils, ny, nx)``
    * 3D Cartesian MRI: ``(nsets, nx, ncoils, nz, ny)``
    * 3D NonCartesian MRI: ``(nsets, ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(self, ndim, sensmap, device=None, multicontrast=True):
        super().__init__(ndim)

        # cast map to tensor
        self.sensmap = torch.as_tensor(sensmap)

        # assign device
        if device is None:
            self.device = self.sensmap.device
        else:
            self.device = device

        # offloat to device
        self.sensmap = self.sensmap.to(self.device)

        # multicontrast
        self.multicontrast = multicontrast

        if self.multicontrast and self.ndim == 2:
            self.sensmap = self.sensmap.unsqueeze(-3)
        if self.multicontrast and self.ndim == 3:
            self.sensmap = self.sensmap.unsqueeze(-4)

    def forward(self, x):
        """
        Forward coil operator.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input combined images of shape ``(nslices, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(..., nz, ny, nx)`` (3D NonCartesian MRI).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output images of shape ``(nsets, nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian) or ``(nsets, ncoils, ..., nz, ny, nx)`` (3D NonCartesian MRI)
            modulated by coil sensitivity profiles.

        """
        if isinstance(x, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False

        # convert to tensor
        x = torch.as_tensor(x)

        # transfer to device
        self.sensmap = self.sensmap.to(x.device)

        # unsqueeze
        if self.multicontrast:
            if self.ndim == 2:
                x = x.unsqueeze(-4)
            elif self.ndim == 3:
                x = x.unsqueeze(-5)
        else:
            if self.ndim == 2:
                x = x.unsqueeze(-3)
            elif self.ndim == 3:
                x = x.unsqueeze(-4)

        # project
        y = self.sensmap * x

        # convert back to numpy if required
        if isnumpy:
            y = y.numpy(force=True)

        return y

    def _adjoint_linop(self):
        if self.multicontrast and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.multicontrast and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.multicontrast is False:
            sensmap = self.sensmap
        return SenseAdjointOp(self.ndim, sensmap, self.device, self.multicontrast)


class SenseAdjointOp(base.Linop):
    """
    Perform coil combination.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(self, ndim, sensmap, device=None, multicontrast=True):
        super().__init__(ndim)

        # cast map to tensor
        self.sensmap = torch.as_tensor(sensmap)

        # assign device
        if device is None:
            self.device = self.sensmap.device
        else:
            self.device = device

        # offloat to device
        self.sensmap = self.sensmap.to(self.device)

        # multicontrast
        self.multicontrast = multicontrast

        if self.multicontrast and self.ndim == 2:
            self.sensmap = self.sensmap.unsqueeze(-3)
        if self.multicontrast and self.ndim == 3:
            self.sensmap = self.sensmap.unsqueeze(-4)

    def forward(self, y):
        """
        Adjoint coil operator (coil combination).

        Parameters
        ----------
        y : np.ndarray | torch.Tensor
            Output images of shape ``(nsets, nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(nsets, ncoils, ..., nz, ny, nx)``
            (3D NonCartesian MRI) modulated by coil sensitivity profiles.

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output combined images of shape ``(nslices, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(..., nz, ny, nx)`` (3D NonCartesian MRI).

        """
        if isinstance(y, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False

        # convert to tensor
        y = torch.as_tensor(y)

        # transfer to device
        self.sensmap = self.sensmap.to(y.device)

        # apply sensitivity
        tmp = self.sensmap.conj() * y

        # combine  (over channels and sets)
        if self.multicontrast:
            if self.ndim == 2:
                x = tmp.sum(axis=-4).sum(axis=0)
            elif self.ndim == 3:
                x = tmp.sum(axis=-5).sum(axis=0)
        else:
            if self.ndim == 2:
                x = tmp.sum(axis=-3).sum(axis=0)
            elif self.ndim == 3:
                x = tmp.sum(axis=-4).sum(axis=0)

        # convert back to numpy if required
        if isnumpy:
            x = x.numpy(force=True)

        return x

    def _adjoint_linop(self):
        if self.multicontrast and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.multicontrast and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.multicontrast is False:
            sensmap = self.sensmap
        return SenseOp(self.ndim, sensmap, self.device, self.multicontrast)
