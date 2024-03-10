"""Sensitivity coil linear operator."""

__all__ = ["CoilOp"]

import numpy as np
import torch

from . import base


class CoilOp(base.Linop):
    """
    Multiply input image by coil sensitivity profile.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    Similarly, input images are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ..., ny, nx)``
    * 3D MRI: ``(nsets, ncoils, ..., nz, ny, nx)``

    where the inner axes (i.e., ``...``) represents other dimensions such
    as time frames (e.g., dynamic acquisitions) or multiple contrasts
    (e.g., variable flip angle, multi-echo).

    """

    def __init__(self, ndim, sensmap, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self._ndim = ndim
        self._sensmap = torch.as_tensor(sensmap, device=device)

    def A(self, x):
        """
        Forward coil operator.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input combined images of shape ``(nslices, 1, 1, ..., ny, nx)``.
            (2D MRI) or ``(1, 1, ..., nz, ny, nx)`` (3D MRI).

        Returns
        -------
        y : np.ndarray | torch.Tensor
            Output images of shape ``(nslices, nsets, ncoils, ..., ny, nx)``.
            (2D MRI) or ``(nsets, ncoils, ..., nz, ny, nx)`` (3D MRI) modulated
            by coil sensitivity profiles.

        """
        if isinstance(x, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False

        # convert to tensor
        x = torch.as_tensor(x)

        # transfer to device
        self._sensmap = self._sensmap.to(x.device)

        # expand input dim
        naxis = len(x.shape)

        if self._ndim == 2:
            while len(self._sensmap.shape) < naxis:
                self._sensmap = self._sensmap[:, :, :, None, ...]
        if self._ndim == 3:
            while len(self._sensmap.shape) < naxis:
                self._sensmap = self._sensmap[:, :, None, ...]

        # project
        y = self._sensmap * x

        # convert back to numpy if required
        if isnumpy:
            y = y.numpy(force=True)

        return y

    def A_adjoint(self, y):
        """
        Adjoint coil operator (coil combination).

        Parameters
        ----------
        y : np.ndarray | torch.Tensor
            Output images of shape ``(nslices, nsets, ncoils, ..., ny, nx)``.
            (2D MRI) or ``(nsets, ncoils, ..., nz, ny, nx)`` (3D MRI) modulated
            by coil sensitivity profiles.

        Returns
        -------
        x : np.ndarray | torch.Tensor
            Output combined images of shape ``(nslices, 1, 1, ..., ny, nx)``.
            (2D MRI) or ``(1, 1, ..., nz, ny, nx)`` (3D MRI).

        """
        if isinstance(y, np.ndarray):
            isnumpy = True
        else:
            isnumpy = False

        # convert to tensor
        y = torch.as_tensor(y)

        # expand input dim
        naxis = len(y.shape)

        if self._ndim == 2:
            while len(self._sensmap.shape) < naxis:
                self._sensmap = self._sensmap[:, :, :, None, ...]
        if self._ndim == 3:
            while len(self._sensmap.shape) < naxis:
                self._sensmap = self._sensmap[:, :, None, ...]

        # combine
        tmp = self._sensmap.conj() * y
        if self._ndim == 2:
            x = tmp.sum(axis=1, keepdim=True).sum(
                axis=2, keepdim=True
            )  # sum over sets and channels
        if self._ndim == 2:
            x = tmp.sum(axis=0, keepdim=True).sum(
                axis=1, keepdim=True
            )  # sum over sets and channels

        # convert back to numpy if required
        if isnumpy:
            x = x.numpy(force=True)

        return x
