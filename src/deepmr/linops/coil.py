"""Sensitivity coil linear operator."""

__all__ = ["SenseOp", "SenseAdjointOp", "SoftSenseOp", "SoftSenseAdjointOp"]

from . import base


class SenseOp(base.Linop):
    """
    Multiply input image by coil sensitivity profile.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nslices, ncoils, ny, nx)``
    * 3D Cartesian MRI: ``(nx, ncoils, nz, ny)``
    * 3D NonCartesian MRI: ``(ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(self, ndim, sensmap, batchmode=True, device=None):
        super().__init__()

        # keep number of spatial dimensions
        self.ndim = ndim

        # assign device
        self.device = device

        # offload to device
        self.sensmap = sensmap.to(device=self.device)
        self.sensmap = self.sensmap.squeeze()

        # get sensmap shape
        if self.ndim == 2:
            if len(self.sensmap.shape) == 4:
                self.multislice = True
            else:
                self.multislice = False

        # multicontrast
        self.batchmode = batchmode

        if self.batchmode and self.ndim == 2:
            self.sensmap = self.sensmap.unsqueeze(-3)
        if self.batchmode and self.ndim == 3:
            self.sensmap = self.sensmap.unsqueeze(-4)

    def forward(self, x):
        """
        Forward coil operator.

        Parameters
        ----------
        x : torch.Tensor
            Input combined images of shape ``(nslices, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(..., nz, ny, nx)`` (3D NonCartesian MRI).
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(..., ny, nx)``)

        Returns
        -------
        y : torch.Tensor
            Output images of shape ``(nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian) or ``(ncoils, ..., nz, ny, nx)`` (3D NonCartesian MRI)
            modulated by coil sensitivity profiles.
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(ncoils, ..., ny, nx)``)

        """
        # get device
        if self.device is None:
            device = x.device
        else:
            device = self.device

        # transfer to device
        odevice = x.device
        x = x.to(device=device)
        self.sensmap = self.sensmap.to(device=device)
        # collapse
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                batchshape = x.shape[1 : -self.ndim]
                x = x.reshape(
                    x.shape[0], 1, -1, *x.shape[-self.ndim :]
                )  # (nslices, 1, prod(batchshape), ny, nx)
            else:
                batchshape = x.shape[: -self.ndim]
                x = x.reshape(
                    1, -1, *x.shape[-self.ndim :]
                )  # (1, prod(batchshape), *x.shape[-ndim:])

        # project
        y = (
            self.sensmap * x
        )  # (ncoils, prod(batchshape), *x.shape[-ndim:]) or # (nslices, ncoils, prod(batchshape), ny, nx)

        # reshape back
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                y = y.reshape(
                    *y.shape[:2], *batchshape, *y.shape[-self.ndim :]
                )  # (nslices, ncoils, *batchshape, ny, nx)
            else:
                y = y.reshape(
                    y.shape[0], *batchshape, *y.shape[-self.ndim :]
                )  # (ncoils, *batchshape, *x.shape[-ndim:])

        return y.to(odevice)

    def _adjoint_linop(self):
        if self.batchmode and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.batchmode and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.batchmode is False:
            sensmap = self.sensmap
        return SenseAdjointOp(self.ndim, sensmap, self.batchmode, self.device)


class SenseAdjointOp(base.Linop):
    """
    Perform coil combination.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nslices, ncoils, ny, nx)``
    * 3D Cartesian MRI: ``(nx, ncoils, nz, ny)``
    * 3D NonCartesian MRI: ``(ncoils, nz, ny, nx)``

    where ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(self, ndim, sensmap, batchmode=True, device=None):
        super().__init__()

        # keep number of spatial dimensions
        self.ndim = ndim

        # assign device
        self.device = device

        # offload to device
        self.sensmap = sensmap.to(device=self.device)
        self.sensmap = self.sensmap.squeeze()

        # get sensmap shape
        if self.ndim == 2:
            if len(self.sensmap.shape) == 4:
                self.multislice = True
            else:
                self.multislice = False

        # multicontrast
        self.batchmode = batchmode

        if self.batchmode and self.ndim == 2:
            self.sensmap = self.sensmap.unsqueeze(-3)
        if self.batchmode and self.ndim == 3:
            self.sensmap = self.sensmap.unsqueeze(-4)

    def forward(self, y):
        """
        Adjoint coil operator (coil combination).

        Parameters
        ----------
        y : np.torch.Tensor
            Input images of shape ``(nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(ncoils, ..., nz, ny, nx)``
            (3D NonCartesian MRI) modulated by coil sensitivity profiles.
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(..., ny, nx)``)

        Returns
        -------
        x : torch.Tensor
            Output combined images of shape ``(nslices, nsets, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(nsets, ..., nz, ny, nx)`` (3D NonCartesian MRI).
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(nsets, ..., ny, nx)``)

        """
        # get device
        if self.device is None:
            device = y.device
        else:
            device = self.device

        # transfer to device
        odevice = y.device
        y = y.to(device=device)
        self.sensmap = self.sensmap.to(device=device)

        # collapse
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                batchshape = y.shape[2 : -self.ndim]
                y = y.reshape(
                    *y.shape[:2], -1, *y.shape[-self.ndim :]
                )  # (nslices, ncoils, prod(batchshape), ny, nx)
            else:
                batchshape = y.shape[1 : -self.ndim]
                y = y.reshape(
                    y.shape[0], -1, *y.shape[-self.ndim :]
                )  # (ncoils, prod(batchshape), *x.shape[-ndim:])

        # project and sum over coils
        x = (
            self.sensmap.conj() * y
        )  # (ncoils, prod(batchshape), *y.shape[-ndim:]) or # (nslices, ncoils, prod(batchshape), ny, nx)
        if self.batchmode:
            x = x.sum(
                axis=-self.ndim - 2
            )  # (prod(batchshape), *y.shape[-ndim:]) or (nslices, prod(batchshape), ny, nx)
        else:
            x = x.sum(
                axis=-self.ndim - 1
            )  # (*y.shape[-ndim:]) or (nslices, ny, nx)
            
        # reshape back
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                x = x.reshape(
                    x.shape[0], *batchshape, *x.shape[-self.ndim :]
                )  # (nslices, *batchshape, ny, nx)
            else:
                x = x.reshape(
                    *batchshape, *x.shape[-self.ndim :]
                )  # (*batchshape, *x.shape[-ndim:])

        return x.to(odevice)

    def _adjoint_linop(self):
        if self.batchmode and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.batchmode and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.batchmode is False:
            sensmap = self.sensmap
        return SenseOp(self.ndim, sensmap, self.batchmode, self.device)


class SoftSenseOp(base.Linop):
    """
    Multiply input image by coil sensitivity profile.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ny, nx)``
    * 3D Cartesian MRI: ``(nx, nsets, ncoils, nz, ny)``
    * 3D NonCartesian MRI: ``(nsets, ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(self, ndim, sensmap, batchmode=True, device=None):
        super().__init__()

        # keep number of spatial dimensions
        self.ndim = ndim

        # assign device
        self.device = device

        # offload to device
        self.sensmap = sensmap.to(device=self.device)
        self.sensmap = self.sensmap.squeeze()

        # get sensmap shape
        if self.ndim == 2:
            if len(self.sensmap.shape) == 5:
                self.multislice = True
            else:
                self.multislice = False

        # multicontrast
        self.batchmode = batchmode

        if self.batchmode and self.ndim == 2:
            self.sensmap = self.sensmap.unsqueeze(-3)
        if self.batchmode and self.ndim == 3:
            self.sensmap = self.sensmap.unsqueeze(-4)

    def forward(self, x):
        """
        Forward coil operator.

        Parameters
        ----------
        x : torch.Tensor
            Input combined images of shape ``(nslices, nsets, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(nsets, ..., nz, ny, nx)`` (3D NonCartesian MRI).
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(nsets, ..., ny, nx)``)

        Returns
        -------
        y : torch.Tensor
            Output images of shape ``(nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian) or ``(ncoils, ..., nz, ny, nx)`` (3D NonCartesian MRI)
            modulated by coil sensitivity profiles.
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(ncoils, ..., ny, nx)``)

        """
        # get device
        if self.device is None:
            device = x.device
        else:
            device = self.device

        # transfer to device
        odevice = x.device
        x = x.to(device=device)
        self.sensmap = self.sensmap.to(device=device)

        # collapse
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                batchshape = x.shape[2 : -self.ndim]
                x = x.reshape(
                    *x.shape[:2], 1, -1, *x.shape[-self.ndim :]
                )  # (nslices, nsets, 1, prod(batchshape), ny, nx)
            else:
                batchshape = x.shape[1 : -self.ndim]
                x = x.reshape(
                    x.shape[0], 1, -1, *x.shape[-self.ndim :]
                )  # (nsets, 1, prod(batchshape), *x.shape[-ndim:])

        # project and sum over sets
        y = (
            self.sensmap * x
        )  # (nsets, ncoils, prod(batchshape), *x.shape[-ndim:]) or # (nslices, nsets, ncoils, prod(batchshape), ny, nx)
        if self.batchmode:
            y = y.sum(
                axis=-self.ndim - 3
            )  # (ncoils, prod(batchshape), *x.shape[-ndim:]) or (nslices, ncoils, prod(batchshape), ny, nx)
        else:
            y = y.sum(
                axis=-self.ndim - 2
            )  # (ncoils, *x.shape[-ndim:]) or (nslices, ncoils, ny, nx)

        # reshape back
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                y = y.reshape(
                    *y.shape[:2], *batchshape, *y.shape[-self.ndim :]
                )  # (nslices, ncoils, *batchshape, ny, nx)
            else:
                y = y.reshape(
                    y.shape[0], *batchshape, *y.shape[-self.ndim :]
                )  # (ncoils, *batchshape, *x.shape[-ndim:])

        return y.to(odevice)

    def _adjoint_linop(self):
        if self.batchmode and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.batchmode and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.batchmode is False:
            sensmap = self.sensmap
        return SoftSenseAdjointOp(self.ndim, sensmap, self.batchmode, self.device)


class SoftSenseAdjointOp(base.Linop):
    """
    Perform coil combination.

    Coil sensitivity profiles are expected to have the following dimensions:

    * 2D MRI: ``(nslices, nsets, ncoils, ny, nx)``
    * 3D Cartesian MRI: ``(nx, nsets, ncoils, nz, ny)``
    * 3D NonCartesian MRI: ``(nsets, ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.

    """

    def __init__(self, ndim, sensmap, batchmode=True, device=None):
        super().__init__()

        # keep number of spatial dimensions
        self.ndim = ndim

        # assign device
        self.device = device

        # offload to device
        self.sensmap = sensmap.to(device=self.device)
        self.sensmap = self.sensmap.squeeze()

        # get sensmap shape
        if self.ndim == 2:
            if len(self.sensmap.shape) == 5:
                self.multislice = True
            else:
                self.multislice = False

        # multicontrast
        self.batchmode = batchmode

        if self.batchmode and self.ndim == 2:
            self.sensmap = self.sensmap.unsqueeze(-3)
        if self.batchmode and self.ndim == 3:
            self.sensmap = self.sensmap.unsqueeze(-4)

    def forward(self, y):
        """
        Adjoint coil operator (coil combination).

        Parameters
        ----------
        y : np.torch.Tensor
            Input images of shape ``(nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(ncoils, ..., nz, ny, nx)``
            (3D NonCartesian MRI) modulated by coil sensitivity profiles.
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(..., ny, nx)``)

        Returns
        -------
        x : torch.Tensor
            Output combined images of shape ``(nslices, nsets, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(nsets, ..., nz, ny, nx)`` (3D NonCartesian MRI).
            For single-slice 2D data, ``slices`` axis can be omitted (i.e., ``(nsets, ..., ny, nx)``)

        """
        # get device
        if self.device is None:
            device = y.device
        else:
            device = self.device

        # transfer to device
        odevice = y.device
        y = y.to(device=device)
        self.sensmap = self.sensmap.to(device=device)

        # collapse
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                batchshape = y.shape[2 : -self.ndim]
                y = y.reshape(
                    y.shape[0], 1, y.shape[1], -1, *y.shape[-self.ndim :]
                )  # (nslices, 1, ncoils, prod(batchshape), ny, nx)
            else:
                batchshape = y.shape[1 : -self.ndim]
                y = y.reshape(
                    1, y.shape[0], -1, *y.shape[-self.ndim :]
                )  # (1, ncoils, prod(batchshape), *x.shape[-ndim:])

        # project and sum over coils
        x = (
            self.sensmap.conj() * y
        )  # (nsets, ncoils, prod(batchshape), *y.shape[-ndim:]) or # (nslices, nsets, ncoils, prod(batchshape), ny, nx)
        if self.batchmode:
            x = x.sum(
                axis=-self.ndim - 2
            )  # (nsets, prod(batchshape), *y.shape[-ndim:]) or (nslices, nsets, prod(batchshape), ny, nx)
        else:
            x = x.sum(
                axis=-self.ndim - 1
            )  # (nsets, *y.shape[-ndim:]) or (nslices, nsets, ny, nx)
            

        # reshape back
        if self.batchmode:
            if self.ndim == 2 and self.multislice:
                x = x.reshape(
                    *x.shape[:2], *batchshape, *x.shape[-self.ndim :]
                )  # (nslices, nsets, *batchshape, ny, nx)
            else:
                x = x.reshape(
                    x.shape[0], *batchshape, *x.shape[-self.ndim :]
                )  # (nsets, *batchshape, *x.shape[-ndim:])

        return x.to(odevice)

    def _adjoint_linop(self):
        if self.batchmode and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.batchmode and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.batchmode is False:
            sensmap = self.sensmap
        return SoftSenseOp(self.ndim, sensmap, self.batchmode, self.device)
