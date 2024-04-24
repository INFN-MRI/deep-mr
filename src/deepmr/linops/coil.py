"""Sensitivity coil linear operator."""

__all__ = ["SoftSenseOp", "SenseAdjointOp"]

from . import base

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

        # offloat to device
        self.sensmap = sensmap.to(self.device)
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
            Input combined images of shape ``(nslices, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(..., nz, ny, nx)`` (3D NonCartesian MRI).

        Returns
        -------
        y : torch.Tensor
            Output images of shape ``(nslices, nsets, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian) or ``(nsets, ncoils, ..., nz, ny, nx)`` (3D NonCartesian MRI)
            modulated by coil sensitivity profiles.

        """
        # get device
        if self.device is None:
            device = x.device
        else:
            device = self.device
            
        # transfer to device
        odevice = x.device
        self.x = self.x.to(device)
        self.sensmap = self.sensmap.to(device)

        # collapse
        if self.batchmode:
            oshape = x.shape
            if self.ndim == 2 and self.multislice:
                batchshape = x.shape[2:-self.ndim]      
            else:
                batchshape = x.shape[:-self.ndim]     
                
        
        if self.batchmode:
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

    * 2D MRI: ``(nslices, nsets, ncoils, ny, nx)``
    * 3D MRI: ``(nsets, ncoils, nz, ny, nx)``

    where ``nsets`` represents multiple sets of coil sensitivity estimation
    for soft-SENSE implementations (e.g., ESPIRIT), equal to ``1`` for conventional SENSE
    and ``ncoils`` represents the number of receiver channels in the coil array.
    For standard SENSE, ``nsets`` axis can be omitted:
        
    * 2D MRI: ``(nslices, ncoils, ny, nx)``
    * 3D Cartesian MRI: ``(nx, ncoils, nz, ny)``
    * 3D NonCartesian MRI: ``(ncoils, nz, ny, nx)``

    """

    def __init__(self, ndim, sensmap, device=None, batchmode=True):
        super().__init__(ndim)

        # keep number of spatial dimensions
        self.ndim = ndim

        # assign device
        if device is None:
            self.device = self.sensmap.device
        else:
            self.device = device

        # offloat to device
        self.sensmap = sensmap.to(self.device)
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
            Input images of shape ``(nsets, nslices, ncoils, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(nsets, ncoils, ..., nz, ny, nx)``
            (3D NonCartesian MRI) modulated by coil sensitivity profiles.

        Returns
        -------
        x : torch.Tensor
            Output combined images of shape ``(nslices, ..., ny, nx)``.
            (2D MRI / 3D Cartesian MRI) or ``(..., nz, ny, nx)`` (3D NonCartesian MRI).

        """
        # get device
        if self.device is None:
            device = y.device
        else:
            device = self.device
            
        # transfer to device
        odevice = y.device
        self.y = self.y.to(device)
        self.sensmap = self.sensmap.to(device)

        # apply sensitivity
        tmp = self.sensmap.conj() * y

        # combine  (over channels and sets)
        if self.batchmode:
            if self.ndim == 2:
                x = tmp.sum(axis=-4).sum(axis=0)
            elif self.ndim == 3:
                x = tmp.sum(axis=-5).sum(axis=0)
        else:
            if self.ndim == 2:
                x = tmp.sum(axis=-3).sum(axis=0)
            elif self.ndim == 3:
                x = tmp.sum(axis=-4).sum(axis=0)

        return x.to(odevice)

    def _adjoint_linop(self):
        if self.batchmode and self.ndim == 2:
            sensmap = self.sensmap.squeeze(-3)
        if self.batchmode and self.ndim == 3:
            sensmap = self.sensmap.squeeze(-4)
        if self.batchmode is False:
            sensmap = self.sensmap
        return SenseOp(self.ndim, sensmap, self.batchmode, self.device)
