"""Deepinv + Lighning mixin."""

__all__ = ["LightningDenoiser"]

from torch import nn

import lightning as pl

import deepinv as dinv


class LightningDenoiser(dinv.models.Denoiser, pl.LightningModule):
    """
    Base class for denoiser models using PyTorch Lightning, inheriting from `deepinv.models.Denoiser`.

    This class encapsulates common functionality like forward pass, and handles
    integration with PyTorch Lightning as well as the sigma parameter for denoising.

    Attributes
    ----------
    model: nn.Module
        Backbone model.
    device : str, optional
        Computational device. The default is ``"cpu"``


    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        dinv.models.Denoiser.__init__(
            self, device=device
        )  # Initialize deepinv.models.Denoiser
        pl.LightningModule.__init__(self)
        self.model = model.to(device)
