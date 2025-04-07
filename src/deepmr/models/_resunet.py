"""ResUNet denoisers."""

__all__ = ["ProxResUnet", "ResUNetDenoiser", "LitResUNet2D", "LitResUNet3D"]


import numpy as np

import torch
import torch.nn as nn
import mrops._sigpy as sp

from mrinufft._array_compat import with_torch
from ._base import LightningDenoiser

# Model parameters (TODO: make it selectable)
nn_kernel = 3
nn_block_size = 64
nn_inf_block_size = 64
overlap_fract = 1 / 4


class ProxResUnet(sp.prox.Prox):
    """
    ResUNet proximal operator for PNP optimization.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input shape.
    ndim : int
        Number of dimensions (2 or 3).
    checkpoint : str, optional
        Path to pre-trained weights.
    device : str | None, optional
        Computational device
    batch_size : int, optional
        Batch size. The default is ``2`'.
    verbose : bool, optional
        Toggle verbosity. The default is ``False`` (silent mode).

    """

    def __init__(
        self,
        shape: list[int] | tuple[int],
        ndim: int,
        checkpoint: str,
        device: int | None = None,
        batch_size: int = 2,
        verbose: bool = False,
    ):
        super().__init__(shape)
        self.denoiser = ResUNetDenoiser(ndim, checkpoint, device, batch_size, verbose)

    def _prox(self, alpha, input):
        return self.denoiser(input)


class ResUNetDenoiser:
    """
    ResUNet Denoiser.

    Parameters
    ----------
    ndim : int
        Number of dimensions (2 or 3).
    checkpoint : str, optional
        Path to pre-trained weights.
    device : str | None, optional
        Computational device
    batch_size : int, optional
        Batch size. The default is ``2`'.
    verbose : bool, optional
        Toggle verbosity. The default is ``False`` (silent mode).

    """

    def __init__(
        self,
        ndim: int,
        checkpoint: str,
        device: int | None = None,
        batch_size: int = 2,
        verbose: bool = False,
    ):
        assert ndim in [2, 3], "ndim must be '3' or '3'"
        self.ndim = ndim
        self.model = LitResUNet3D() if ndim == 3 else LitResUNet2D()

        # Load checkpoint if provided
        if checkpoint:
            self.model.load_from_checkpoint(checkpoint)

        # Device setup
        self.device_type = (
            device
            if device
            else (
                f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu"
            )
        )
        self.model.to(self.device_type)
        self.model.freeze()

        self._batch_size = batch_size
        self._verbose = verbose

    def __call__(self, x):
        """Override __call__ to direct the call to forward or inference."""
        return self._infer(x) if self.is_infer else self.model(x)

    def create_blend_mask(self, device, shape, overlap):
        """Create blend mask for overlapping inference blocks."""
        xp = device.xp
        assert len(shape) in [2, 3]
        with device:
            M = xp.ones(shape, dtype=xp.float32)
            rise = lambda n: xp.linspace(0, 1, int(n))
            fall = lambda n: xp.linspace(1, 0, int(n))

            for axis, size, ov in zip(range(len(shape)), shape, overlap):
                if ov == 0:
                    continue
                blend = rise(ov)
                inv_blend = fall(ov)
                slices = [slice(None)] * len(shape)

                # Front
                slices[axis] = slice(0, ov)
                M[tuple(slices)] *= blend.reshape(
                    [-1 if i == axis else 1 for i in range(len(shape))]
                )

                # Back
                slices[axis] = slice(size - ov, size)
                M[tuple(slices)] *= inv_blend.reshape(
                    [-1 if i == axis else 1 for i in range(len(shape))]
                )

            return M

    def infer(self, input):
        """Run inference on input volume with blocking and blending."""
        device = sp.get_device(input)
        xp = device.xp

        # Get number of coefficients
        ncoeff = input.shape[0]

        # Inspect input
        is_3d = self.ndim == 3
        block_shape = (nn_inf_block_size,) * (3 if is_3d else 2)
        overlap = tuple(int(s * overlap_fract) for s in block_shape)

        # Setup blocking
        B = sp.linop.ArrayToBlocks(input.shape, block_shape, overlap)
        n_blocks = int(np.prod(B.oshape[1:-1]))
        B = sp.linop.Reshape(
            (ncoeff, n_blocks) + B.oshape[-len(block_shape) :], B.oshape
        )
        B = sp.linop.Transpose(B.oshape, (1, 0) + tuple(range(2, len(B.oshape)))) * B

        if self._verbose:
            print(f">> Inference using {n_blocks} blocks.")

        blend = self.create_blend_mask(block_shape, overlap)
        blend = xp.reshape(
            blend[None, ...].repeat(n_blocks, axis=0), (n_blocks,) + block_shape
        )

        # Normalize input
        scale = xp.linalg.norm(input.ravel(), ord=xp.inf) + xp.finfo(xp.float32).eps
        input = input / scale
        blocks = B(input)

        output_shape = list(blocks.shape)
        output_shape[1] = 2 * ncoeff
        with device:
            output = xp.zeros(output_shape, dtype=xp.complex64)

        for n in range(0, blocks.shape[0], self._batch_size):
            _make_inference(
                output, self.model, is_3d, n, self._batch_size, blocks, blend
            )

        output = output[:, :ncoeff, ...] + 1j * output[:, ncoeff:, ...]
        return B.H(output).squeeze() * scale

    @property
    def batch_size(self):
        """Getter for batch_size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Setter for batch_size with validation."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Batch size must be a positive integer.")
        self._batch_size = value

    def freeze(self):
        """Freeze the model for inference and switch to the 'infer' method."""
        self.model.eval()
        self.is_infer = True
        for param in self.model.parameters():
            param.requires_grad = False
        print("Model frozen. Ready for inference.")

    def unfreeze(self):
        """Unfreeze the model for training."""
        self.model.train()
        self.is_infer = False
        for param in self.model.parameters():
            param.requires_grad = True
        print("Model unfrozen. Ready for training.")


class LitResUNet2D(LightningDenoiser):
    """
    ResUNet 3D implementation that inherits from LightningDenoiser.

    Implements the denoising logic using the ResUNet2D architecture.
    """

    def __init__(self, device="cpu"):
        model = _ResUNet2D()  # Load ResUNet2D model
        super().__init__(model=model, device=device)

    def forward(self, x, sigma=None):
        """Denoising logic for ResUNet2D."""
        return self.model(x)


class LitResUNet3D(LightningDenoiser):
    """
    ResUNet 3D implementation that inherits from LightningDenoiser.

    Implements the denoising logic using the ResUNet23D architecture.
    """

    def __init__(self, device="cpu"):
        model = _ResUNet3D()  # Load ResUNet2D model
        super().__init__(model=model, device=device)

    def forward(self, x, sigma=None):
        """Denoising logic for ResUNet3D."""
        return self.model(x)


# %% utils
class _ResUNet2D(nn.Module):
    """Residual U-Net for 2D data."""

    def __init__(self, ncoeffs=5):
        super().__init__()
        self.tk = ncoeffs

        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.tk * 2, 64, kernel_size=nn_kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=nn_kernel, padding=1),
        )
        self.shortcut_conv = nn.Conv2d(self.tk * 2, 64, kernel_size=1, padding=0)

        # Encoder path
        self.enc1 = _ResidualBlock2D(64, 128, stride=2)
        self.enc2 = _ResidualBlock2D(128, 256, stride=2)
        self.enc3 = _ResidualBlock2D(256, 512, stride=2)

        # Decoder path
        self.dec1 = _DecoderBlock2D(512, 256)
        self.dec2 = _DecoderBlock2D(256, 128)
        self.dec3 = _DecoderBlock2D(128, 64)

        # Output
        self.final_conv = nn.Conv2d(64, self.tk * 2, kernel_size=1, padding=0)

    def forward(self, x):
        # Initial conv + shortcut
        out = self.initial_conv(x)
        skip = self.shortcut_conv(x)
        enc1 = out + skip

        # Encoder
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)
        bridge = self.enc3(enc3)

        # Decoder
        dec1 = self.dec1(bridge, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)

        return self.final_conv(dec3)


class _ResidualBlock2D(nn.Module):
    """2D Residual block with two convolutions and a shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=nn_kernel, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=nn_kernel, padding=1, stride=1
        )

        # Shortcut path with 1x1 convolution
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, stride=stride
        )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class _DecoderBlock2D(nn.Module):
    """2D Decoder block: Upsample + concatenate + residual block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = _ResidualBlock2D(in_channels + out_channels, out_channels)

    def forward(self, decoder_input, encoder_output):
        upsampled = self.upsample(decoder_input)
        concatenated = torch.cat([upsampled, encoder_output], dim=1)
        return self.res_block(concatenated)


class _ResUNet3D(nn.Module):
    """Residual U-Net architecture for 3D inputs using custom residual and decoder blocks."""

    def __init__(self, ncoeffs=5):
        super().__init__()
        self.tk = ncoeffs

        self.initial_conv = nn.Sequential(
            nn.Conv3d(self.tk * 2, 64, kernel_size=nn_kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=nn_kernel, padding=1),
        )
        self.shortcut_conv = nn.Conv3d(ncoeffs * 2, 64, kernel_size=1, padding=0)

        # Encoder blocks
        self.enc1 = _ResidualBlock3D(64, 128, stride=2)
        self.enc2 = _ResidualBlock3D(128, 256, stride=2)
        self.enc3 = _ResidualBlock3D(256, 512, stride=2)

        # Decoder blocks
        self.dec1 = _DecoderBlock3D(512, 256)
        self.dec2 = _DecoderBlock3D(256, 128)
        self.dec3 = _DecoderBlock3D(128, 64)

        # Final 1x1 convolution to get desired output channels
        self.final_conv = nn.Conv3d(64, self.tk * 2, kernel_size=1, padding=0)

    def forward(self, x):
        # Initial convolution and skip connection
        out = self.initial_conv(x)
        skip = self.shortcut_conv(x)
        enc1 = out + skip

        # Encoder path
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)

        # Bridge
        bridge = self.enc3(enc3)

        # Decoder path
        dec1 = self.dec1(bridge, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)

        # Final output
        return self.final_conv(dec3)


class _ResidualBlock3D(nn.Module):
    """Residual block with two 3D convolutions and a shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=nn_kernel, padding=1, stride=stride
        )
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=nn_kernel, padding=1, stride=1
        )

        # Shortcut path with 1x1x1 convolution
        self.shortcut = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, padding=0, stride=stride
        )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class _DecoderBlock3D(nn.Module):
    """Decoder block that upsamples input and applies a residual block after concatenation with encoder features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )
        self.res_block = _ResidualBlock3D(in_channels + out_channels, out_channels)

    def forward(self, decoder_input, encoder_output):
        upsampled = self.upsample(decoder_input)
        concatenated = torch.cat([upsampled, encoder_output], dim=1)
        return self.res_block(concatenated)


@with_torch
def _make_inference(output, model, is_3d, n, batch_size, blocks, blend):
    blk = blocks[n : n + batch_size, ...]
    msk = blend[n : n + batch_size, ...]

    if not is_3d and blk.ndim == 4:
        blk = blk[:, :, None, :, :]  # add depth dim for 2D

    blk[torch.isnan(blk)] = 0
    blk[torch.isinf(blk)] = 0
    blk = torch.cat((blk.real, blk.imag), dim=1)

    with torch.no_grad():
        pred = model * msk[:, None, ...]
    output[n : n + batch_size, ...] = pred
