"""Local Low Rank denoising."""

__all__ = ["ProxLLR"]

import torch

from mrops import _sigpy as sp

from mrinufft._array_compat import with_torch


class ProxLLR(sp.prox.Prox):
    """
    Local Low Rank denoiser.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input shape ``(B, ...)``.
    lamda : float
        Threshold scaling.
    block : int
        Block size - assume isotropic.
    device : int | Device
        Computational device.
    randshift : bool, optional
        Toggle random shifting of blocks. The default is ``True``.
    stride : int | None, optional
        Stride of blocks. Default to ``block`` (non overlapping blocks).

    """

    def __init__(
        self,
        shape: list[int] | tuple[int],
        lamda: float,
        block: int,
        device: int | sp.Device,
        randshift: bool = True,
        stride: int | None = None,
    ):
        self.N = len(shape[1:])
        assert self.N == 2 or self.N == 3

        self.lamda = lamda
        self.block = block
        self.randshift = randshift

        if stride is None:
            stride = block

        # Block
        B = sp.linop.ArrayToBlocks(shape[1:], (block,) * self.N, (stride,) * self.N)
        if stride != block:
            xp = device.xp
            self.w = (B.H * B)(xp.ones(B.ishape, dtype=xp.complex64))
        else:
            self.w = None
        B = sp.linop.ArrayToBlocks(shape, (block,) * self.N, (stride,) * self.N)

        # Tensor reshape
        if self.N == 3:
            T = sp.linop.Transpose(B.oshape, (1, 2, 3, 0, 4, 5, 6))
            n = T.oshape[0] * T.oshape[1] * T.oshape[2]
        else:
            T = sp.linop.Transpose(B.oshape, (1, 2, 0, 3, 4))
            n = T.oshape[0] * T.oshape[1]
        R = sp.linop.Reshape((n, shape[0], block**self.N), T.oshape)
        self.L = R * T * B

        super().__init__(shape)

    def _prox(self, alpha, input):
        return _llr(
            input,
            self.lamda * alpha,
            self.N,
            self.L,
            self.w,
            self.block,
            self.randshift,
        )


# %% utils
def _llr(x, lamda, N, L, w, block, randshift):
    device = sp.get_device(x)
    xp = device.xp

    # perform random shifting
    if randshift:
        shift = [xp.random.randint(block) - int(block / 2) for n in range(N)]
        for k in range(N):
            x = xp.roll(x, shift[k], axis=-(k + 1))

    with device:
        # LLR denoising
        mats = L(x)
        u, s, vt = _svd(mats)
        thresh_s = s - lamda
        thresh_s[thresh_s < 0] = 0
        mats[...] = xp.matmul(u * thresh_s[..., None, :], vt.conj())
        x = L.H(mats)
        if w is not None:
            x = x / w[None, ...]

        # invert shifting
        if randshift:
            for k in range(N):
                x = xp.roll(x, -shift[k], axis=-(k + 1))

        return xp.nan_to_num(x, posinf=0.0, neginf=0.0)


@with_torch
def _svd(input):
    u, s, vh = torch.linalg.svd(input, full_matrices=False)
    return u, s, vh.conj()
