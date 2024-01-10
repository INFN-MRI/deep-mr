"""Wavelet denoising prior."""

__all__ = ["WaveletPrior"]

import torch
import torch.nn as nn

import ptwt
import pywt

class WaveletPrior(nn.Module):
    r"""
    Wavelet denoising with the :math:`\ell_1` norm.

    This denoiser is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|\Psi x\|_n

    where :math:`\Psi` is an orthonormal wavelet transform, :math:`\lambda>0` is a hyperparameter, and where
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``) or
    the :math:`\ell_0` norm (``non_linearity="hard"``). A variant of the :math:`\ell_0` norm is also available
    (``non_linearity="topk"``), where the thresholding is done by keeping the :math:`k` largest coefficients
    in each wavelet subband and setting the others to zero.

    The solution is available in closed-form, thus the denoiser is cheap to compute.

    Attributes
    ----------
    dim: int
        Number of spatial dimensions.
    level : int, optional
        Decomposition level of the wavelet transform. The default is 3.
    wv : str, optional wv: 
        Mother wavelet (follows the `PyWavelets convention <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_) 
        The default is "db8".
    device : str, optional
        ``"cpu"`` or ``"gpu"``. The default if ``"cpu"``.
    non_linearity : str, optional
        ``"soft"``, ``"hard"`` or ``"topk"`` thresholding.
        If ``"topk"``, only the top-k wavelet coefficients are kept.
        The default is ``"soft"``.
        
    """

    def __init__(self, dim, level=3, wv="db8", device="cpu", non_linearity="soft"):
        super().__init__()
        
        # select correct wavelet transform
        wavelet = pywt.Wavelet(wv)
        if dim == 1:
            dwt = ptwt.wavedec
            iwt = ptwt.waverec
        if dim == 2:
            dwt = ptwt.wavedec2
            iwt = ptwt.waverec2
        if dim == 3:
            dwt = ptwt.wavedec3
            iwt = ptwt.waverec3
            
        self.level = level
        self.device = device
        self.dwt = lambda x : dwt(x.to(self.device), wavelet, mode="zero", level=self.level)
        self.iwt = lambda x : iwt(x.to(self.device), wavelet, mode="zero", level=self.level)
        self.non_linearity = non_linearity

    def _get_ths_map(self, ths):
        if isinstance(ths, float) or isinstance(ths, int):
            ths_map = ths
        elif len(ths.shape) == 0 or ths.shape[0] == 1:
            ths_map = ths.to(self.device)
        else:
            ths_map = (
                ths.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(self.device)
            )
        return ths_map

    def prox_l1(self, x, ths=0.1):
        """
        Soft thresholding of the wavelet coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Wavelet coefficients.
        ths : float, torch.Tensor, optional
            Threshold. The default is 0.1.

        """
        ths_map = self._get_ths_map(ths)
        return torch.maximum(
            torch.tensor([0], device=x.device).type(x.dtype), x - ths_map
        ) + torch.minimum(torch.tensor([0], device=x.device).type(x.dtype), x + ths_map)

    def prox_l0(self, x, ths=0.1):
        """
        Hard thresholding of the wavelet coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Wavelet coefficients.
        ths : float, torch.Tensor, optional
            Threshold. The default is 0.1.

        """
        if isinstance(ths, float):
            ths_map = ths
        else:
            ths_map = self._get_ths_map(ths)
            ths_map = ths_map.repeat(
                1, 1, 1, x.shape[-2], x.shape[-1]
            )  # Reshaping to image wavelet shape
        out = x.clone()
        out[abs(out) < ths_map] = 0
        return out

    def hard_threshold_topk(self, x, ths=0.1):
        r"""
        Hard thresholding of the wavelet coefficients.
        
        Keeps only the top-k coefficients and setting the others to 0.

        Parameters
        ----------
        x : torch.Tensor
            wavelet coefficients.
        ths : float,  int, optional
            top k coefficients to keep. If ``float``, it is interpreted as a proportion of the total
            number of coefficients. If ``int``, it is interpreted as the number of coefficients to keep.
            The default is 0.1.

        """
        if isinstance(ths, float):
            k = int(ths * x.shape[-2] * x.shape[-1])
        else:
            k = int(ths)

        # Reshape arrays to 2D and initialize output to 0
        x_flat = x.view(x.shape[0], -1)
        out = torch.zeros_like(x_flat)

        topk_indices_flat = torch.topk(abs(x_flat), k, dim=-1)[1]

        # Convert the flattened indices to the original indices of x
        batch_indices = (
            torch.arange(x.shape[0], device=x.device).unsqueeze(1).repeat(1, k)
        )
        topk_indices = torch.stack([batch_indices, topk_indices_flat], dim=-1)

        # Set output's top-k elements to values from original x
        out[tuple(topk_indices.view(-1, 2).t())] = x_flat[
            tuple(topk_indices.view(-1, 2).t())
        ]
        return torch.reshape(out, x.shape)

    def forward(self, x, ths=0.1):
        """
        Run the model on a noisy image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy image.
        ths : int, float, torch.Tensor, optional
            Thresholding parameter. 
            If ``non_linearity`` equals ``"soft"`` or ``"hard"``, ``ths`` serves as a (soft or hard)
            thresholding parameter for the wavelet coefficients. If ``non_linearity`` equals ``"topk"``,
            ``ths`` can indicate the number of wavelet coefficients
            that are kept (if ``int``) or the proportion of coefficients that are kept (if ``float``)
            The default is 0.1.

        """
        # h, w = x.size()[-2:]
        # padding_bottom = h % 2
        # padding_right = w % 2
        # x = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(x)

        coeffs = self.dwt(x)
        for l in range(self.level):
            ths_cur = _get_threshold(ths, l)
            if self.non_linearity == "soft":
                coeffs[1][l] = self.prox_l1(coeffs[1][l], ths_cur)
            elif self.non_linearity == "hard":
                coeffs[1][l] = self.prox_l0(coeffs[1][l], ths_cur)
            elif self.non_linearity == "topk":
                coeffs[1][l] = self.hard_threshold_topk(coeffs[1][l], ths_cur)
        y = self.iwt(coeffs)

        # y = y[..., :h, :w]
        return y
    
def _get_threshold(ths, l):
    ths_cur = (
            ths
            if (
                isinstance(ths, int)
                or isinstance(ths, float)
                or len(ths.shape) == 0
                or ths.shape[0] == 1
            )
            else ths[l]
        )
    return ths_cur