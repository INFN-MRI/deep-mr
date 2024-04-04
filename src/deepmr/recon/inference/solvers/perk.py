"""PERK regression routines."""

__all__ = ["perk_train", "perk_eval"]

from dataclasses import dataclass
import math

import numpy as np
import torch


def perk_eval(input, train, prior=None, reg=2**-41, chunk_size=1e5, device=None):
    """
    Perform PERK evaluation.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input image series to be fitted of shape ``(ncontrasts, ny, nx)``
        (single-slice 2D) or ``(ncontrasts, nz, ny, nx)`` (multi-slice 2D or 3D).
    train : RFFKernel
        Trained kernel object with the following fields:

        * .mean.z - sample mean of signals of length ``(npriors+ncontrasts,)``.
        * .mean.x - sample mean of latent parameters of length ``(nparams,)``.
        * .cov.zz - sample auto-cov of signals of shape ``(H, H)``.
        * .cov.xz - sample cross-cov b/w latent parameters and signals of shape ``(nparams, H)``.
        * .freq - random "frequency" vector of shape ``(H, npriors+ncontrasts)``.
        * .ph - random phase vector  of length ``(H,)``.

        with ``H`` being the embedding dimension (see :func:`perk_train`).

    prior : torch.Tensor, optional
        Known parameters (e.g., ``B0``, ``B1+``) of shape ``(natoms, npriors)``.
        The default is ``None``.
    reg : float, optional
        Fitting regularization. The default is ``2**-41``.
    chunk_size : int, optional
        Number of voxels to be processed in parallel.
        Increase this number to speed-up computation at cost
        of increased memory footprint.
        The default is ``1e5``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Output parametric maps of shape ``(nparams, ny, nx)``
        (single-slice 2D) or ``(nparams, nz, ny, nx)`` (multi-slice 2D or 3D).

    """
    # check if numpy
    if isinstance(input, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    # cast to tensor
    input = torch.as_tensor(input, dtype=torch.complex64)

    if prior is not None:
        nu = torch.as_tensor(prior, dtype=torch.float32)
    else:
        nu = None

    if nu is not None and len(nu.shape) == 1:
        nu = nu.unsqueeze(1)

    # parse device
    idevice = input.device
    if device is None:
        device = train.device

    # offload
    input = input.to(device)
    train = train.to(device)
    if nu is not None:
        nu = nu.to(device)

    # get shape
    ishape = input.shape[1:]  # (nz, ny, nx)

    # reshape
    input = input.reshape(input.shape[0], -1)  # (nechoes, nvoxels)

    # normalize
    norm = (input * input.conj()).sum(axis=0) ** 0.5
    input = input / (norm + 0.000000000001)

    # unwind phase
    ph0 = input[0]
    ph0 = ph0 / (abs(ph0) + 0.000000000001)  # keep only phase
    input = input * ph0.conj()

    # get real part only
    input = input.real

    # append known signals
    if nu is not None:
        nu = nu.reshape(nu.shape[0], -1)
        input = torch.cat((input, nu), axis=0)

    # prepare batches
    chunk_input = torch.split(input, int(chunk_size), dim=-1)

    # feature maps
    output = []
    for chunk in chunk_input:
        z = rff_map(input, train.H, train.freq, train.ph)

        # kernel ridge regression
        den = z - train.mean.z
        num = train.cov.zz + reg * _eye(train.H, train.cov.zz)
        tmp = torch.linalg.solve(num, den.t())  # (H, V) (V=voxels)
        tmp = tmp.t()  # (V, H)

        # estimate values
        covxz = train.cov.xz
        if nu is not None:
            nknown = nu.shape[0]
            covxz = covxz[nknown:, :]

        # prepare
        chunk_output = []
        for n in range(covxz.shape[0]):  # iterate over parameters
            chunk_output.append(tmp @ covxz[n] + train.mean.x[n])

        # stack
        output.append(torch.stack(chunk_output, axis=0))

    # concatenate chunks
    output = torch.cat(output, axis=-1)

    # cast back to original device
    output = output.to(idevice)

    # cast back to numpy
    if isnumpy:
        output = output.numpy(force=True)

    return output.reshape(-1, *ishape).squeeze()


def perk_train(
    train_signals,
    train_labels,
    train_priors=None,
    H=10**3,
    lamda=2**-1.5,
    c=2**0.6,
    noise=0.0,
    device=None,
    seed=42,
):
    """
    Perform PERK training.

    Parameters
    ----------
    train_signals : np.ndarray | torch.Tensor
        Input training of signal ``(natoms, ncontrasts)``.
    train_labels : np.ndarray | torch.Tensor
        Input labels of training signals (e.g., each atom T1/T2/...) of shape
        ``(natoms, nparams)``.
    train_priors : np.ndarray ! torch.Tensor, optional
        Known parameters (e.g., ``B0``, ``B1+``) of shape ``(natoms, npriors)``.
        The default is ``None``.
    H : int, optional
        Embedding dimension. The default is ``10**3``.
    lamda : float, optional
        Regularization parameter for latent parameters. The default is ``2**-1.5``.
    c : float, optional
        Global kernel length scale parameter. The default is ``2**0.6``.
    noise : float, optional
        Noise variance - should be estimated from data. The default is ``0.0``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    seed : int, optional
        Random number generator seed. The default is ``42``.

    Returns
    -------
    train : RFFKernel
        Kernel object with the following fields:

        * .mean.z - sample mean of signals of length ``(npriors+ncontrasts,)``.
        * .mean.x - sample mean of latent parameters of length ``(nparams,)``.
        * .cov.zz - sample auto-cov of signals of shape ``(H, H)``.
        * .cov.xz - sample cross-cov b/w latent parameters and signals of shape ``(nparams, H)``.
        * .freq - random "frequency" vector of shape ``(H, npriors+ncontrasts)``.
        * .ph - random phase vector  of length ``(H,)``.

    """
    # set seed
    torch.manual_seed(seed)

    # send everything to pytorch
    train_y = torch.as_tensor(train_signals, dtype=torch.complex64)
    train_x = torch.as_tensor(train_labels, dtype=torch.float32)
    if train_priors is not None:
        train_nu = torch.as_tensor(train_priors, dtype=torch.float32)
    else:
        train_nu = None

    # expand dims if required
    if len(train_x.shape) == 1:
        train_x = train_x.unsqueeze(1)
    if train_nu is not None and len(train_nu.shape) == 1:
        train_nu = train_nu.unsqueeze(1)

    # get device
    if device is None:
        device = train_y.device

    # offload
    train_y = train_y.to(device)
    train_x = train_x.to(device)
    if train_nu is not None:
        train_nu = train_nu.to(device)

    # prepare output
    train = RFFKernel(device)

    # transpose
    train_y = train_y.t()

    # normalize
    norm = (train_y * train_y.conj()).sum(axis=0) ** 0.5
    train_y = train_y / (norm + 0.000000000001)

    # add noise
    train_y = train_y + noise * torch.randn(
        train_y.shape, dtype=torch.complex64, device=train_y.device
    )

    # unwind phase
    ph0 = train_y[0]
    ph0 = ph0 / (abs(ph0) + 0.000000000001)  # keep phase only
    train_y = train_y * ph0.conj()

    # get real part only
    train_y = train_y.real

    # transpose back
    train_y = train_y.t()

    # training inputs
    if train_nu is not None:
        train_y = torch.cat(
            (train_y, train_nu), axis=-1
        )  # (ncontrasts+npriors, natoms)

    # get lengthscales
    lengthscale = lamda * train_y.mean(axis=0)  # (nechoes+nknown,)
    Q = lengthscale.shape[0]
    K = train_y.shape[0]

    # Random Fourier Features
    # To approximate Gaussian kernel N(0, Sigma):
    # 1. Construct rff.cov = inv((2*pi)^2*Sigma)
    # 1. Draw rff.freq from N(0, rff.cov)
    # 2. Draw rff.ph from unif(0, 1)

    tmp = lengthscale * (2 * math.pi * c)
    tmp = 1.0 / (tmp**2 + 0.000000000000001)
    cov = torch.diag(tmp)  # (ncontrasts+npriors, ncontrasts+npriors)
    freq = torch.randn(
        H, Q, dtype=cov.dtype, device=cov.device
    ) @ torch.linalg.cholesky(cov)
    ph = torch.rand(H, dtype=cov.dtype, device=cov.device)

    # Feature maps
    z = rff_map(train_y.t(), H, freq, ph)  # (natoms, H)

    # Sample means
    train.mean.z = z.mean(axis=0)  # (H,)
    train.mean.x = train_x.mean(axis=0)  # (nparams,)

    # Sample covariances
    tmp = train_x - train.mean.x  # (natoms, nparams) - (nparams,)
    z = z - train.mean.z

    train.cov.zz = (z.t() @ z) / K  # (H, H)
    train.cov.xz = (tmp.t() @ z) / K  # (nparams, H)

    train.freq = freq
    train.ph = ph
    train.H = H

    # Make sure train is on the desired device
    train = train.to(device)

    return train


# %% local utils
@dataclass
class SampleMean:
    """
    Sample mean.

    Attributes
    ----------
    z : torch.Tensor
        Sample mean of signals of length ``(npriors+ncontrasts,)``.
    x : torch.Tensor
        Sample mean of latent parameters of length ``(nparams,)``.

    """

    z: torch.Tensor = None
    x: torch.Tensor = None

    def to(self, device):
        self.z = self.z.to(device)
        self.x = self.x.to(device)
        return self


@dataclass
class SampleCov:
    """
    Sample covariance.

    Attributes
    ----------
    zz : torch.Tensor
        Sample auto-cov of signals of shape ``(H, H)``
    xz : torch.Tensor
        Sample cross-cov b/w latent parameters and signals of shape ``(nparams, H)``.

    """

    zz: torch.Tensor = None
    xz: torch.Tensor = None

    def to(self, device):
        self.zz = self.zz.to(device)
        self.xz = self.xz.to(device)
        return self


@dataclass
class RFFKernel:
    """
    Random Fourier Features kernel.

    Attributes
    ----------
    mean : SampleMean
        Sample mean.
    cov : SampleCov
        Sample covariance.
    freq : torch.Tensor
        Random "frequency" vector of shape ``(H, npriors+ncontrasts)``.
    ph : torch.Tensor
        Random phase vector  of length ``(H,)``.

    """

    device: str
    mean: SampleMean = None
    cov: SampleCov = None
    freq: torch.Tensor = None
    ph: torch.Tensor = None
    H: int = None

    def __post_init__(self):
        self.mean = SampleMean()
        self.cov = SampleCov()

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.freq = self.freq.to(device)
        self.ph = self.ph.to(device)

        return self


def rff_map(train_y, H, freq, ph):
    """
    Feature mapping via random Fourier features.
    """
    # Check freq and ph dimensions
    if freq.shape[0] != H or ph.shape[0] != H:
        raise ValueError("Length mismatch: freq and/or ph not of length H!?")

    # Random Fourier Features
    tmp = freq @ train_y
    tmp = tmp + ph[:, None]
    tmp = torch.cos(2 * math.pi * tmp)
    z = (2 / H) ** 0.5 * tmp

    return z.t()


def _eye(sz, tensor):
    return torch.eye(sz, dtype=tensor.dtype, device=tensor.device)
