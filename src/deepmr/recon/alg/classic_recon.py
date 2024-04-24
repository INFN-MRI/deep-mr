"""Classical iterative reconstruction wrapper."""

__all__ = ["recon_lstsq"]

import numpy as np
import torch


from ... import linops as _linops
from ... import optim as _optim
from ... import prox as _prox


from .. import calib as _calib


from . import linop as _linop

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def recon_lstsq(
    data,
    head,
    mask=None,
    niter=1,
    prior=None,
    prior_ths=0.01,
    prior_params=None,
    solver_params=None,
    lamda=0.0,
    stepsize=1.0,
    basis=None,
    nsets=1,
    device=None,
    cal_data=None,
    sensmap=None,
    toeplitz=True,
    use_dcf=True,
):
    """
    Classical MR reconstruction.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input k-space data of shape ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    head : deepmr.Header
        DeepMR acquisition header, containing ``traj``, ``shape`` and ``dcf``.
    mask : np.ndarray | torch.Tensor, optional
        Sampling mask for Cartesian imaging.
        Expected shape is ``(ncontrasts, nviews, nsamples)``.
        The default is ``None``.
    niter : int, optional
        Number of recon iterations. If single iteration,
        perform simple zero-filled recon. The default is ``1``.
    prior : str | deepinv.optim.Prior, optional
        Prior for image regularization. If string, it must be one of the following:

        * ``"L1Wav"``: L1 Wavelet regularization.
        * ``"TV"``: Total Variation regularization.

        The default is ``None`` (no regularizer).
    prior_ths : float, optional
        Threshold for denoising in regularizer. The default is ``0.01``.
    prior_params : dict, optional
        Parameters for Prior initializations.
        See :func:`deepmr.prox`.
        The defaul it ``None`` (use each regularizer default parameters).
    solver_params : dict, optional
        Parameters for Solver initializations.
        See :func:`deepmr.optim`.
        The defaul it ``None`` (use each solver default parameters).
    lamda : float, optional
        Regularization strength. If 0.0, do not apply regularization.
        The default is ``0.0``.
    stepsize : float, optional
        Iterations step size. If not provided, estimate from Encoding
        operator maximum eigenvalue. The default is ``None``.
    basis : np.ndarray | torch.Tensor, optional
        Low rank subspace basis of shape ``(ncontrasts, ncoeffs)``. The default is ``None``.
    nsets : int, optional
        Number of coil sensitivity sets of maps. The default is ``1.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    cal_data : np.ndarray | torch.Tensor, optional
        Calibration dataset for coil sensitivity estimation.
        The default is ``None`` (use center region of ``data``).
    toeplitz : bool, optional
        Use Toeplitz approach for normal equation. The default is ``True``.
    use_dcf : bool, optional
        Use dcf to accelerate convergence. The default is ``True``.

    Returns
    -------
    img np.ndarray | torch.Tensor
        Reconstructed image of shape:

        * 2D Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 2D Non Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 2D Non Cartesian: ``(nslices, ncontrasts, ny, nx).
        * 3D Non Cartesian: ``(ncontrasts, nz, ny, nx).

    """
    if isinstance(data, np.ndarray):
        data = torch.as_tensor(data)
        isnumpy = True
    else:
        isnumpy = False

    if device is None:
        device = data.device
    data = data.to(device)

    if use_dcf and head.dcf is not None:
        dcf = head.dcf.to(device)
    else:
        dcf = None

    # toggle off Topelitz for non-iterative
    if niter == 1:
        toeplitz = False

    # get ndim
    if head.traj is not None:
        ndim = head.traj.shape[-1]
    else:
        ndim = 2  # assume 3D data already decoupled along readout

    # build encoding operator
    E, EHE = _linop.EncodingOp(
        data,
        mask,
        head.traj,
        dcf,
        head.shape,
        nsets,
        basis,
        device,
        cal_data,
        sensmap,
        toeplitz,
    )

    # transfer
    E = E.to(device)
    EHE = EHE.to(device)

    # perform zero-filled reconstruction
    if dcf is not None:
        img = E.H(dcf**0.5 * data[:, None, ...])
    else:
        img = E.H(data[:, None, ...])

    # if non-iterative, just perform linear recon
    if niter == 1:
        output = img
        if isnumpy:
            output = output.numpy(force=True)
        return output

    # default solver params
    if solver_params is None:
        solver_params = {}

    # rescale
    img = _calib.intensity_scaling(img, ndim=ndim)

    # if no prior is specified, use CG recon
    if prior is None:
        output, _ = _optim.cg_solve(
            img, EHE, niter=niter, lamda=lamda, ndim=ndim, **solver_params
        )
        if isnumpy:
            output = output.numpy(force=True)
        return output

    # modify EHE
    if lamda != 0.0:
        _EHE = EHE + lamda * _linops.Identity(ndim)
    else:
        _EHE = EHE

    # compute spectral norm
    xhat = torch.rand(img.shape, dtype=img.dtype, device=img.device)
    max_eig = _optim.power_method(None, xhat, AHA=_EHE, device=device, niter=30)
    if max_eig != 0.0:
        stepsize = stepsize / max_eig

    # if a single prior is specified, use PDG
    if isinstance(prior, (list, tuple)) is False:
        # default prior params
        if prior_params is None:
            prior_params = {}

        # get prior
        D = _get_prior(prior, ndim, lamda, device, **prior_params)

        # solve
        output, _ = _optim.pgd_solve(
            img, stepsize, _EHE, D, niter=niter, accelerate=True, **solver_params
        )
    else:
        npriors = len(prior)
        if prior_params is None:
            prior_params = [{} for n in range(npriors)]
        else:
            assert (
                isinstance(prior_params, (list, tuple)) and len(prior_params) == npriors
            ), "Please provide parameters for each regularizer (or leave completely empty to use default)"

        # get priors
        D = []
        for n in range(npriors):
            d = _get_prior(prior[n], ndim, lamda, device, **prior_params[n])
            D.append(d)

        # solve
        output, _ = _optim.admm_solve(
            img, stepsize, _EHE, D, niter=niter, **solver_params
        )
    if isnumpy:
        output = output.numpy(force=True)

    return output


# %% local utils
def _get_prior(ptype, ndim, lamda, device, **params):
    if isinstance(ptype, str):
        if ptype == "L1Wave":
            return _prox.WaveletDenoiser(ndim, ths=lamda, device=device, **params)
        elif ptype == "TV":
            return _prox.TVDenoiser(ndim, ths=lamda, device=device, **params)
        elif ptype == "LLR":
            return _prox.LLRDenoiser(ndim, ths=lamda, device=device, **params)
        else:
            raise ValueError(
                f"Prior type = {ptype} not recognized; either specify 'L1Wave', 'TV' or 'LLR', or 'nn.Module' object."
            )
    else:
        raise NotImplementedError("Direct prior object not implemented.")
