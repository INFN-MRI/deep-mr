"""Sub-package containing MR Encoding Operator builder."""

__all__ = ["EncodingOp"]

from ... import linops as _linops
from .. import calib as _calib


def EncodingOp(
    data,
    mask=None,
    traj=None,
    dcf=None,
    shape=None,
    nsets=1,
    basis=None,
    device=None,
    cal_data=None,
    sensmap=None,
    toeplitz=False,
):
    """
    Prepare MR encoding operator for Cartesian / Non-Cartesian imaging.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input k-space data of shape ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    mask : np.ndarray | torch.Tensor, optional
        Sampling mask for Cartesian imaging.
        Expected shape is ``(ncontrasts, nviews, nsamples)``.
        The default is ``None``.
    traj : np.ndarray | torch.Tensor, optional
        K-space trajectory for Non Cartesian imaging.
        Expected shape is ``(ncontrasts, nviews, nsamples, ndims)``.
        The default is ``None``.
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation.
        Expected shape is ``(ncontrasts, nviews, nsamples)``.
        The default is ``None``.
    shape : Iterable[int], optional
        Cartesian grid size of shape ``(nz, ny, nx)``.
        The default is ``None``.
    nsets : int, optional
        Number of coil sensitivity sets of maps. The default is ``1.
    basis : np.ndarray | torch.Tensor, optional
        Low rank subspace basis of shape ``(ncontrasts, ncoeffs)``. The default is ``None``.
    device : str, optional
        Computational device. The default is ``None`` (same as ``data``).
    cal_data : np.ndarray | torch.Tensor, optional
        Calibration dataset for coil sensitivity estimation.
        The default is ``None`` (use center region of ``data``).
    toeplitz : bool, optional
        Use Toeplitz approach for normal equation. The default is ``False``.

    Returns
    -------
    E : deepmr.linops.Linop
        MR encoding operator (i.e., ksp = E(img)).
    EHE : deepmr.linops.NormalLinop
        MR normal operator (i.e., img_n = EHE(img_n-1)).

    """
    # parse number of coils
    ncoils = data.shape[-4]

    # get device
    if device is None:
        device = data.device

    if mask is not None and traj is not None:
        raise ValueError("Please provide either mask or traj, not both.")
    if mask is not None:  # Cartesian
        # Fourier
        F = _linops.FFTOp(mask, basis, device)

        # Normal operator
        if toeplitz:
            FHF = _linops.FFTGramOp(mask, basis, device)
        else:
            FHF = F.H * F

        # Sensititivy
        if ncoils == 1:
            return F, FHF
        else:
            if sensmap is None:
                if cal_data is not None:
                    sensmap, _ = _calib.espirit_cal(cal_data.to(device), nsets=nsets)
                else:
                    sensmap, _ = _calib.espirit_cal(data.to(device), nsets=nsets)

            # infer from mask shape whether we are using multicontrast or not
            if len(mask.shape) == 2:
                multicontrast = False  # (ny, nx) / (nz, ny)
            else:
                multicontrast = True  # (ncontrast, ny, nx) / (ncontrast, nz, ny)

            # Coil operator
            C = _linops.SenseOp(2, sensmap, multicontrast=multicontrast)

            # Full encoding operator
            E = F * C
            EHE = C.H * FHF * C

            return E, EHE

    if traj is not None:
        assert shape is not None, "Please provide shape for Non-Cartesian imaging."
        ndim = traj.shape[-1]

        # Fourier
        F = _linops.NUFFTOp(traj, shape[-ndim:], basis, dcf, device=device)

        # Normal operator
        if toeplitz:
            FHF = _linops.NUFFTGramOp(traj, shape[-ndim:], basis, dcf, device=device)
        else:
            FHF = F.H * F

        # Sensititivy
        if ncoils == 1:
            return F, FHF
        else:
            if sensmap is None:
                if cal_data is not None:
                    sensmap, _ = _calib.espirit_cal(
                        cal_data.to(device),
                        nsets=nsets,
                        coord=traj,
                        shape=shape,
                        dcf=dcf,
                    )
                else:
                    sensmap, _ = _calib.espirit_cal(
                        data.to(device), nsets=nsets, coord=traj, shape=shape, dcf=dcf
                    )

            # infer from mask shape whether we are using multicontrast or not
            if len(traj.shape) < 4:
                multicontrast = False  # (nviews, nsamples, naxes) / (nsamples, naxes)
            else:
                multicontrast = True  # (ncontrast, nviews, nsamples, naxes

            # Coil operator
            C = _linops.SenseOp(ndim, sensmap, multicontrast=multicontrast)

            # Full encoding operator
            E = F * C
            EHE = C.H * FHF * C

            return E, EHE
