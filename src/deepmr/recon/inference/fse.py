"""FSE T2 mapping fitting routines."""

__all__ = ["fse_fit"]

import numpy as np
import torch

from ... import bloch

from . import solvers


def fse_fit(input, t2grid, flip, ESP, phases=None):
    """
    Fit T2 from input Fast Spin Echo data.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input image series of shape (ncontrasts, nz, ny, nx).
    t2grid : Iterable[float]
        T2 grid (start, stop, nsteps) in [ms].
    flip : np.ndarray | torch.Tensor
        Refocusing flip angles in [deg].
    ESP : float
        Echo spacing in [ms].
    phases : np.ndarray | torch.Tensor, optional
        Refocusing pulses phases. The default is 0 * flip.

    Returns
    -------
    m0 : np.ndarray | torch.Tensor
        Proton Density map of shape (nz, ny, nx).
    t2map : np.ndarray | torch.Tensor
        T2 map of shape (nz, ny, nx) in [ms].

    """

    if isinstance(input, torch.Tensor):
        istorch = True
        device = input.device
        input = input.numpy(force=True)
    else:
        istorch = False

    # default
    if phases is None:
        phases = 0.0 * flip

    # first build grid
    t2lut = np.linspace(t2grid[0], t2grid[1], t2grid[2])
    t1 = 1100.0

    # build dictionary
    atoms = bloch.fse(flip, phases, ESP, t1, t2lut)
    blochdict = solvers.BlochDictionary(atoms, t2lut[:, None], ["T2"])

    # perform matching
    m0, maps = solvers.tsmi2map(blochdict, input)

    # here, we only have T2
    t2map = maps["T2"]

    # cast back
    if istorch:
        m0 = torch.as_tensor(m0, device=device)
        t2map = torch.as_tensor(t2map, device=device)

    return m0, t2map
