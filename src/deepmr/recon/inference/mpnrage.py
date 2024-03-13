"""MPnRAGE T1 mapping fitting routines."""

__all__ = ["mpnrage_fit"]

import numpy as np
import torch

from ... import bloch

from . import solvers


def mpnrage_fit(input, t1grid, flip, TR, TI):
    """
    Fit T1 from input MPnRAGE data.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input image series of shape (ncontrasts, nz, ny, nx).
    t1grid : Iterable[float]
        T1 grid (start, stop, nsteps) in [ms].
    flip : np.ndarray | torch.Tensor
        Excitation flip angles in [deg].
    TR : float
        Repetition Time in [ms].
    TI : float
        Inversion Time in [ms].

    Returns
    -------
    m0 : np.ndarray | torch.Tensor
        Proton Density map of shape (nz, ny, nx).
    t1map : np.ndarray | torch.Tensor
        T1 map of shape (nz, ny, nx) in [ms].

    """

    if isinstance(input, torch.Tensor):
        istorch = True
        device = input.device
        input = input.numpy(force=True)
    else:
        istorch = False

    # first build grid
    t1lut = np.linspace(t1grid[0], t1grid[1], t1grid[2])
    t2 = 10.0

    # build dictionary
    nshots = input.shape[0]
    flip = flip * np.ones(nshots)
    atoms = bloch.mprage(nshots, flip, TR, t1lut, t2, TI=TI)
    blochdict = solvers.BlochDictionary(atoms, t1lut[:, None], ["T1"])

    # perform matching
    m0, maps = solvers.tsmi2map(blochdict, input)

    # here, we only have T1
    t1map = maps["T1"]

    # cast back
    if istorch:
        m0 = torch.as_tensor(m0, device=device)
        t1map = torch.as_tensor(t1map, device=device)

    return m0, t1map
