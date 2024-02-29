"""RF phase generation routines."""

__all__ = ["phase_cycling", "rf_spoiling"]

import numpy as np
import torch


def phase_cycling(length, dphi=180.0):
    """
    Generate a linear phase cycling scheme.

    Parameters
    ----------
    length : int
        Flip angle train length.
    dphi : float, optional
        Linear phase increment in ``[deg]``.
        The default is ``180.0 [deg]``.

    Returns
    -------
    phase : torch.Tensor
        RF pulse phase for each pulse in a train.

    """
    # generate phase
    phase = np.arange(length, dtype=np.float32) * dphi
    phase = phase % 360.0

    return torch.as_tensor(phase)


def rf_spoiling(length, dphi=117.0):
    """
    Generate a quadratic phase cycling scheme
    for rf spoiling or partial spoiling.

    Parameters
    ----------
    length : int
        Flip angle train length.
    dphi : float, optional
        Quadratic phase increment in ``[deg]``.
        The default is ``117.0 [deg]``.

    Returns
    -------
    phase : torch.Tensor
        RF pulse phase for each pulse in a train.

    """
    # generate phase
    phase = np.zeros(length, dtype=np.float32)
    for n in range(length):
        phase[n] = (n * (n + 1) / 2.0 * dphi) % 360

    return torch.as_tensor(phase)
