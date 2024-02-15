"""Phase Cycling schemes."""

__all__ = ["rf_phase_cycle"]

import numpy as np

def rf_phase_cycle(n, arg2):
    """
    Calculate phase cycling pattern (for acquisition and simulation).

    Args:
        n (int): number of RF pulses.
        Phi0 (float, str): (quadratic) spoiling phase increment in [deg].

    Returns:
        (array): phase list.
    
    Ported from:
     Shaihan Malik July 2017

    """

    if isinstance(arg2, str) is False:
        spgr = True
        Phi0 = np.deg2rad(arg2)
    else:
        spgr = False

    if spgr:
        # quadratic 
        p = np.arange(n)
        phi = p * (p+1) / 2 * Phi0
    else:
        # balanced case
        if n % 2 == 0:
            phi = np.tile(np.asarray([0, np.pi]), int(n / 2))
        else:
            phi = np.tile(np.asarray([0, np.pi]), int(np.floor(n / 2)))
            phi = np.concatenate((phi, [0]))
    
    return phi