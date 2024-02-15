"""Utils for timing calculation."""

__all__ = ["calculate_timing"]

import numpy as np


def calculate_timing(nechoes, echo_idx, gdt, k, grad):
    """
    Calculate Echo Times and readout time map for a given interleave.

    Args:
        nechoes (int): number of echoes.
        echo_idx (int): index along the gradient waveform corresponding to echo sampling.
        gdt (float) gradient sampling time in [us].
        k (array): k-space trajectory array of shape (nkpts, ndim).
        grad (array): gradient waveform array of shape (ngpts, ndim).

    Returns:
        (array): echo time(s) in [ms].
        (array): relative time coordinates (starting from 0.0) along the k-space interleave
            of shape (nkpts,) in [ms].
    """
    # cast to float
    gdt = float(gdt)

    # get min_te
    min_te = echo_idx * gdt

    # get echo spacing
    if grad is not None:
        esp = grad.shape[-1] * gdt
    else:
        esp = k.shape[-1] * gdt

    # calculate te
    te = np.arange(nechoes, dtype=np.float32) * esp + min_te

    # calculate timing along readout
    t = np.arange(k.shape[-1], dtype=np.float32)

    # cast to ms
    te *= 1e-3  # us -> ms
    t *= 1e-3  # us -> ms

    return te, t
