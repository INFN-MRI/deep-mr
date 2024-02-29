"""Variable flip angle train generation routines."""

__all__ = ["piecewise_fa", "sinusoidal_fa"]

import math
import numpy as np
import torch


def piecewise_fa(fa_ref=[1.5556, 70.0, 0.7778], length=[350, 230], recovery=300):
    """
    Design a multi-segment linear flip angle train.

    The resulting train, similar to the one introduced by Gomez et al. [1],
    consists of ``len(fa_max)`` linear sections. The ``n``-th sections starts at
    ``fa_ref[n] [deg]``, ends at ``fa_ref[n+1] [deg]`` and consists of ``length[n]`` pulses.
    The train is followed by a constant flip angle section of length ``recovery``.

    Parameters
    ----------
    fa_ref : Iterable[float], optional
        Starting flip angle in ``[deg]`` per linear segment.
        The default is ``(1.5556, 70., 0.7778) [deg]``.
    length : Iterable[int], optional
        Linear segment length.
        The default is ``(350, 200)`` pulses.
    recovery : int, optional
        Constant flip angle recovery train length.
        The default is ``300``.

    Returns
    -------
    rf_schedule : torch.Tensor
        Piecewise linear flip angle train in ``[deg]``.

    References
    ----------
    [1] Gómez, P.A., Cencini, M., Golbabaee, M. et al.
    Rapid three-dimensional multiparametric MRI with quantitative transient-state imaging.
    Sci Rep 10, 13769 (2020). https://doi.org/10.1038/s41598-020-70789-2

    """
    # generate segments
    rf_schedule = [
        np.linspace(fa_ref[n], fa_ref[n + 1], length[n], dtype=np.float32)
        for n in range(len(fa_ref - 1))
    ]
    rf_schedule = np.concatenate(rf_schedule)

    # add recovery
    if recovery > 0:
        rf_schedule = np.concatenate((rf_schedule, np.ones(recovery, dtype=np.float32)))

    return torch.as_tensor(rf_schedule)


def sinusoidal_fa(
    fa_max=(35.0, 43.0, 70.0, 45.0, 27.0),
    length=200,
    spacing=10,
    recovery=0,
    offset=5.0,
):
    """
    Design a multi-segment sinusoidal flip angle train.

    The resulting train, similar to the one introduced by Jiang et al. [1],
    consists of ``len(fa_max)`` sinusoidal section (positive wave),
    each of length ``length` separed by constant flip angle sections
    of length ``spacing``. The ``n``-th sinusoidal section peaks at
    ``fa_max[n] [deg]``. The train is followed by a constant flip angle
    section of length ``recovery``. The overall schedule minimum flip angle
    is determined by the ``offset`` parameter (in ``[deg]``).

    Parameters
    ----------
    fa_max : Iterable[float], optional
        Maximum flip angle in ``[deg]`` per sinusoidal segment.
        The default is ``(35., 43., 70., 45., 27.) [deg]``.
    length : int, optional
        Sinusoidal segment length.
        The default is ``200`` pulses.
    spacing : int, optional
        Zero degrees flip angle pulses in between segments.
        The default is ``10``.
    recovery : int, optional
        Constant flip angle recovery train length.
        The default is ``0``.
    offset : float, optional
        Minimum flip angle in ``[deg]``. The default is ``5. [deg]``.

    Returns
    -------
    rf_schedule : torch.Tensor
        Sinusoidal flip angle train in ``[deg]``.

    Examples
    --------
    >>> import deepmr

    A ``5`` sections flip angle train with ``10``-pulses long separation
    and no recovery (e.g., for 2D MR Fingerprinting) can be generated as:

    >>> fa_train = deepmr.sinusoidal_fa((35., 43., 70., 45., 27.), 200, 10)

    A final ``100`` constant flip angle segment (e.g., for 3D MR Fingerprinting)
    can be added via the ``recovery`` argument:

    >>> fa_train = deepmr.sinusoidal_fa((35., 43., 70., 45., 27.), 200, 10, recovery=100)

    References
    ----------
    [1] Jiang, Y., Ma, D., Seiberlich, N., Gulani, V. and Griswold, M.A. (2015),
    MR fingerprinting using fast imaging with steady state precession (FISP) with spiral readout.
    Magn. Reson. Med., 74: 1621-1631. https://doi.org/10.1002/mrm.25559


    """

    # get maximum flip angle
    max_fa = np.array(fa_max, dtype=np.float32) - offset
    n_segments = len(max_fa)

    # build schedule
    n = np.arange(length, dtype=np.float32) + 1
    rest = np.zeros(spacing, dtype=np.float32)
    rf_schedule = np.concatenate((np.sin(n * np.pi / length) * max_fa[0], rest))

    for i in range(1, n_segments):
        segment = np.concatenate((np.sin(n * math.pi / length) * max_fa[i], rest))
        rf_schedule = np.concatenate((rf_schedule, segment))

    # add recovery
    if recovery > 0:
        rf_schedule = np.concatenate(
            (rf_schedule, np.zeros(recovery, dtype=np.float32))
        )

    # add back offset
    rf_schedule += offset

    return torch.as_tensor(rf_schedule)
