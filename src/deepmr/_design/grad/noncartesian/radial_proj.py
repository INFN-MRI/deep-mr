"""3D Radial-Projection trajectory design routine."""

__all__ = ["radial_proj"]

from .radial import radial

from .. import utils


def radial_proj(fov, shape, accel=1, nspokes=None, osf=1.0, **kwargs):
    r"""
    Design a 3D radial projection trajectory.

    Args:
        fov (float): field of view in [mm].
        shape (tuple of ints): matrix size (npix, echoes=1, frames=1).
        accel (tuple of ints): acceleration (Rplane, Rangular). Ranges from (1, 1) (fully sampled) to (nspokes, nplanes).
        nspokes (int): number of spokes to be designed. Defaults to Nyquist.
        osf (float): oversampling factor along readout. Defaults to 1.

    Kwargs:
        dummy (bool): If true, add a dummy repetition for driven equilibrium (defaults to True).
        ordering (str): acquire planes sequentially ("sequentially"), interleaved ("interleaved"), shuffled ("shuffle") or
            shuffle across multiple axis ("multiaxis-shuffle"). Default to "multiaxis-shuffle")."
        tilt_type (tuple of str): tilt of the shots in-plane (tilt_type[0]) and through-plane (tilt_type[1]).
            If str, assume through-plane and defaults in-plane to "uniform".
        acs_shape (tuple of ints): matrix size for calibration regions (ACSplane, ACSangular). Defaults to (None, None).
        variant (str): type of spiral. Allowed values are
                - 'fullspoke': starts at the edge of k-space and ends on the opposite side (default).
                - 'center-out': starts at the center of k-space and ends at the edge.
        gdt (float): trajectory sampling rate in [us].
        gmax (float): maximum gradient amplitude in [mT / m].
        smax (float): maximum slew rate in [T / m / s].
        rew_derate (float): derate slew rate during rewinder and z phase-encode blip by this factor, to reduce PNS. Default: 0.1.
        fid (tuple of ints): number of fid points before and after readout (defaults to (0, 0)).

    Returns:
        (dict): structure containing info for reconstruction (coordinates, dcf, matrix, timing...).
        (dict): structure containing info for acquisition (gradient waveforms...).

    Notes:
        The following values are accepted for the tilt name, with :math:`N` the number of
        partitions:

        - "uniform": uniform tilt: 2:math:`\pi / N`
        - "inverted": inverted tilt :math:`\pi/N + \pi`
        - "golden": golden angle tilt :math:`\pi(\sqrt{5}-1)/2`. For 3D, refers to through plane axis (in-plane is uniform).
        - "tiny-golden": tiny golden angle tilt 2:math:`\pi(15 -\sqrt{5})`. For 3D, refers to through plane axis (in-plane is uniform).
        - "tgas": tiny golden angle tilt with shuffling along through-plane axis (3D only)`

    """
    # parsing
    fov, shape, accel, kwargs, ordering = utils.config_projection(
        fov, shape, accel, kwargs
    )

    # get in-plane trajectory (single frame, 1 echo, )
    traj, grad = radial(fov, shape[0], 1, nspokes, osf, **kwargs[0])

    # get actual number of interleaves
    nspokes = traj["kr"].shape[0]

    # put together
    traj = utils.make_projection(
        ordering, traj["compressed"], [nspokes] + shape[1], accel, kwargs[1]
    )

    return traj, grad
