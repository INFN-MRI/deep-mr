"""Configuration utils."""

__all__ = [
    "config_cartesian_2D",
    "config_cartesian_3D",
    "config_stack",
    "config_projection",
]


import numpy as np

from .nyquist import radial_fov


# %% cartesian
def config_cartesian_2D(fov, shape, accel, kwargs):
    """
    Prepare arguments for Cartesian 2D design.

    Args:
        fov (tuple of floats): Field of view (FOVx, FOVy) in [mm].
            If scalar, assume isotropic FOV.
        shape (tuple of ints): Matrix size (nx, ny, nechoes, nframes).
            If scalar, assume square matrix and nechoes=nframes=1.
        accel (tuple): Acceleration factor (Ry, PF).
            Ranges from (1, 1) (fully sampled) to (ny, 0.75).

    Kwargs:
        acs_shape (int): number of ACS lines.
        ordering (str): acquire views sequentially ("sequentially") or not ("interleaved") when nframes > 1.
            Default to "interleaved".

    Returns:
        (tuple of tuples): [FOVx, FOVy] for readout and phase plan design.
        (tuple of tuples): [read_shape, phase_shape] for readout (nx, necho) and phase plan (ny, nframes) design.
        (tuple of floats): [Ry, PF] accelerations for phase plan design.
        (tuple of dicts): [readargs, phaseargs] additional arguments for readout and phase plan design.
        (str): view ordering.

    """
    # parse fov
    if np.isscalar(fov):
        FOVx = fov
        FOVy = fov
    else:
        FOVx, FOVy = fov

    # parse shape
    if np.isscalar(shape):
        shape = [shape, shape]
    tmp = [None, None, 1, 1]  # (nx, ny, nechoes, nframes)
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp

    # actual shape parsing
    read_shape, phase_shape = [shape[0], shape[-2]], [shape[1], shape[-1]]

    # acceleration parsing
    if np.isscalar(accel):
        accel = [accel]
    tmp = [1, 1.0]  # (Ry, PF)
    for n in range(len(accel)):
        tmp[n] = accel[n]
    accel = tmp
    Ry, pf = accel
    assert pf >= 0.75, f"Partial Fourier factor must be higher than 0.75 (found {pf})"

    # get phase encoding kwargs
    phaseargs = {}
    if "gmax" in kwargs:
        phaseargs["gmax"] = kwargs["gmax"]
    if "smax" in kwargs:
        phaseargs["smax"] = kwargs["smax"]
    if "gdt" in kwargs:
        phaseargs["gdt"] = kwargs["gdt"]
    if "rew_derate" in kwargs:
        phaseargs["rew_derate"] = kwargs["rew_derate"]
    if "acs_shape" in kwargs:
        phaseargs["acs_shape"] = kwargs["acs_shape"]
        kwargs.pop("acs_shape")
    else:
        phaseargs["acs_shape"] = None

    if "shift" in kwargs:  # caipirinha shift
        phaseargs["shift"] = kwargs["shift"]
        kwargs.pop("shift")
    else:
        phaseargs["shift"] = 0

    # get in-plane kwargs
    readargs = kwargs

    # ordering
    if "ordering" in kwargs:
        ordering = kwargs["ordering"]
        kwargs.pop("ordering")
    else:
        ordering = "interleaved"

    return (
        [FOVx, FOVy],
        [read_shape, phase_shape],
        [Ry, pf],
        [readargs, phaseargs],
        ordering,
    )


def config_cartesian_3D(fov, shape, accel, kwargs):
    """
    Prepare arguments for Cartesian 3D design.

    Args:
        fov (tuple of floats): Field of view (FOVx, FOVy, FOVz) in [mm].
            If len(fov) == 2, assume squared in-plane FOV (i.e., [FOVp, FOVz], FOVx=FOVy=FOVp).
        shape (tuple of ints): Matrix size (nx, ny, nz, nechoes, nframes).
            If scalar, assume cubic matrix and nechoes=nframes=1.
            If len(shape) == 2, assume squared in-plane matrix (i.e., [npix, nz], nx=ny=npix).
        accel (tuple): Acceleration factor (Ry, PF).
            Ranges from (1, 1) (fully sampled) to (ny, 0.75).

    Kwargs:
        acs_shape (int): number of ACS lines (ACSy, ACSz). If scalar, assume squared ACS.
        ordering (tuple of str): loop ordering from inner to outer ("frames", "views", "slices") or views distribution ("random", "poisson").
            If nframes > 1 and len(ordering) == 2, assume sequential frames (i.e., innermost). Defaults to ("views", "slices").

    Returns:
        (tuple of tuples): [FOVx, [FOVy, FOVz]] for readout and phase plan design.
        (tuple of tuples): [read_shape, phase_shape] for readout (nx, necho) and phase plan (ny, nz, nframes) design.
        (tuple of tuples): [[Ry], [Rz, PF]] accelerations for phase plan design.
        (tuple of dicts): [readargs, phaseargs] additional arguments for readout and phase plan design.
        (str): view ordering.

    """
    # parse fov
    if np.isscalar(fov):
        FOVx = fov
        FOVy = fov
        FOVz = fov
    elif len(fov) == 2:
        FOVp, FOVz = fov
        FOVx = FOVp
        FOVy = FOVp
    else:
        FOVx, FOVy, FOVz = fov

    # parse shape
    if np.isscalar(shape):
        shape = [shape, shape, shape]
    elif len(shape) == 2:
        shape = [shape[0], shape[1], shape[2]]
    tmp = [None, None, None, 1, 1]  # (nx, ny, nz, nechoes, nframes)
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp

    # actual shape parsing
    read_shape, phase_shape = [shape[0], shape[3]], [shape[1], shape[2], shape[-1]]

    # acceleration parsing
    if np.isscalar(accel):
        accel = [accel]
    tmp = [1, 1, 1.0]  # (Ry, Rz, PF)
    for n in range(len(accel)):
        tmp[n] = accel[n]
    accel = tmp
    Ry, Rz, pf = accel
    assert pf >= 0.75, f"Partial Fourier factor must be higher than 0.75 (found {pf})"

    # get phase encoding kwargs
    phaseargs = {}
    if "gmax" in kwargs:
        phaseargs["gmax"] = kwargs["gmax"]
    if "smax" in kwargs:
        phaseargs["smax"] = kwargs["smax"]
    if "gdt" in kwargs:
        phaseargs["gdt"] = kwargs["gdt"]
    if "rew_derate" in kwargs:
        phaseargs["rew_derate"] = kwargs["rew_derate"]
    if "acs_shape" in kwargs:
        acs_shape = kwargs["acs_shape"]
        if np.isscalar(acs_shape):
            acs_shape = [acs_shape, acs_shape]
        phaseargs["acs_shape"] = acs_shape
    else:
        phaseargs["acs_shape"] = None

    if "shift" in kwargs:  # caipirinha shift
        phaseargs["shift"] = kwargs["shift"]
        kwargs.pop("shift")
    else:
        phaseargs["shift"] = 0

    # get in-plane kwargs
    readargs = kwargs

    # get ordering
    if "ordering" in kwargs:
        ordering = kwargs["ordering"]
        kwargs.pop("ordering")
    else:
        ordering = "interleaved"

    return (
        [FOVx, [FOVy, FOVz]],
        [read_shape, phase_shape],
        [Ry, Rz, pf],
        [readargs, phaseargs],
        ordering,
    )


# %% noncartesian: stack
def config_stack(fov, shape, accel, kwargs):
    """
    Prepare arguments for Non Cartesian stack-of-trajectory design.

    Args:
        fov (tuple of floats): Field of view (FOVr, FOVy) in [mm].
        shape (tuple of ints): Matrix size (nr, nz , nechoes, nframes).
            If scalar, assume cubic matrix and nechoes=nframes=1.
        accel (tuple): Acceleration factor (Rplane, Rz, PF).
            Ranges from (1, 1, 1) (fully sampled) to (nshots, nz, 0.75).

    Kwargs:
        acs_shape (tuple of ints): number of ACS lines (ACSr, ACSz). If scalar, assume cubic ACS.
        ordering (str): acquire slices sequentially ("sequentially") or not ("interleaved") when nframes > 1.
            Default to "interleaved".

    Returns:
        (tuple of tuples): [FOVr, FOVy] for plane and phase plan design.
        (tuple of tuples): [plane_shape, phase_shape] for plane and phase plan design.
        (tuple of tuples): [[Rplane], [Rz, PF]] accelerations for plane and phase plan design.
        (tuple of dicts): [planeargs, zargs] additional arguments for plane and phase plan design.
        (str): slice ordering.

    """
    # parse fov
    if np.isscalar(fov):
        FOVr = fov
        FOVz = fov
    else:
        FOVr, FOVz = fov

    # parse shape
    if np.isscalar(shape):
        shape = [shape]
    assert len(shape) >= 2, "Please provide at least (npix, nz) as 'shape' argument."
    tmp = [None, None, 1, 1]  # (npix, nz, nechoes, nframes)
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp

    # actual shape parsing
    read_shape, phase_shape = [shape[0], shape[2], shape[3]], shape[1]

    # acceleration parsing
    if np.isscalar(accel):
        accel = [accel]
    tmp = [1, 1, 1.0]  # (Rplane, Rz, PF)
    for n in range(len(accel)):
        tmp[n] = accel[n]
    accel = tmp
    Rplane, Rz, pf = accel
    assert pf >= 0.75, f"Partial Fourier factor must be higher than 0.75 (found {pf})"

    # get phase encoding kwargs
    zargs = {}
    if "tilt" in kwargs:
        zargs["tilt"] = kwargs["tilt"]
        kwargs.pop("tilt")
    else:
        zargs["tilt"] = False
    if "gmax" in kwargs:
        zargs["gmax"] = kwargs["gmax"]
    if "smax" in kwargs:
        zargs["smax"] = kwargs["smax"]
    if "gdt" in kwargs:
        zargs["gdt"] = kwargs["gdt"]
    if "rew_derate" in kwargs:
        zargs["rew_derate"] = kwargs["rew_derate"]
    if "acs_shape" in kwargs:
        acs_shape = kwargs["acs_shape"]
        if np.isscalar(acs_shape):
            acs_shape = [acs_shape]
        if accel[1] != 1:
            assert (
                len(acs_shape) == 2
            ), "Please provide ACS shape both for in-plane and z directions."
        else:
            acs_shape.append(shape[1])
    else:
        acs_shape = None
    if acs_shape is not None:
        zargs["acs_shape"] = acs_shape[1]
        kwargs["acs_shape"] = acs_shape[0]

    # get in-plane kwargs
    planeargs = kwargs

    # ordering
    if "ordering" in kwargs:
        ordering = kwargs["ordering"]
        kwargs.pop("ordering")
    else:
        ordering = "interleaved"

    return (
        [FOVr, FOVz],
        [read_shape, phase_shape],
        [Rplane, [Rz, pf]],
        [planeargs, zargs],
        ordering,
    )


# %% noncartesian: proj
def config_projection(fov, shape, accel, kwargs):
    """
    Prepare arguments for Non Cartesian 3D projection design.

    Args:
        fov (floats): Field of view in [mm].
        shape (int): Matrix size (npix, nframes, nechoes).
            If scalar, assume nechoes=nframes=1.
        accel (tuple): Acceleration factor (Rplane, Rangle).
            Ranges from (1, 1) (fully sampled) to (nshots, nplanes).

    Kwargs:
        acs_shape (int): ACS region size.
        ordering (str): acquire slices sequentially ("sequentially") or not ("interleaved") when nframes > 1.
            Default to "interleaved".

    Returns:
        (float): FOV for in-plane trajectory design
        (int): matrix size for in-plane trajectory design
        (tuple of floats): [Rplanes, Rangle] accelerations for in-plane and rotation design.
        (tuple of dicts): [planeargs, rotargs] additional arguments for plane and rotation design.
        (str): rotation ordering.

    """
    # parse shape
    if np.isscalar(shape):
        shape = [shape]
    tmp = [None, 1, 1]  # (npix, nechoes, nframes)
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp

    # parse for later
    npix, nechoes, nframes = shape

    # compute matrix shape
    res = fov / npix  # m

    # calculate Nyquist sampling
    _, ptheta = radial_fov(fov, res)

    # unpack Nyquist params
    nplanes, _ = ptheta

    # ordering
    if (
        "ordering" in kwargs
    ):  # allowed are (sequential, interleaved, shuffled, multiaxis-shuffled)
        ordering = kwargs["ordering"]
        allowed = ["sequential", "interleaved", "shuffle", "multiaxis-shuffle"]
        msg = f"Ordering not recognized - allowed values are {allowed}"
        assert ordering in allowed, msg
        kwargs.pop("ordering")
    else:
        if nframes > nplanes:
            ordering = "multiaxis-shuffle"
        else:
            ordering = "interleaved"

    # correct nplanes
    if nframes > nplanes:
        nplanes = int(np.ceil(nplanes / nframes) * nframes)

    # get rotation kwargs
    rotargs = {}
    if "tilt_type" in kwargs:
        tilt_type = kwargs["tilt_type"]
        if isinstance(tilt_type, (list, tuple)) is False:
            tilt_type = [
                tilt_type,
                "uniform",
            ]  #  assume provided order is for through-plane rotation
        # assign to rotations
        rotargs["tilt_type"] = tilt_type
        kwargs.pop("tilt_type")
    else:
        rotargs["tilt_type"] = ["uniform", "tgas"]
    if "dummy" in kwargs:
        rotargs["dummy"] = kwargs["dummy"]
        kwargs.pop("dummy")
    else:
        if nframes > 1:
            rotargs["dummy"] = True
        else:
            rotargs["dummy"] = False

    # get in-plane kwargs
    planeargs = kwargs

    # parsing acceleration
    if nframes >= nplanes:
        default_angular_usf = nplanes
    else:
        default_angular_usf = nframes

    if np.isscalar(accel):
        accel = [accel]
    tmp = [None, default_angular_usf]  # (Rplane, Rangular)
    for n in range(len(accel)):
        tmp[n] = accel[n]
    accel = tmp

    return (
        fov,
        [npix, [nplanes, nechoes, nframes]],
        accel,
        [planeargs, rotargs],
        ordering,
    )
