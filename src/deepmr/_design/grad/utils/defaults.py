"""Default options utils."""

__all__ = ["get_noncart_defaults", "get_cartesian_defaults"]


def get_noncart_defaults(kwargs):  # noqa
    """
    Get default values for gradient design.
    """
    if "gmax" in kwargs:
        gmax = kwargs["gmax"]
    else:
        gmax = None

    if "smax" in kwargs:
        smax = kwargs["smax"]
    else:
        smax = None

    if "gdt" in kwargs:
        gdt = kwargs["gdt"]
    else:
        gdt = 4.0

    if "rew_derate" in kwargs:
        rew_derate = kwargs["rew_derate"]
    else:
        rew_derate = 0.8

    if "fid" in kwargs:
        fid = kwargs["fid"]
    else:
        fid = (4, 4)

    if "acs_shape" in kwargs:
        acs_shape = kwargs["acs_shape"]
    else:
        acs_shape = None

    if "moco_shape" in kwargs:
        moco_shape = kwargs["moco_shape"]
    else:
        moco_shape = None

    return gmax, smax, gdt, rew_derate, fid, acs_shape, moco_shape


def get_cartesian_defaults(kwargs):
    """
    Get default values for cartesian gradient design.
    """
    gmax, smax, gdt, rew_derate, fid, acs_shape, _ = get_noncart_defaults(kwargs)

    if "flyback" in kwargs:
        flyback = kwargs["flyback"]
    else:
        flyback = False

    if "shift" in kwargs:
        shift = kwargs["shift"]
    else:
        shift = 0

    return gmax, smax, gdt, rew_derate, fid, acs_shape, flyback, shift
