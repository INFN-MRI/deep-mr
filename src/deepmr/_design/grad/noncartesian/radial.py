"""2D Radial trajectory design routine."""

__all__ = ["radial"]

import copy

import numpy as np

from .. import utils

gamma_bar = 42.575 * 1e6  # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar  # rad / T / s


def radial(fov, shape, accel=1, nspokes=None, osf=1.0, **kwargs):
    r"""
    Design a radial trajectory.

    Args:
        fov (float): field of view in [mm].
        shape (tuple of ints): matrix size (plane, echoes=1, frames=1).
        accel (int): in-plane acceleration. Ranges from 1 (fully sampled) to nintl.
        nspokes (int): number of spokes to be designed. Defaults to Nyquist.
        osf (float): oversampling factor along readout. Defaults to 1.

    Kwargs:
        tilt_type (str): tilt of the shots.
        tilt (bool): if True, keep rotating the spokes through echo train. If false, keep same spoke for each echo (defaults to False).
        acs_shape (int): matrix size for autocalibration (defaults to None).
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
    # parse defaults
    (
        gmax,
        smax,
        gdt,
        rew_derate,
        fid,
        acs_shape,
        flyback,
        _,
    ) = utils.get_cartesian_defaults(kwargs)

    # build base interleaf and rotation angles
    kr, phi, shape, acs_shape, variant = _make_radial_interleave(
        fov, shape, accel, nspokes, osf, kwargs
    )

    # get nframes and nechoes for clarity
    mtx, nechoes, nframes = shape[0], shape[1], shape[2]

    # optionally, enforce system constraint (and design base interleaf waveform)
    grad, adc, echo_idx = _make_radial_gradient(
        kr, gmax, smax, gdt, rew_derate, fid, flyback
    )

    # compute density compensation factor
    dcf = utils.analytical_dcf(kr.T)

    # compute timing
    if grad is None:
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, kr, grad)
    else:
        gtmp = np.concatenate((grad["pre"], grad["read"], grad["rew"]), axis=-1)
        te, t = utils.calculate_timing(nechoes, echo_idx, gdt, kr, gtmp)

    # compute loop indexes (kt, kecho)
    kt, phi = utils.broadcast_tilt(np.arange(nframes), phi, loop_order="new-first")
    kecho, phi = utils.broadcast_tilt(np.arange(nechoes), phi, loop_order="old-first")

    # prepare acs
    acs = utils.extract_acs(kr, dcf, shape, acs_shape)

    # prepare grad struct
    if grad is None:
        grad = {"read": None, "rew": None, "pre": None, "rot": None}
    else:
        grad["rot"] = phi

    # prepare compressed structure
    R = utils.angleaxis2rotmat(phi, [0, 0, 1])
    compressed = {
        "kr": utils.scale_traj(kr),
        "kecho": kecho,
        "kt": kt,
        "rot": R,
        "t": t,
        "te": te,
        "mtx": [mtx, mtx],
        "dcf": dcf,
        "adc": adc,
        "acs": copy.deepcopy(acs),
    }

    # expand
    kr = utils.projection(kr, R).astype(np.float32).transpose(1, 2, 0)
    acs["kr"] = utils.projection(acs["kr"], R).astype(np.float32).transpose(1, 2, 0)

    # prepare trajectory structure
    traj = {
        "kr": utils.scale_traj(kr),
        "kecho": kecho,
        "kt": kt,
        "t": t,
        "te": te,
        "mtx": [mtx, mtx],
        "dcf": dcf,
        "adc": adc,
        "acs": acs,
        "compressed": compressed,
    }
    if nechoes == 1:
        traj.pop("kecho", None)
        traj["compressed"].pop("kecho", None)
    if nframes == 1:
        traj.pop("kt", None)
        traj["compressed"].pop("kt", None)

    # prepare protocol header
    # prot = {"kr": utils.scale_traj(kr), "phi": phi, "kecho": kecho, "kt": kt, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": acs_shape}

    # plot reports
    # TODO

    # return traj, grad, prot
    return traj, grad


# %% local utils
def _make_radial_interleave(fov, shape, accel, nspokes, osf, kwargs):
    if "tilt_type" in kwargs:
        tilt_type = kwargs["tilt_type"]
    else:
        tilt_type = "uniform"
    if "tilt" in kwargs:
        tilt = kwargs["tilt"]
    else:
        tilt = False
    if "variant" in kwargs:
        variant = kwargs["variant"]
    else:
        variant = "fullspoke"
    if "acs_shape" in kwargs:
        acs_shape = kwargs["acs_shape"]
    else:
        acs_shape = None

    # check variant
    message = (
        f"Error! Unrecognized spiral variant = {variant} - valid types are 'fullspoke',"
        " 'center-out'"
    )
    assert variant in ["fullspoke", "center-out"], message

    # shape
    tmp = [None, 1, 1]  # (mtx, nechoes, nframes)
    if np.isscalar(shape):
        shape = [shape]
    for n in range(len(shape)):
        tmp[n] = shape[n]
    shape = tmp

    # get nframes and nechoes
    mtx, nechoes, nframes = shape

    # transform to array
    mtx = np.array(mtx)

    # cast dimensions
    fov *= 1e-3  # mm -> m

    # get resolution
    res = fov / mtx

    # calculate Nyquist sampling
    pr, ptheta = utils.radial_fov(fov, res)

    # unpack Nyquist params
    nr, _ = pr
    ntheta, _ = ptheta

    # set default spokes if not provided
    if nspokes is None:
        nspokes = ntheta

    # convert nspokes to array
    nspokes = np.array(nspokes)

    # compute number of radial samples
    nsamp = int(osf * nr)

    # calculate maximum k space radius
    kmax = np.pi / res

    # calculate k space radius
    if variant == "fullspoke":
        base_k = np.linspace(0.0, 2.0, nsamp) - 1
        halfplane = True
    else:
        base_k = np.linspace(0.0, 1.0, nsamp // 2)
        halfplane = False

    # rescale
    base_k *= kmax

    # convert to 2D
    base_k = np.stack((base_k, 0 * base_k), axis=0)

    # generate angles
    if halfplane:
        nspokes = 2 * nspokes
    angles = utils.make_tilt(tilt_type, nspokes, accel, nframes, nechoes, tilt)

    return base_k, angles, shape, acs_shape, variant


def _make_radial_gradient(kr, gmax, smax, gdt, rew_derate, fid, flyback):
    # build gradient
    grad, adc, echo_idx = utils.make_cartesian_gradient(
        kr, gmax, smax, gdt, rew_derate, fid, flyback
    )

    # postprocess
    if grad is not None:
        grad["rew"] = -np.flip(grad["pre"], axis=-1)

    return grad, adc, echo_idx
