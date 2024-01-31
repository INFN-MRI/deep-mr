"""
Functions to generate qMR Shepp-Logan phantom.

References:
    https://github.com/mckib2/phantominator

"""

__all__ = ["mr_shepp_logan"]

from dataclasses import asdict

import numpy as np
import torch

from . import tissue_classes


def mr_shepp_logan(npix, nslices, B0=3.0, model="single"):
    """
    Generate a Shepp Logan phantom with a given shape.

    Parameters
    ----------
    npix : Iterable[int]
        In-plane matrix size.
    nslices : int
        Number of slices.
    B0 : float, optional
        Static field strength in T. The default is 3.0.
    model : str, optional
        Signal mode. Can be "single", "bm" (Bloch-McConnell), "mt" (Magnetization Transfer)
        or "bm-mt" (Bloch-McConnell + Magnetization Transfer). The default is "single".

    Returns
    -------
    tissue_mask : TYPE
        DESCRIPTION.
    mrtp : TYPE
        DESCRIPTION.
    emtp : TYPE
        DESCRIPTION.

    """
    assert model in [
        "single",
        "bm",
        "mt",
        "bm-mt",
    ], (
        f"Error! signal model = {model} not recognized; allowed values are 'single',"
        " 'mt', 'bm' and 'bm-mt''"
    )

    # tissue classes
    classes = tissue_classes.__all__
    classes = ["tissue_classes." + cl for cl in classes]

    # get shape
    if nslices != 1:
        shape = [npix, npix, npix]
    else:
        shape = [1, npix, npix]

    # prepare tissue masks
    tissue_mask = _shepp_logan_segmentation(shape)

    # fix axis order
    ax = list(range(len(tissue_mask.shape)))
    ax.reverse()
    tissue_mask = tissue_mask.transpose(*ax).swapaxes(-1, -2)

    # get slices
    if nslices != 1:
        center = int(npix // 2)
        width = int(nslices // 2)
        tissue_mask = tissue_mask[center - width : center + width]

    # prepare tissue parameters
    tissue_params = [asdict(eval(cl)(n_atoms=1, B0=B0, model=model)) for cl in classes]

    # get mr tissue properties (mrtp)
    mrtp = {
        k: []
        for k in [
            "M0",
            "T1",
            "T2",
            "T2star",
            "chemshift",
            "D",
            "v",
        ]
    }
    mrtp["bm"] = {}
    mrtp["mt"] = {}
    bm = {"T1": [], "T2": [], "chemshift": [], "k": [], "weight": []}
    mt = {"k": [], "weight": []}

    emtp = {"chi": [], "sigma": [], "epsilon": []}
    # "chi": np.zeros(tissue_mask.shape, np.float32),
    # "sigma": np.zeros(tissue_mask.shape, np.float32),
    # "epsilon": np.zeros(tissue_mask.shape, np.float32),
    # }
    for n in range(len(tissue_params)):
        par = tissue_params[n]
        for k in [
            "M0",
            "T1",
            "T2",
            "T2star",
            "chemshift",
            "D",
            "v",
        ]:
            mrtp[k].append(par[k])
        if par["bm"]:
            for k in ["T1", "T2", "chemshift", "k", "weight"]:
                bm[k].append(par["bm"][k])
        if par["mt"]:
            for k in ["k", "weight"]:
                mt[k].append(par["mt"][k])

        # emtp["chi"][tissue_mask == n] = par["chi"]
        # emtp["sigma"][tissue_mask == n] = par["sigma"]
        # emtp["epsilon"][tissue_mask == n] = par["epsilon"]
        emtp["chi"].append(par["chi"][0])
        emtp["sigma"].append(par["sigma"][0])
        emtp["epsilon"].append(par["epsilon"][0])

    # concatenate mrtp
    for k in [
        "M0",
        "T1",
        "T2",
        "T2star",
        "chemshift",
        "D",
        "v",
    ]:
        mrtp[k] = np.concatenate(mrtp[k], axis=0)
    # concatenate bm and mt
    if par["bm"]:
        for k in ["T1", "T2", "chemshift", "k", "weight"]:
            bm[k] = np.concatenate(bm[k], axis=0)
        mrtp["bm"] = bm
    if par["mt"]:
        for k in ["k", "weight"]:
            mt[k] = np.concatenate(mt[k], axis=0)
        mrtp["mt"] = mt

    return torch.as_tensor(tissue_mask.copy(), dtype=int).squeeze(), mrtp, emtp


def _shepp_logan_segmentation(shape):
    # initialize shepp logan
    discrete_model = np.round(_shepp_logan(shape, dtype=np.float32)).astype(np.int32)

    # collapse vessels rois, csf rois and re-order indexes
    out = discrete_model.copy()
    out[discrete_model == 1] = 8  # blood
    out[discrete_model == 2] = 8  # blood
    out[discrete_model == 3] = 8  # blood
    out[discrete_model == 4] = 8  # blood
    out[discrete_model == 5] = 4  # csf
    out[discrete_model == 6] = 4  # csf
    out[discrete_model == 7] = 2  # white matter
    out[discrete_model == 8] = 3  # gray matter
    out[discrete_model == 9] = 1  # fat

    return out


# %% local utils
def _shepp_logan(shape, dtype=np.complex64):
    out = phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, dtype)

    if len(out.shape) == 3:
        out = out.transpose(1, 2, 0)
        out = np.flip(out, axis=0)
    else:
        out = np.flip(out, axis=-2)

    return out


sl_amps = [9, 8, 7, 6, 5, 4, 3, 2, 1]


sl_scales = [
    [0.6900, 0.920, 0.810],  # white big
    [0.6624, 0.874, 0.780],  # gray big
    [0.5524, 0.774, 0.680],  # gray big
    [0.1100, 0.310, 0.220],  # right black
    [0.1600, 0.410, 0.280],  # left black
    [0.1500, 0.180, 0.410],  # gray center blob
    [0.0230, 0.023, 0.020],  # left small dot
    [0.0230, 0.023, 0.020],  # mid small dot
    [0.0460, 0.046, 0.050],
]


sl_offsets = [
    [0.0, 0.0, 0],
    [0.0, 0.0, 0],
    [0.0, 0.0, 0],
    [0.22, 0.0, 0],
    [-0.22, 0.0, 0],
    [0.0, 0.35, 0],
    [-0.06, -0.605, 0],
    [0.0, -0.606, 0],
    [0.08, -0.605, 0],
]


sl_angles = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [-18, 0, 10],
    [18, 0, 10],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]


def phantom(shape, amps, scales, offsets, angles, dtype):
    """Generate a cube of given shape using a list of ellipsoid parameters."""
    if len(shape) == 2:
        ndim = 2
        shape = (1, shape[-2], shape[-1])
    elif len(shape) == 3:
        ndim = 3
    else:
        raise ValueError("Incorrect dimension")

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[
        -(shape[-3] // 2) : ((shape[-3] + 1) // 2),
        -(shape[-2] // 2) : ((shape[-2] + 1) // 2),
        -(shape[-1] // 2) : ((shape[-1] + 1) // 2),
    ]

    coords = np.stack(
        (
            x.ravel() / shape[-1] * 2,
            y.ravel() / shape[-2] * 2,
            z.ravel() / shape[-3] * 2,
        )
    )

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):
        ellipsoid(amp, scale, offset, angle, coords, out)

    if ndim == 2:
        return out[0, :, :]
    else:
        return out


def ellipsoid(amp, scale, offset, angle, coords, out):
    """
    Generate a cube containing an ellipsoid defined by its parameters.

    If out is given, fills the given cube instead of creating a new
    one.

    """
    R = rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / np.reshape(
        scale, (3, 1)
    )
    r2 = np.sum(coords**2, axis=0).reshape(out.shape)

    out[r2 <= 1] = amp


def rotation_matrix(angle):
    # rotation parameters
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))

    # build rotation matris
    alpha = [
        [
            cpsi * cphi - ctheta * sphi * spsi,
            cpsi * sphi + ctheta * cphi * spsi,
            spsi * stheta,
        ],
        [
            -spsi * cphi - ctheta * sphi * cpsi,
            -spsi * sphi + ctheta * cphi * cpsi,
            cpsi * stheta,
        ],
        [stheta * sphi, -stheta * cphi, ctheta],
    ]

    return np.array(alpha)
