"""
Functions to generate Brainweb phantom.

References:
    https://brainweb.bic.mni.mcgill.ca/
    https://github.com/casperdcl/brainweb

"""

import gzip
import logging
import os
import shutil
from dataclasses import asdict
from os import path

import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress the warnings from urllib3
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from . import tissue_classes

def create_brainweb(
    npix: int,
    nslices: int = 1,
    B0: float = 3.0,
    model: str = "single",
    fuzzy: bool = False,
    idx: int = 0,
):
    """
    Brainweb phantom with MR tissue parameters.

    Args:
        Args:
            npix (int or tuple of ints): shape, can be scalar or tuple (ny, nx). If scalar, assume squared FOV.
            nslices (int, optional): Number of slices in the phantom (default is 1).
            B0 (float, optional): Static field strength in units of [T]; ignored if `mr` is False (default is 3.0 T).
            model (str):  signal model to be used:
                    - "single": single pool T1 / T2.
                    - "bm": 2-pool model with exchange (iew pool + mw pool).
                    - "mt": 2-pool model with exchange( iew/mw pool + zeeman semisolid pool)
                    - "bm-mt": 3-pool model (iew pool, mw pool, zeeman semisolid pool; mw exchange with iew and ss).
            fuzzy (bool): if true, use fuzzy model for different classes; otherwise, use discrete model (defaults: False).
            idx (int): subject index (0 to 19). Defaults to 0.

    Returns:
        (array): if fuzzy = True, 4D phantom segmentation. Otherwise, 3D discrte segmentation.
        (list): list of dictionaries containing 1) free water T1/T2/T2*/ADC/v, 2) bm/mt/ihmt T1/T2/fraction, 3) exchange matrix
                for each class (index along the list correspond to value in segmentation mask)
        (list): list of dictionaries containing electromagnetic tissue properties for each class.

    Example:
        >>> seg, mrtp, emtp = create_brainweb_logan(npix=256, nslices=3, B0=1.5, model="mt")

    """
    assert model in [
        "single",
        "bm",
        "mt",
        "bm-mt",
    ], (
        f"Error! signal model = {model} not recognized; allowed values are 'single',"
        " 'mt', 'bm' and 'bm-mt'"
    )

    # tissue classes
    classes = tissue_classes.__all__
    classes = ["tissue_classes." + cl for cl in classes]

    if np.isscalar(npix):
        npix = [npix, npix]

    # get shape
    shape = 3 * [npix[0]]

    # prepare tissue masks
    tissue_mask = _brainweb_segmentation(shape, idx)

    # get slices and phase fov
    center = int(npix[0] // 2)
    xwidth = int(npix[-1] // 2)
    if nslices != 1:
        zwidth = int(nslices // 2)
        tissue_mask = tissue_mask[
            :, center - zwidth : center + zwidth, :, center - xwidth : center + xwidth
        ]
    else:
        tissue_mask = tissue_mask[:, [center], :, center - xwidth : center + xwidth]

    # get discrete model
    discrete_model = np.argmax(tissue_mask, axis=0)

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

    emtp = {
        "chi": np.zeros(discrete_model.shape, np.float32),
        "sigma": np.zeros(discrete_model.shape, np.float32),
        "epsilon": np.zeros(discrete_model.shape, np.float32),
    }
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

        emtp["chi"][discrete_model == n] = par["chi"]
        emtp["sigma"][discrete_model == n] = par["sigma"]
        emtp["epsilon"][discrete_model == n] = par["epsilon"]

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

    # assign units
    # mrtp["T1"] = utils.assign_unit(mrtp["T1"], "ms")
    # mrtp["T2"] = utils.assign_unit(mrtp["T2"], "ms")
    # mrtp["T2star"] = utils.assign_unit(mrtp["T2star"], "ms")
    # mrtp["chemshift"] = utils.assign_unit(mrtp["chemshift"], "Hz")
    # mrtp["D"] = utils.assign_unit(mrtp["D"], "um**2 / ms")
    # mrtp["v"] = utils.assign_unit(mrtp["v"], "cm / s")
    # mrtp["k"] = utils.assign_unit(mrtp["k"], "1 / s")
    # emtp["sigma"] = utils.assign_unit(emtp["sigma"], "S / m")

    # if "bm" in mrtp and mrtp["bm"]:
    #     mrtp["bm"]["T1"] = utils.assign_unit(mrtp["bm"]["T1"], "ms")
    #     mrtp["bm"]["T2"] = utils.assign_unit(mrtp["bm"]["T2"], "ms")
    #     mrtp["bm"]["chemical_shift"] = utils.assign_unit(
    #         mrtp["bm"]["chemical_shift"], "Hz"
    #     )

    if fuzzy:
        return tissue_mask, mrtp, emtp
    else:
        return discrete_model.astype(np.float32), mrtp, emtp


def _brainweb_segmentation(shape, idx):
    # download files
    fuzzy_model = get_subj(idx)

    # load brain model for the given subject
    values = list(fuzzy_model.values())
    values = np.stack(values, axis=0)

    # resize to desired matrix size
    scale = np.asarray(shape) / np.asarray(values.shape[-3:])
    scale = [1.0] + scale.tolist()
    values = zoom(values, scale)

    # normalize such as total fraction = 1
    values = values / values.sum(axis=0)
    values = [val for val in values]

    # prepare model
    model = dict(zip(list(fuzzy_model.keys()), values))

    # sort for output
    return np.stack(
        [
            model["bck"],
            model["ves"],
            model["csf"],
            model["wht"],
            model["gry"],
            model["fat"],
            model["skl"],
        ],
        axis=0,
    )


# %% local utils
subj_id = [
    "04",
    "05",
    "06",
    "18",
    "20",
    "38",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
]

tissue_id = [
    "bck",
    "csf",
    "gry",
    "wht",
    "fat",
    "mus",
    "m-s",
    "skl",
    "ves",
    "fat2",
    "dura",
    "mrw",
]

SUBJ_LINKS = [
    {
        i: (
            "http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject"
            + j
            + "_"
            + i
            + "&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D"
        )
        for i in tissue_id
    }
    for j in subj_id
]

log = logging.getLogger(__name__)


def get_file(fname, origin, cache_dir):
    """
    Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.brainweb`, and given the filename `fname`.
    The final location of a file
    `field_A.bin.gz` would therefore be `~/.brainweb/field_A.bin.gz`.

    Vaguely based on:
    https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py

    Args:
        fname  Name of the file. If an absolute path
               `/path/to/file.txt` is specified the file will be saved at that
               location.
        origin  Original URL of the file.
        cache_dir  Location to store cached files, when None it defaults to `~/.brainweb`.

    Returns:
        Path to the downloaded file
    """
    fpath = path.join(cache_dir, fname)

    if not path.exists(fpath):
        print(f"Downloading {fpath} from {origin}")
        try:
            # download
            session = requests.Session()
            session.verify = False

            with requests.get(origin, stream=True, verify=False) as r:
                with open(fpath, "wb") as fo:
                    shutil.copyfileobj(r.raw, fo)

        except (Exception, KeyboardInterrupt):
            if path.exists(fpath):
                os.remove(fpath)
            raise

    return fpath


def load_subj(fpath, value):
    """
    Uncompress the specified file and read the binary output as an array.
    """
    with gzip.open(fpath) as fi:
        data = np.frombuffer(fi.read(), dtype=np.uint16) / 4096

    # reshape
    data = data.reshape((362, 434, 362))

    # pad
    data = np.pad(data, ((36, 36), (0, 0), (36, 36)), constant_values=value)

    # cast
    data = data.astype(np.float32)

    # flip
    data = np.flip(data, axis=1).copy()

    return data


def get_subj(idx, cache_dir=None):
    """
    Returns list of files which can be `numpy.load`ed
    """
    # create cache dir
    if cache_dir is None:
        cache_dir = path.join("~", ".brainweb")

    cache_dir = path.expanduser(cache_dir)

    if not path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except Exception:
            log.warn("cannot create:" + cache_dir)

    if not os.access(cache_dir, os.W_OK):
        cache_dir = path.join("/tmp", ".brainweb")
        if not path.exists(cache_dir):
            os.makedirs(cache_dir)

    # get subject
    SUBJ_LINK = SUBJ_LINKS[idx]

    classes = list(SUBJ_LINK.keys())
    files = {}

    # download
    for c in classes:
        fname = "sub" + str(idx).zfill(2) + "_" + c + ".bin.gz"
        url = SUBJ_LINK[c]
        files[c] = get_file(fname, url, cache_dir=cache_dir)

    # convert to niftis
    fpath = path.join(cache_dir, "sub" + str(idx).zfill(2) + ".nii.gz")

    if path.exists(fpath):
        # load data
        data = nib.load(fpath)
        data = data.get_fdata().transpose()

        # create subject dict
        subj = dict(zip(["bck", "csf", "gry", "wht", "fat", "skl", "ves"], data))

    else:
        background_value = {k: 0 for k in files.keys()}
        background_value["bck"] = 1

        # load brain model for the given subject
        data = [load_subj(files[k], background_value[k]) for k in files.keys()]

        # create subject dict
        subj = dict(zip(classes, data))

        # collapse extra-brain tissues into fat
        subj["fat"] += subj["mus"]
        subj.pop("mus", None)

        subj["fat"] += subj["m-s"]
        subj.pop("m-s", None)

        subj["fat"] += subj["fat2"]
        subj.pop("fat2", None)

        subj["fat"] += subj["mrw"]
        subj.pop("mrw", None)

        subj["fat"] += subj["dura"]
        subj.pop("dura", None)

        # save values
        values = np.stack(list(subj.values()), axis=0).transpose()
        cache = nib.Nifti1Image(values, np.eye(4))  # Save axis for data (just identity)
        cache.to_filename(fpath)  # Save as NiBabel file

    return subj
