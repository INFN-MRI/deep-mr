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
import requests

from dataclasses import asdict
from os import path
from pathlib import Path
# from urllib3.exceptions import InsecureRequestWarning

# Suppress the warnings from urllib3
# requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

import numpy as np
import torch

from scipy.ndimage import zoom

import nibabel as nib

from . import tissue_classes

def brainweb(npix, nslices=1, B0=3.0, idx=0, cache_dir=None, model="single", fuzzy=False):
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
    tissue_mask = _brainweb_segmentation(shape, idx, cache_dir)

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

    emtp = {"chi": [], "sigma": [], "epsilon": []}
    #     "chi": np.zeros(discrete_model.shape, np.float32),
    #     "sigma": np.zeros(discrete_model.shape, np.float32),
    #     "epsilon": np.zeros(discrete_model.shape, np.float32),
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

        # emtp["chi"][discrete_model == n] = par["chi"]
        # emtp["sigma"][discrete_model == n] = par["sigma"]
        # emtp["epsilon"][discrete_model == n] = par["epsilon"]
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
        return torch.as_tensor(discrete_model.copy(), dtype=int).squeeze(), mrtp, emtp


def _brainweb_segmentation(shape, idx, cache_dir):
    # download files
    fuzzy_model = get_subj(idx, cache_dir)

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

    return np.stack(
        [
            model["bck"],
            model["fat"],
            model["wht"],
            model["gry"],
            model["csf"],
            model["ves"],
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

            with requests.get(origin, stream=True, verify=True) as r:
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
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    elif "BRAINWEB_DIR" in os.environ:
        cache_dir = Path(os.environ["BRAINWEB_DIR"])
    else:
        cache_dir = Path.home() / ".brainweb"
    
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
