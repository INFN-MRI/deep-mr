"""This module contains NIfTI writing routines."""

__all__ = ["write"]

import json
import os
from typing import Dict

import nibabel as nib
import numpy as np

from deepmr.io.utils.geometry import *


def write(
    image: np.ndarray,
    info: Dict,
    series_description: str = None,
    filename: str = "output.nii",
    outpath: str = "./",
):
    """
    Write parametric map to dicom.

    Args:
        image: ndarray of image data to be written.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
        filename: name of the output nifti file.
        outpath: desired output path
    """
    # generate file name
    if filename.endswith(".nii") is False and filename.endswith(".nii.gz") is False:
        filename += ".nii"

    # generate output path
    outpath = os.path.abspath(outpath)

    # create output folder
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # reformat image
    image = np.flip(image.transpose(), axis=1)

    # cast image
    minval = np.iinfo(np.int16).min
    maxval = np.iinfo(np.int16).max
    image[image < minval] = minval
    image[image > maxval] = maxval
    image = image.astype(np.int16)

    if info["nifti_template"]:
        affine = info["nifti_template"]["affine"]
        header = info["nifti_template"]["header"]
        json_dict = info["nifti_template"]["json"]
        nifti = nib.Nifti1Image(image, affine, header)

    elif info["dcm_template"]:
        # we do not have json dict in this case
        json_dict = None

        # get voxel size
        dx, dy = np.array(info["dcm_template"][0].PixelSpacing).round(4)
        dz = round(float(info["dcm_template"][0].SliceThickness), 4)

        # get affine
        affine, _ = _get_nifti_affine(info["dcm_template"], image.shape[-3:])

        try:
            windowMin = 0.5 * np.percentile(image[image < 0], 95)
        except:
            windowMin = 0
        try:
            windowMax = 0.5 * np.percentile(image[image > 0], 95)
        except:
            windowMax = 0

        # write nifti
        nifti = nib.Nifti1Image(image, affine)
        nifti.header["pixdim"][1:4] = np.array([dx, dy, dz])
        nifti.header["sform_code"] = 0
        nifti.header["qform_code"] = 2
        nifti.header["cal_min"] = windowMin
        nifti.header["cal_max"] = windowMax
        nifti.header.set_xyzt_units("mm", "sec")

    # actual writing
    nib.save(nifti, os.path.join(outpath, filename))

    # write json
    if json_dict is not None:
        # fix series description
        if series_description is not None:
            json_dict["SeriesDescription"] = series_description
        jsoname = filename.split(".")[0] + ".json"
        with open(os.path.join(outpath, jsoname), "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
