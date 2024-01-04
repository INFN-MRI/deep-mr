"""NIfTI reading routines."""

__all__ = ["read_nifti"]

# import copy
import glob
import json
import os

import numpy as np
import nibabel as nib

from ..utils.geometry import Geometry

def read_nifti(filepath: str | list | tuple) -> (np.ndarray, dict):
    """
    Read multi-contrast images for parameter mapping.

    Parameters
    ----------
    filepath : str | list | tuple
        Path to nifti folder.

    Returns
    -------
    np.ndarray
        Sorted image data.
    dict
        Image metadata.

    """
    # parse dicom
    if isinstance(filepath, str):
        filepath = sorted(glob.glob(filepath))
                
    return _read_nifti(filepath)


# %% subroutines
def _read_nifti(file_path):
    """
    load single or list of NIFTI files and automatically gather real/imag or magnitude/phase to complex image.
    """
    # get list of nifti files
    nifti_path = _get_nifti_paths(file_path)

    # get list of json files
    json_path = _get_json_paths(nifti_path)
    
    # load list of json dicts
    json_list = _json_read(json_path)

    # load nifti
    img, head, affine = _nifti_read(nifti_path)
    
    # build header
    header = Geometry.from_nifti(img, head, affine, json_list)

    return img, header, {"head": head, "affine": affine, "json": json_list}


def _nifti_read(file_path):
    """
    Wrapper to nibabel to handle multi-file datasets.
    """
    if isinstance(file_path, (list, tuple)):
        # convert to array
        file_path = np.array(file_path)

        # check for complex images
        # phase
        idx = np.argwhere(np.array(["phase" in name for name in file_path])).squeeze()
        files_phase = file_path[idx]
        if isinstance(files_phase, str):
            files_phase = np.array([files_phase])
        else:
            files_phase = np.array(files_phase)
        if files_phase.size > 0:
            file_path.pop(idx)
            img_phase = [nib.load(file) for file in files_phase]
            data_phase = np.stack(
                [d.get_fdata() for d in img_phase], axis=-1
            ).squeeze()
            affine = img_phase[0].affine
            head = img_phase[0].header
        else:
            img_phase = np.array([])

        # real
        idx = np.argwhere(np.array(["real" in name for name in file_path])).squeeze()
        files_real = file_path[idx]
        if isinstance(files_real, str):
            files_real = np.array([files_real])
        else:
            files_real = np.array(files_real)
        if files_real.size > 0:
            file_path.pop(idx)
            img_real = [nib.load(file) for file in files_real]
            data_real = np.stack(
                [d.get_fdata() for d in img_real], axis=-1
            ).squeeze()
            affine = img_real[0].affine
            head = img_real[0].header
        else:
            files_real = np.array([])

        # imaginary
        idx = np.argwhere(np.array(["imag" in name for name in file_path])).squeeze()
        files_imag = file_path[idx]
        if isinstance(files_imag, str):
            files_imag = np.array([files_imag])
        else:
            files_imag = np.array(files_imag)
        if files_imag.size > 0:
            file_path.pop(idx)
            img_imag = [nib.load(file) for file in files_imag]
            data_imag = np.stack(
                [d.get_fdata() for d in img_imag], axis=-1
            ).squeeze()
            affine = img_imag[0].affine
            head = img_imag[0].header
        else:
            img_imag = np.array([])

        # magnitude
        files_mag = file_path
        if isinstance(files_mag, str):
            files_mag = np.array([files_mag])
        else:
            files_mag = np.array(files_mag)
        if files_mag.size > 0:
            img_mag = [nib.load(file) for file in files_mag]
            data = np.stack([d.get_fdata() for d in img_mag], axis=-1).squeeze()
            affine = img_mag[0].affine
            head = img_mag[0].header
        else:
            img_mag = np.array([])
                
        # cast to complex image
        if files_mag.shape[0] != 0 and files_phase.shape[0] != 0:
            scale = 2 * np.pi / 4095
            offset = -np.pi
            data = data * np.exp(1j * scale * data_phase + offset)
        if files_real.shape[0] != 0 and files_imag.shape[0] != 0:
            data = data_real + 1j * data_imag

    else:
        file_path = [os.path.normpath(os.path.abspath(file_path))]
        img = nib.load(file_path[0])
        data = img.get_fdata()
        affine = img.affine
        head = img.header

    return np.ascontiguousarray(data.transpose()), head, affine


def _json_read(file_path):
    """
    Wrapper to handle multi-file json.
    """
    if not isinstance(file_path, (tuple, list)):
        file_path = [file_path]

    for json_path in file_path:
        with open(json_path) as json_file:
            json_list = json.loads(json_file.read())

    return json_list


# %% paths
def _get_json_paths(input):
    """
    Get path to all sidecar JSONs.
    """
    if isinstance(input, (list, tuple)):
        json_path = [path.split(".nii")[0] + ".json" for path in input]
    else:
        json_path = input.split(".nii")[0] + ".json"
    return json_path


def _get_nifti_paths(input):
    """
    Get path to all NIFTIs in a directory or a list of directories.
    """
    # get all files in nifti dir
    if isinstance(input, (list, tuple)):
        file_path = []
        # get file path
        for file in input:
            tmp = _get_full_path(file)[0]
            if tmp.endswith(".nii") or tmp.endswith(".nii.gz"):
                file_path.append(tmp)
            else:
                tmp = glob.glob(os.path.join(tmp, "*nii*"))
                file_path += tmp
        file_path = sorted(file_path)
    else:
        file_path = _get_full_path(input)[0]

    return file_path


def _get_full_path(file_path):
    """
    Get full path.
    """
    return [os.path.normpath(os.path.abspath(file_path))]

