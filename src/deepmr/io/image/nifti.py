"""NIfTI reading routines."""

__all__ = ["read_nifti"]

# import copy
import glob
import json
import os

import numpy as np
import nibabel as nib

from ..utils import nifti
from ..utils.header import Header


def read_nifti(filepath: str | list | tuple):
    """
    Read image from nifti files.

    Parameters
    ----------
    filepath : str | list | tuple
        Path to nifti folder.

    Returns
    -------
    image : np.ndarray
        Complex image data of shape (ncoils, ncontrasts, nslices, ny, nx).
    header : deepmr.Header
        Metadata for image reconstruction.

    """
    # parse nifti
    if isinstance(filepath, str):
        filepath = sorted(glob.glob(filepath))
    
    # load nifti
    image, header, affine, json_list = _read_nifti(filepath)
        
    # get constrats info
    inversionTimes = nifti._get_inversion_times(json_list)
    echoTimes = nifti._get_echo_times(json_list)
    echoNumbers = nifti._get_echo_numbers(json_list)        
    repetitionTimes = nifti._get_repetition_times(json_list)        
    flipAngles = nifti._get_flip_angles(json_list)
    
    # get sequence matrix
    contrasts = np.stack((inversionTimes, echoTimes, echoNumbers, repetitionTimes, flipAngles), axis=1)
    
    # get unique contrast and indexes
    uContrasts = nifti._get_unique_contrasts(contrasts)
    
    # unpack sequence
    TI, TE, EC, TR, FA = uContrasts.transpose()
    
    # initialize header
    header = Header.from_nifti(image, header, affine, json_list[0])
        
    # update header
    header.FA = FA
    header.TI = TI
    header.TE = TE
    header.TR = TR
    
    return image, header


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
    image, header, affine = _nifti_read(nifti_path, json_list)
    
    return image, header, affine, json_list


def _nifti_read(file_path, json_dict):
    """
    Wrapper to nibabel to handle multi-file datasets.
    """
    if isinstance(file_path, (list, tuple)):
        # convert to array
        file_path = np.array(file_path)

        # check for complex images
        data_phase = _get_phase(file_path)
        data_real = _get_real(file_path)
        data_imag = _get_imag(file_path)
        data, head, affine = _get_magn(file_path)
        
        # cast to complex image
        if data_phase.size != 0:
            scale = 2 * np.pi / 4095
            offset = -np.pi
            data = data * np.exp(1j * scale * data_phase + offset)
        if data_real.size != 0 and data_imag.size != 0:
            data = data_real + 1j * data_imag
            
    else:
        file_path = [os.path.normpath(os.path.abspath(file_path))]
        img = nib.load(file_path[0])
        data = img.get_fdata()
        affine = img.affine
        head = img.header
        
     # fix fftshift along z
    if np.iscomplexobj(data) and json_dict['Manufacturer'] == 'GE':
        phase = np.angle(data)
        phase[..., 1::2, :, :] = ((1e5 * (phase[..., 1::2, :, :] + 2 * np.pi)) % (2 * np.pi * 1e5)) / 1e5 - np.pi
        data = np.abs(data) * np.exp(1j * phase)

    return np.ascontiguousarray(data.transpose()), head, affine


def _json_read(file_path):
    """
    Wrapper to handle multi-file json.
    """
    if not isinstance(file_path, (tuple, list)):
        file_path = [file_path]
    
    json_list = []
    for json_path in file_path:
        with open(json_path) as json_file:
            json_list.append(json.loads(json_file.read()))

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

# %% complex data handling
def _get_real(file_path):
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
    else:
        data_real = np.asarray([])
        
    return data_real


def _get_imag(file_path):
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
    else:
        data_imag = np.asarray([])
        
    return data_imag

    
def _get_phase(file_path):
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
    else:
        data_phase = np.asarray([])
        
    return data_phase


def _get_magn(file_path):
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

    return data, head, affine


