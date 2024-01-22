"""NIfTI reading routines."""

__all__ = ["read_nifti"]

import glob
import json
import math
import os

import numpy as np
import nibabel as nib

from ..types import nifti
from ..types.header import Header

def read_nifti(filepath):
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

# def write_nifti(image, head):
#     """
#     Write parametric map to dicom.
    
#     Args:
#         image: ndarray of image data to be written.
#         info: dict with the following fields:
#             - template: the DICOM template.
#             - slice_locations: ndarray of Slice Locations [mm].
#             - TI: ndarray of Inversion Times [ms].
#             - TE: ndarray of Echo Times [ms].
#             - TR: ndarray of Repetition Times [ms].
#             - FA: ndarray of Flip Angles [deg].
#         filename: name of the output nifti file.
#         outpath: desired output path
#     """
#     # generate file name
#     if filename.endswith('.nii') is False and filename.endswith('.nii.gz') is False:
#         filename += '.nii'
    
#     # generate output path
#     outpath = os.path.abspath(outpath)
    
#     # create output folder
#     if not os.path.exists(outpath):
#         os.makedirs(outpath)
        
#     # cast image to numpy
        
#     # reformat image
#     image = np.flip(image.transpose(), axis=1)
    
#     # cast image
#     minval = np.iinfo(np.int16).min
#     maxval = np.iinfo(np.int16).max
#     image[image < minval] = minval
#     image[image > maxval] = maxval
#     image = image.astype(np.int16)
               
#     # # get voxel size
#     # dx, dy = np.array(info['dcm_template'][0].PixelSpacing).round(4)
#     # dz = round(float(info['dcm_template'][0].SliceThickness), 4)
    
#     # # get affine
#     # affine, _ = utils._get_nifti_affine(info['dcm_template'], image.shape[-3:])
                
#     try:
#         windowMin = 0.5 * np.percentile(image[image < 0], 95)
#     except:
#         windowMin = 0
#     try:
#         windowMax = 0.5 * np.percentile(image[image > 0], 95)
#     except:
#         windowMax = 0
        
#     # write nifti
#     nifti = nib.Nifti1Image(image, affine)
#     nifti.header['pixdim'][1:4] = np.array([dx, dy, dz])
#     nifti.header['sform_code'] = 0
#     nifti.header['qform_code'] = 2
#     nifti.header['cal_min'] = windowMin 
#     nifti.header['cal_max'] = windowMax 
#     nifti.header.set_xyzt_units('mm', 'sec')
    
#     # actual writing
#     nib.save(nifti, os.path.join(outpath, filename))
    
#     # write json
#     if json_dict is not None:
#         # fix series description
#         if series_description is not None:
#             json_dict['SeriesDescription'] = series_description
#         jsoname = filename.split('.')[0] + '.json'
#         with open(os.path.join(outpath, jsoname), 'w', encoding='utf-8') as f:
#             json.dump(json_dict, f, ensure_ascii=False, indent=4)
            
            
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
            scale = 2 * math.pi / 4095
            offset = -math.pi
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
        phase[..., 1::2, :, :] = ((1e5 * (phase[..., 1::2, :, :] + 2 * math.pi)) % (2 * math.pi * 1e5)) / 1e5 - math.pi
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


