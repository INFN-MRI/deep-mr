"""DICOM reading routines."""

__all__ = ["read_dicom"]

# import copy
import glob
import multiprocessing
import os
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pydicom

from ..utils.geometry import Geometry

def read_dicom(filepath: str | list | tuple) -> (np.ndarray, dict):
    """
    Read multi-contrast images for parameter mapping.

    Parameters
    ----------
    filepath : str | list | tuple
        Path to dicom folder.

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

    # load dicom
    return _read_dcm(filepath)


# %% subroutines
def _read_dcm(dicomdir):
    """
    load list of dcm files and automatically gather real/imag or magnitude/phase to complex image.
    """
    # get list of dcm files
    dcm_paths = _get_dicom_paths(dicomdir)

    # check inside paths for subfolders
    dcm_paths = _probe_dicom_paths(dcm_paths)

    # make pool of workers
    pool = ThreadPool(multiprocessing.cpu_count())

    # each thread load a dicom
    dsets = pool.map(_dcmread, dcm_paths)

    # cloose pool and wait finish
    pool.close()
    pool.join()

    # filter None
    dsets = [dset for dset in dsets if dset is not None]

    # cast image to complex
    img, dsets = _cast_to_complex(dsets)
    
    # build header
    header = Geometry.from_dicom(dsets)

    return img, header, dsets


def _dcmread(dcm_path):
    """
    Wrapper to pydicom dcmread to automatically handle not dicom files
    """
    try:
        return pydicom.dcmread(dcm_path)
    except:
        return None


def _dcmwrite(input):
    """
    Wrapper to pydicom dcmread to automatically handle path / file tuple
    """
    filename, dataset = input
    pydicom.dcmwrite(filename, dataset)

    
# %% paths
def _get_dicom_paths(dicomdir):
    """
    Get path to all DICOMs in a directory or a list of directories.
    """
    # get all files in dicom dir
    if isinstance(dicomdir, (tuple, list)):
        dcm_paths = _get_full_path(dicomdir[0], sorted(os.listdir(dicomdir[0])))
        for d in range(1, len(dicomdir)):
            dcm_paths += _get_full_path(dicomdir[d], sorted(os.listdir(dicomdir[d])))
        dcm_paths = sorted(dcm_paths)
    else:
        dcm_paths = _get_full_path(dicomdir, sorted(os.listdir(dicomdir)))

    return dcm_paths


def _get_full_path(root, file_list):
    """
    Create list of full file paths from file name and root folder path.
    """
    return [
        os.path.normpath(os.path.abspath(os.path.join(root, file)))
        for file in file_list
    ]


def _probe_dicom_paths(dcm_paths_in):
    """
    For each element in list, check if it is a folder and read dicom paths inside it.
    """
    dcm_paths_out = []

    # loop over paths in input list
    for path in dcm_paths_in:
        if os.path.isdir(path):
            dcm_paths_out += _get_dicom_paths(path)
        else:
            dcm_paths_out.append(path)

    return dcm_paths_out


# %% complex data handling
def _cast_to_complex(dsets_in):
    """
    Attempt to retrive complex image, with the following priority:

        1) Real + 1j Imag
        2) Magnitude * exp(1j * Phase)

    If neither Real / Imag nor Phase are found, returns Magnitude only.
    """
    # get vendor
    vendor = _get_vendor(dsets_in[0])

    # actual conversion
    if vendor == "GE":
        return _cast_to_complex_ge(dsets_in)

    if vendor == "Philips":
        return _cast_to_complex_philips(dsets_in)

    if vendor == "Siemens":
        return _cast_to_complex_siemens(dsets_in)


def _get_vendor(dset):
    """
    Get vendor from DICOM header.
    """
    if dset.Manufacturer == "GE MEDICAL SYSTEMS":
        return "GE"

    if dset.Manufacturer == "Philips Medical Systems":
        return "Philips"

    if dset.Manufacturer == "SIEMENS":
        return "Siemens"


def _cast_to_complex_ge(dsets_in):
    """
    Attempt to retrive complex image for GE DICOM, with the following priority:

        1) Real + 1j Imag
        2) Magnitude * exp(1j * Phase)

    If neither Real / Imag nor Phase are found, returns Magnitude only.
    """
    # initialize
    real = []
    imag = []
    magnitude = []
    phase = []
    do_recon = True

    # allocate template out
    dsets_out = []

    # loop over dataset
    for dset in dsets_in:
        if dset[0x0043, 0x102F].value == 0:
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)

        if dset[0x0043, 0x102F].value == 1:
            phase.append(dset.pixel_array)

        if dset[0x0043, 0x102F].value == 2:
            real.append(dset.pixel_array)

        if dset[0x0043, 0x102F].value == 3:
            imag.append(dset.pixel_array)

    if real and imag and do_recon:
        img = np.stack(real, axis=0).astype(np.float32) + 1j * np.stack(
            imag, axis=0
        ).astype(np.float32)
        do_recon = False

    if magnitude and phase and do_recon:
        scale = 2 * np.pi / 4095
        offset = -np.pi
        img = np.stack(magnitude, axis=0).astype(np.float32) * np.exp(
            1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32)
        )
        do_recon = False
    elif do_recon:
        img = np.stack(magnitude, axis=0).astype(np.float32)

    # fix phase shift along z
    if np.iscomplexobj(img):
        phase = np.angle(img)
        phase[..., 1::2, :, :] = (
            (1e5 * (phase[..., 1::2, :, :] + 2 * np.pi)) % (2 * np.pi * 1e5)
        ) / 1e5 - np.pi
        img = np.abs(img) * np.exp(1j * phase)

    # count number of instances
    ninstances = img.shape[0]

    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0
        dsets_out[n][0x0025, 0x1007].value = ninstances
        dsets_out[n][0x0025, 0x1019].value = ninstances

    return img, dsets_out


def _cast_to_complex_philips(dsets_in):
    """
    Attempt to retrive complex image for Philips DICOM:
    If Phase is not found, returns Magnitude only.
    """
    # initialize
    magnitude = []
    phase = []

    # allocate template out
    dsets_out = []

    # loop over dataset
    for dset in dsets_in:
        if dset.ImageType[-2] == "M":
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)

        if dset.ImageType[-2] == "P":
            phase.append(dset.pixel_array)

    if magnitude and phase:
        scale = 2 * np.pi / 4095
        offset = -np.pi
        img = np.stack(magnitude, axis=0).astype(np.float32) * np.exp(
            1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32)
        )
    else:
        img = np.stack(magnitude, axis=0).astype(np.float32)

    # count number of instances
    ninstances = img.shape[0]

    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0

    return img, dsets_out


def _cast_to_complex_siemens(dsets_in):
    """
    Attempt to retrive complex image for Siemens DICOM:
    If Phase is not found, returns Magnitude only.
    """
    # initialize
    magnitude = []
    phase = []

    # allocate template out
    dsets_out = []

    # loop over dataset
    for dset in dsets_in:
        if dset.ImageType[2] == "M":
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)

        if dset.ImageType[2] == "P":
            phase.append(dset.pixel_array)

    if magnitude and phase:
        scale = 2 * np.pi / 4095
        offset = -np.pi
        img = np.stack(magnitude, axis=0).astype(np.float32) * np.exp(
            1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32)
        )
    else:
        img = np.stack(magnitude, axis=0).astype(np.float32)

    # count number of instances
    ninstances = img.shape[0]

    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0

    return img, dsets_out
    