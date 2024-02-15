"""NIfTI reading routines."""

__all__ = ["read_nifti", "write_nifti"]

import copy
import glob
import json
import math
import os
import time

import numpy as np
import nibabel as nib

import torch

from ..._types import nifti
from ..._types.header import Header

from .common import _prepare_image, _anonymize


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
    (
        inversionTimes,
        echoTimes,
        echoNumbers,
        repetitionTimes,
        flipAngles,
    ) = np.broadcast_arrays(
        inversionTimes, echoTimes, echoNumbers, repetitionTimes, flipAngles
    )
    contrasts = np.stack(
        (
            inversionTimes.squeeze(),
            echoTimes.squeeze(),
            echoNumbers.squeeze(),
            repetitionTimes.squeeze(),
            flipAngles.squeeze(),
        ),
        axis=1,
    )

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


def write_nifti(
    filename,
    image,
    filepath="./",
    head=None,
    series_description="",
    series_number_offset=0,
    series_number_scale=1000,
    rescale=False,
    anonymize=False,
    verbose=False,
):
    """
    Write image to NIfTI.

    Parameters
    ----------
    filename : str
        Name of the file.
    image : np.ndarray
        Complex image data of shape (ncontrasts, nslices, ny, nx).
    filepath : str, optional
        Path to file. The default is "./".
    head : deepmr.Header, optional
        Structure containing trajectory of shape (ncontrasts, nviews, npts, ndim)
        and meta information (shape, resolution, spacing, etc). If None,
        assume 1mm isotropic resolution, contiguous slices and axial orientation.
        The default is None
    series_description : str, optional
        Custom series description. The default is "".
    series_number_offset : int, optional
        Series number offset with respect to the acquired one.
        Final series number is series_number_scale * acquired_series_number + series_number_offset.
        he default is 0.
    series_number_scale : int, optional
        Series number multiplicative scaling with respect to the acquired one.
        Final series number is series_number_scale * acquired_series_number + series_number_offset.
        The default is 1000.
    rescale : bool, optional
        If true, rescale image intensity between 0 and int16_max.
        Beware! Avoid this if you are working with quantitative maps.
        The default is False.
    anonymize : bool, optional
        If True, remove sensible info from header. The default is "False".
    verbose : bool, optional
        Verbosity flag. The default is "False".

    """
    # convert image to nupy
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # cast header to numpy
    if head is not None:
        head = copy.deepcopy(head)
        head.numpy()

    # anonymize
    if head is not None and anonymize:
        head = _anonymize(head)

    # expand images if needed
    if len(image.shape) == 3:
        raise UserWarning("Number of dimensions = 3; assuming single contrast.")
        image = image[None, ...]
    if len(image.shape) == 2:
        raise UserWarning(
            "Number of dimensions = 2; assuming single contrast and slice."
        )
        image = image[None, None, ...]

    # get number of instances
    ncontrasts = image.shape[0]

    # generate file name
    if filename.endswith(".nii") is False and filename.endswith(".nii.gz") is False:
        filename += ".nii"

    # generate output path
    filepath = os.path.realpath(filepath)

    # create output folder
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # get permutations and flips
    if head is not None:
        transpose = head.transpose
        flip = head.flip
    else:
        transpose = None
        flip = None

    # cast image to numpy
    image, windowRange = _prepare_image(image, transpose, flip, rescale)

    # initialize header if not provided
    if head is None:
        head = Header(image.shape[-3:])

    # unpack header
    affine = head.affine

    # resolution
    dz = float(head.ref_dicom.SpacingBetweenSlices)
    dx, dy = head.ref_dicom.PixelSpacing
    resolution = np.asarray((dz, dy, dx))

    if head.TR is not None:
        TR = float(head.TR.min())
    else:
        TR = 1000.0

    # prepare json dictionary
    head.ref_dicom.SeriesDescription = series_description
    head.ref_dicom.SeriesNumber = (
        series_number_scale * head.ref_dicom.SeriesNumber + series_number_offset
    )
    json_dict = nifti._initialize_json_dict(head.ref_dicom)

    # add stuff
    json_dict["SliceThickness"] = str(head.ref_dicom.SliceThickness)
    json_dict["EchoNumber"] = [str(n) for n in range(ncontrasts)]
    if head.FA is not None:
        if head.FA.size == 1:
            json_dict["FlipAngle"] = float(abs(head.FA))
        elif len(np.unique(head.FA)) == 1:
            json_dict["FlipAngle"] = float(abs(head.FA[0]))
        else:
            json_dict["FlipAngle"] = list(abs(head.FA).astype(float))
    if head.TE is not None and not (np.isinf(np.sum(head.TE))):
        if head.TE.size == 1:
            json_dict["EchoTime"] = float(head.TE) * 1e-3
        elif len(np.unique(head.TE)) == 1:
            json_dict["EchoTime"] = float(head.TE[0]) * 1e-3
        else:
            json_dict["EchoTime"] = list(head.TE.astype(float) * 1e-3)
    if head.TR is not None and not (np.isinf(np.sum(head.TR))):
        if head.TR.size == 1:
            json_dict["RepetitionTime"] = float(head.TR) * 1e-3
        elif len(np.unique(head.TR)) == 1:
            json_dict["RepetitionTime"] = float(head.TR[0]) * 1e-3
        else:
            json_dict["RepetitionTime"] = list(head.TR.astype(float) * 1e-3)
    if head.TI is not None and not (np.isinf(np.sum(head.TI))):
        if head.TI.size == 1:
            json_dict["InversionTime"] = float(head.TI) * 1e-3
        elif len(np.unique(head.TI)) == 1:
            json_dict["InversionTime"] = float(head.TI[0]) * 1e-3
        else:
            json_dict["InversionTime"] = list(head.TI.astype(float) * 1e-3)

    # # actual writing
    if verbose:
        t0 = time.time()
        print(
            f"Writing output NIfTI image to {os.path.realpath(os.path.join(filepath, filename))}...",
            end="\t",
        )
    _nifti_write(filename, filepath, image, affine, resolution, TR, windowRange)

    # write json
    _json_write(filename, filepath, json_dict)
    if verbose:
        t1 = time.time()
        print(f"done! Elapsed time: {round(t1-t0, 2)} s.")


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
        data_phase, file_path = _get_phase(file_path)
        data_real, file_path = _get_real(file_path)
        data_imag, file_path = _get_imag(file_path)
        data, head, affine = _get_magn(file_path)

        # cast to complex image
        if data_phase.size != 0:
            min_phase = data_phase.min()
            max_phase = data_phase.max()
            data_phase = (data_phase - min_phase) / (
                max_phase - min_phase
            ) * 2 * math.pi - math.pi
            data = data * np.exp(1j * data_phase)
        if data_real.size != 0 and data_imag.size != 0:
            data = data_real + 1j * data_imag

    else:
        file_path = [os.path.normpath(os.path.abspath(file_path))]
        img = nib.load(file_path[0])
        data = img.get_fdata()
        affine = img.affine
        head = img.header

    # fix fftshift along z
    if np.iscomplexobj(data) and "GE" in json_dict[0]["Manufacturer"].upper():
        phase = np.angle(data)
        phase[:, :, 1::2, ...] = (
            (1e5 * (phase[:, :, 1::2, ...] + 2 * math.pi)) % (2 * math.pi * 1e5)
        ) / 1e5 - math.pi
        data = np.abs(data) * np.exp(1j * phase)

    return np.flip(data.transpose(), axis=(-2, -1)), head, affine


def _nifti_write(filename, filepath, image, affine, resolution, TR, windowRange):
    """Actual nifti writing routine."""

    # reformat image
    image = np.flip(image.transpose(), axis=(-2, -1))

    # get voxel size
    dz, dy, dx = np.round(resolution, 2)

    # write nifti
    out = nib.Nifti1Image(image, affine)
    out.header["pixdim"][1:5] = np.asarray([dx, dy, dz, TR])
    out.header["sform_code"] = 0
    out.header["qform_code"] = 2
    out.header["cal_min"] = windowRange[0]
    out.header["cal_max"] = windowRange[1]
    out.header.set_xyzt_units("mm", "sec")

    # actual writing
    outpath = os.path.realpath(os.path.join(filepath, filename))
    nib.save(out, outpath)


def _json_read(filepath):
    """
    Wrapper to handle multi-file json.
    """
    if not isinstance(filepath, (tuple, list)):
        filepath = [filepath]

    json_list = []
    for json_path in filepath:
        with open(json_path) as json_file:
            json_list.append(json.loads(json_file.read()))

    return json_list


def _json_write(filename, filepath, json_dict):
    if json_dict is not None:
        jsoname = filename.split(".")[0] + ".json"
        outpath = os.path.realpath(os.path.join(filepath, jsoname))
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)


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
        file_path = np.delete(file_path, idx)
        img_real = [nib.load(file) for file in files_real]
        data_real = np.stack([d.get_fdata() for d in img_real], axis=-1).squeeze()
    else:
        data_real = np.asarray([])

    return data_real, file_path


def _get_imag(file_path):
    idx = np.argwhere(np.array(["imag" in name for name in file_path])).squeeze()
    files_imag = file_path[idx]
    if isinstance(files_imag, str):
        files_imag = np.array([files_imag])
    else:
        files_imag = np.array(files_imag)
    if files_imag.size > 0:
        file_path = np.delete(file_path, idx)
        img_imag = [nib.load(file) for file in files_imag]
        data_imag = np.stack([d.get_fdata() for d in img_imag], axis=-1).squeeze()
    else:
        data_imag = np.asarray([])

    return data_imag, file_path


def _get_phase(file_path):
    idx = np.argwhere(np.array(["ph" in name for name in file_path])).squeeze()
    files_phase = file_path[idx]
    if isinstance(files_phase, str):
        files_phase = np.array([files_phase])
    else:
        files_phase = np.array(files_phase)
    if files_phase.size > 0:
        file_path = np.delete(file_path, idx)
        img_phase = [nib.load(file) for file in files_phase]
        data_phase = np.stack([d.get_fdata() for d in img_phase], axis=-1).squeeze()
        # data_phase = np.stack([(d.get_fdata() - d.dataobj.inter) / d.dataobj.slope for d in img_phase], axis=-1).squeeze()
    else:
        data_phase = np.asarray([])

    return data_phase, file_path


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
