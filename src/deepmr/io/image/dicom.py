"""DICOM reading routines."""

__all__ = ["read_dicom"]

import copy
import glob
import math
import multiprocessing
import os
import time

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import torch

import pydicom

from ..types import dicom
from ..types.header import Header

from .common import _prepare_image, _anonymize


def read_dicom(filepath):
    """
    Read image from dicom files.

    Parameters
    ----------
    filepath : str | list | tuple
        Path to dicom folder.

    Returns
    -------
    image : np.ndarray
        Complex image data of shape (ncoils, ncontrasts, nslices, ny, nx).
    header : deepmr.Header
        Metadata for image reconstruction.

    """
    # parse dicom
    if isinstance(filepath, str):
        filepath = sorted(glob.glob(filepath))

    # load dicom
    image, dsets, vendor = _read_dcm(filepath)

    # get slice locations
    uSliceLocs, firstVolumeIdx, sliceIdx = dicom._get_slice_locations(dsets)

    # get constrats info
    inversionTimes = dicom._get_inversion_times(dsets)
    echoTimes = dicom._get_echo_times(dsets)
    echoNumbers = dicom._get_echo_numbers(dsets)
    repetitionTimes = dicom._get_repetition_times(dsets)
    flipAngles = dicom._get_flip_angles(dsets)

    # get sequence matrix
    contrasts = np.stack(
        (inversionTimes, echoTimes, echoNumbers, repetitionTimes, flipAngles), axis=1
    )

    # get unique contrast and indexes
    uContrasts, contrastIdx = dicom._get_unique_contrasts(contrasts)

    # get size
    n_slices = len(uSliceLocs)
    n_contrasts = uContrasts.shape[0]
    ninstances, ny, nx = image.shape

    # fill sorted image tensor
    sorted_image = np.zeros((n_contrasts, n_slices, ny, nx), dtype=image.dtype)
    for n in range(ninstances):
        sorted_image[contrastIdx[n], sliceIdx[n], :, :] = image[n]

    # fix phase shift along z
    if "GE" in vendor.upper() and np.iscomplexobj(sorted_image):
        phase = np.angle(sorted_image)
        phase[..., 1::2, :, :] = (
            (1e5 * (phase[..., 1::2, :, :] + 2 * math.pi)) % (2 * math.pi * 1e5)
        ) / 1e5 - math.pi
        sorted_image = np.abs(sorted_image) * np.exp(1j * phase)

    # unpack sequence
    TI, TE, EC, TR, FA = uContrasts.transpose()

    # squeeze
    if sorted_image.shape[0] == 1:
        sorted_image = sorted_image[0]

    # initialize header
    header = Header.from_dicom(dsets, firstVolumeIdx)

    # update header
    header.FA = FA
    header.TI = TI
    header.TE = TE
    header.TR = TR

    return sorted_image, header


def write_dicom(
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
    Write image to DICOM.

    Parameters
    ----------
    filename : str
        Name of the folder containing all the DICOM files.
    image : np.ndarray
        Complex image data of shape (ncoils, ncontrasts, nslices, ny, nx).
    filepath : str, optional
        Path to filaname. The default is "./".
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
    ncontrasts, nslices = image.shape[:2]
    ninstances = ncontrasts * nslices

    # generate output path
    filepath = os.path.realpath(os.path.join(filepath, filename))

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
    windowWidth = windowRange[1] - windowRange[0]

    # initialize header if not provided
    if head is None:
        head = Header(image.shape[-3:])

    # unpack header
    affine = head.affine
    shape = head.shape

    # resolution
    dz = float(head.ref_dicom.SliceThickness)
    dx, dy = head.ref_dicom.PixelSpacing
    resolution = np.asarray((dz, dy, dx))

    # prepare header
    head.ref_dicom.SeriesDescription = series_description
    head.ref_dicom.SeriesNumber = (
        series_number_scale * head.ref_dicom.SeriesNumber + series_number_offset
    )
    head.ref_dicom.WindowWidth = str(windowWidth)
    head.ref_dicom.WindowCenter = str(0.5 * windowWidth)
    head.ref_dicom.ImagesInAcquisition = ninstances

    try:
        head.ref_dicom[0x0025, 0x1007].value = ninstances
    except Exception:
        pass
    try:
        head.ref_dicom[0x0025, 0x1019].value = ninstances
    except Exception:
        pass

    # remove constant parameters from header
    if head.FA is not None:
        if head.FA.size == 1:
            head.ref_dicom.FlipAngle = float(abs(head.FA))
            head.FA = None
        elif len(np.unique(head.FA)) == 1:
            head.ref_dicom.FlipAngle = float(abs(head.FA[0]))
            head.FA = None
        else:
            head.FA = list(abs(head.FA).astype(float))
    if head.TE is not None and not (np.isinf(np.sum(head.TE))):
        if head.TE.size == 1:
            head.ref_dicom.EchoTime = float(head.TE)
            head.TE = None
        elif len(np.unique(head.TE)) == 1:
            head.ref_dicom.EchoTime = float(head.TE[0])
            head.TE = None
        else:
            head.TE = list(head.TE.astype(float))
    else:
        head.TE = None
    if head.TR is not None and not (np.isinf(np.sum(head.TR))):
        if head.TR.size == 1:
            head.ref_dicom.RepetitionTime = float(head.TR)
            head.TR = None
        elif len(np.unique(head.TR)) == 1:
            head.ref_dicom.RepetitionTime = float(head.TR[0])
            head.TR = None
        else:
            head.TR = list(head.TR.astype(float))
    else:
        head.TR = None
    if head.TI is not None and not (np.isinf(np.sum(head.TI))):
        if head.TI.size == 1:
            head.ref_dicom.InversionTime = float(head.TI)
            head.TI = None
        elif len(np.unique(head.TI)) == 1:
            head.ref_dicom.InversionTime = float(head.TI[0])
            head.TI = None
        else:
            head.TI = list(head.TI.astype(float))
    else:
        head.TI = None

    # generate position and slice location
    pos, zloc = dicom._make_geometry_tags(affine, shape, resolution)

    # generate dicom series
    dsetnames, dsets = _generate_dcm_series(image, head, windowWidth, pos, zloc)

    # actual writing
    if verbose:
        t0 = time.time()
        print(
            f"Writing output DICOM image (n={ninstances} images) to {filepath}...",
            end="\t",
        )
    _write_dcm(filepath, dsetnames, dsets)
    if verbose:
        t1 = time.time()
        print(f"done! Elapsed time: {round(t1-t0, 2)} s.")


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
    image, dsets, vendor = _cast_to_complex(dsets)

    return image, dsets, vendor


def _write_dcm(filepath, dsetnames, dsets):
    # get ninstances
    ninstances = len(dsets)

    # get dicompath
    dcm_paths = [os.path.join(filepath, file) for file in dsetnames]

    # generate path / data pair
    path_data = [[dcm_paths[n], dsets[n]] for n in range(ninstances)]

    # make pool of workers
    pool = ThreadPool(multiprocessing.cpu_count())

    # each thread write a dicom
    dsets = pool.map(_dcmwrite, path_data)

    # cloose pool and wait finish
    pool.close()
    pool.join()


def _dcmread(dcm_path):
    """
    Wrapper to pydicom dcmread to automatically handle not dicom files.
    """
    try:
        return pydicom.dcmread(dcm_path)
    except:
        return None


def _dcmwrite(input):
    """
    Wrapper to pydicom dcmread to automatically handle path / file tuple.
    """
    filename, dataset = input
    pydicom.dcmwrite(filename, dataset)


def _generate_dcm_series(input, head, windowWidth, pos, loc):
    """
    Generate dcm from template.
    """
    # get image size
    ncontrasts, nslices = input.shape[:2]
    ninstances = ncontrasts * nslices

    # initialize dsets
    dsets = []
    n = 0
    for z in range(nslices):
        for c in range(ncontrasts):
            dsets.append(copy.deepcopy(head.ref_dicom))

            # image data
            dsets[n].PixelData = input[c, z].tobytes()
            # dsets[n].pixel_array[:] = input[c, z]

            # instance properties
            dsets[n].SOPInstanceUID = pydicom.uid.generate_uid()
            dsets[n].InstanceNumber = str(n + 1)

            # geometrical properties
            dsets[n].ImagePositionPatient = list(pos[z])
            dsets[n].SliceLocation = str(loc[z])

            # contrast properties
            if head.FA is not None:
                dsets[n].FlipAngle = str(abs(head.FA[c]))
            if head.TR is not None:
                dsets[n].RepetitionTime = str(head.TR[c])
            if head.TE is not None:
                dsets[n].EchoTime = str(head.TE[c])
            if head.TI is not None:
                dsets[n].InversionTime = str(head.TI[c])

            # echo number
            dsets[n][0x0018, 0x0086].value = str(c)  # Echo Number
            dsets[n].EchoNumbers = str(c)  # Echo Number

            # update n
            n += 1

    # generate file names
    filename = ["img-" + str(n).zfill(3) + ".dcm" for n in range(ninstances)]

    return filename, dsets


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
    if "GE" in vendor.upper():
        img, dsets_out = _cast_to_complex_ge(dsets_in)
        return img, dsets_out, vendor

    if "PHILIPS" in vendor.upper():
        img, dsets_out = _cast_to_complex_philips(dsets_in)
        return img, dsets_out, vendor

    if "SIEMENS" in vendor.upper():
        img, dsets_out = _cast_to_complex_siemens(dsets_in)
        return img, dsets_out, vendor

    if "DEEPMR" in vendor.upper():
        img, dsets_out = _cast_to_complex_deepmr(dsets_in)
        return img, dsets_out, vendor


def _get_vendor(dset):
    """
    Get vendor from DICOM header.
    """
    if "GE MEDICAL SYSTEMS" in dset.Manufacturer.upper():
        return "GEHC"

    if "PHILIPS" in dset.Manufacturer.upper():
        return "Philips"

    if "SIEMENS" in dset.Manufacturer.upper():
        return "Siemens"

    return "DeepMR"


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
        scale = 2 * math.pi / 4095
        offset = -math.pi
        img = np.stack(magnitude, axis=0).astype(np.float32) * np.exp(
            1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32)
        )
        do_recon = False
    elif do_recon:
        img = np.stack(magnitude, axis=0).astype(np.float32)

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
        phase = np.stack(phase, axis=0).astype(np.float32)
        min_phase = phase.min()
        max_phase = phase.max()
        phase = (phase - min_phase) / (max_phase - min_phase) * 2 * math.pi - math.pi
        img = np.stack(magnitude, axis=0).astype(np.float32) * np.exp(1j * phase)
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
        phase = np.stack(phase, axis=0).astype(np.float32)
        min_phase = phase.min()
        max_phase = phase.max()
        phase = (phase - min_phase) / (max_phase - min_phase) * 2 * math.pi - math.pi
        img = np.stack(magnitude, axis=0).astype(np.float32) * np.exp(1j * phase)
    else:
        img = np.stack(magnitude, axis=0).astype(np.float32)

    # count number of instances
    ninstances = img.shape[0]

    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0

    return img, dsets_out


def _cast_to_complex_deepmr(dsets_in):
    """
    Attempt to get magnitude from DeepMR and cast to complex.
    """
    # initialize
    magnitude = []

    # allocate template out
    dsets_out = []

    # loop over dataset
    for dset in dsets_in:
        magnitude.append(dset.pixel_array)
        dsets_out.append(dset)

    img = np.stack(magnitude, axis=0).astype(np.complex64)

    # count number of instances
    ninstances = img.shape[0]

    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0

    return img, dsets_out
