"""This module contains DICOM reading routines."""
__all__ = ["read_dicom"]

import glob
from typing import Dict, List, Tuple, Union

import numpy as np

from deepmr.io.dicom import _subroutines


def read_dicom(dicomdir: Union[str, List, Tuple]) -> Tuple[np.ndarray, Dict]:
    """
    Load multi-contrast images for parameter mapping.

    Args:
        dicomdir: string or list of strings with DICOM folders path (supports wildcard).

    Returns:
        image: ndarray of sorted image data.
        head: MRD RawHeader corresponding to input DICOM.
    """
    # parse dicom
    if isinstance(dicomdir, str):
        dicomdir = sorted(glob.glob(dicomdir))

    # load dicom
    img, dsets = _subroutines._read_dcm(dicomdir)

    # get slice locations
    uSliceLocs, firstSliceIdx, sliceIdx = _subroutines._get_slice_locations(dsets)

    # get echo times
    inversionTimes = _subroutines._get_inversion_times(dsets)

    # get echo times
    echoTimes = _subroutines._get_echo_times(dsets)

    # get echo numbers
    echoNumbers = _subroutines._get_echo_numbers(dsets)

    # get repetition times
    repetitionTimes = _subroutines._get_repetition_times(dsets)

    # get flip angles
    flipAngles = _subroutines._get_flip_angles(dsets)

    # get sequence matrix
    contrasts = np.stack(
        (inversionTimes, echoTimes, echoNumbers, repetitionTimes, flipAngles), axis=1
    )

    # get unique contrast and indexes
    uContrasts, contrastIdx = _subroutines._get_unique_contrasts(contrasts)

    # get size
    n_slices = len(uSliceLocs)
    n_contrasts = uContrasts.shape[0]
    ninstances, ny, nx = img.shape

    # fill sorted image tensor
    sorted_image = np.zeros((n_contrasts, n_slices, ny, nx), dtype=img.dtype)
    for n in range(ninstances):
        sorted_image[contrastIdx[n], sliceIdx[n], :, :] = img[n]

    # get dicom template
    dcm_template = _subroutines._get_dicom_info(dsets, firstSliceIdx)

    # get header
    head = _subroutines._dcm2mrd(sorted_image, dcm_template)

    # unpack sequence
    TI, TE, EC, TR, FA = uContrasts.transpose()

    # fill sequence
    head["head"].sequenceParameters.TR = TR.tolist()
    head["head"].sequenceParameters.TE = TE.tolist()
    head["head"].sequenceParameters.TI = TI.tolist()
    head["head"].sequenceParameters.flipAngle_deg = FA.tolist()

    # squeeze
    if sorted_image.shape[0] == 1:
        sorted_image = sorted_image[0]

    return sorted_image, head
