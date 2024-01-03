"""This module contains DICOM reading routines."""
__all__ = ["read_dicom"]

import glob
from typing import Dict, List, Tuple, Union

import numpy as np

from . import _subroutines


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
    return _subroutines._read_dcm(dicomdir)
    
    # # get field strength
    # B0 = float(dsets[0].MagneticFieldStrength)

    # # get slice locations
    # uSliceLocs, firstSliceIdx, sliceIdx = _subroutines._get_slice_locations(dsets)

    # # get echo times
    # inversionTimes = _subroutines._get_inversion_times(dsets)

    # # get echo times
    # echoTimes = _subroutines._get_echo_times(dsets)

    # # get echo numbers
    # echoNumbers = _subroutines._get_echo_numbers(dsets)

    # # get repetition times
    # repetitionTimes = _subroutines._get_repetition_times(dsets)

    # # get flip angles
    # flipAngles = _subroutines._get_flip_angles(dsets)

    # # get sequence matrix
    # contrasts = np.stack(
    #     (inversionTimes, echoTimes, echoNumbers, repetitionTimes, flipAngles), axis=1
    # )

    # # get unique contrast and indexes
    # uContrasts, contrastIdx = _subroutines._get_unique_contrasts(contrasts)

    # # get size
    # n_slices = len(uSliceLocs)
    # n_contrasts = uContrasts.shape[0]
    # ninstances, ny, nx = img.shape

    # # fill sorted image tensor
    # sorted_image = np.zeros((n_contrasts, n_slices, ny, nx), dtype=img.dtype)
    # for n in range(ninstances):
    #     sorted_image[contrastIdx[n], sliceIdx[n], :, :] = img[n]

    # # get dicom template
    # dcm_template = _subroutines._get_dicom_info(dsets, firstSliceIdx)
    
    # # unpack sequence
    # TI, TE, EC, TR, FA = uContrasts.transpose()
    
    # # squeeze
    # if sorted_image.shape[0] == 1:
    #     sorted_image = sorted_image[0]
        
    # return sorted_image, {'nifti_template': {}, 'dcm_template': dcm_template, 'B0': B0, 'EC': EC, 'TI': TI, 'TE': TE, 'TR': TR, 'FA': FA}
