"""NIfTI header utils."""

import warnings

import numpy as np

from ..._external.nii2dcm.dcm import DicomMRI

from .common import _reorient, _get_plane_normal


def _get_shape(img):
    """
    Return image shape.
    """
    return img.shape[-3:]


def _get_resolution(header, json):
    """
    Return image resolution.
    """
    if json:
        resolution = (
            float(json["SliceThickness"]),
            header["pixdim"][2],
            header["pixdim"][1],
        )
    else:
        resolution = (header["pixdim"][3], header["pixdim"][2], header["pixdim"][1])

    return resolution


def _get_spacing(header):
    """
    Return slice spacing.
    """
    return header["pixdim"][3]


def _get_image_orientation(resolution, affine):
    """
    Return image orientation matrix.
    """
    # orientation
    dy, dx = resolution[1], resolution[2]

    # direction cosines & position parameters
    dircosX = affine[:3, 0] / dx
    dircosY = affine[:3, 1] / dy
    orientation = (
        dircosX[0],
        dircosX[1],
        dircosX[2],
        dircosY[0],
        dircosY[1],
        dircosY[2],
    )

    return np.around(orientation, 4)


def _get_flip_angles(json_list):
    """
    Return array of flip angles for each for each volume.
    """
    flipAngles = []
    for json_dict in json_list:
        if "FlipAngle" in json_dict:
            if isinstance(json_dict["FlipAngle"], list):
                json_dict["FlipAngle"] = np.asarray(json_dict["FlipAngle"])
            flipAngles.append(json_dict["FlipAngle"])
        else:
            flipAngles.append(90.0)

    return np.asarray(flipAngles)


def _get_echo_times(json_list):
    """
    Return array of echo times for each for each volume.
    """
    echoTimes = []
    for json_dict in json_list:
        if "EchoTime" in json_dict:
            if isinstance(json_dict["EchoTime"], list):
                json_dict["EchoTime"] = np.asarray(json_dict["EchoTime"])
            echoTimes.append(1e3 * json_dict["EchoTime"])
        else:
            echoTimes.append(0.0)

    return np.asarray(echoTimes)


def _get_echo_numbers(json_list):
    """
    Return array of echo numbers for each for each volume.
    """
    echoNumbers = []
    for json_dict in json_list:
        if "EchoNumber" in json_dict:
            if isinstance(json_dict["EchoNumber"], list):
                json_dict["EchoNumber"] = [int(e) for e in json_dict["EchoNumber"]]
            echoNumbers.append(json_dict["EchoNumber"])
        else:
            echoNumbers.append(1)

    return np.asarray(echoNumbers)


def _get_repetition_times(json_list):
    """
    Return array of repetition times for each volume.
    """
    repetitionTimes = []
    for json_dict in json_list:
        if "RepetitionTime" in json_dict:
            if isinstance(json_dict["RepetitionTime"], list):
                json_dict["RepetitionTime"] = np.asarray(json_dict["RepetitionTime"])
            repetitionTimes.append(1e3 * json_dict["RepetitionTime"])
        else:
            repetitionTimes.append(1000.0)

    return np.asarray(repetitionTimes)


def _get_inversion_times(json_list):
    """
    Return array of inversion times for each volume.
    """
    inversionTimes = []
    for json_dict in json_list:
        if "InversionTime" in json_dict:
            if isinstance(json_dict["InversionTime"], list):
                json_dict["InversionTime"] = np.asarray(json_dict["InversionTime"])
            inversionTimes.append(1e3 * json_dict["InversionTime"])
        else:
            inversionTimes.append(np.Inf)

    return np.asarray(inversionTimes)


def _get_unique_contrasts(constrasts):
    """
    Return ndarray of unique contrasts and contrast index for each dataset in dsets.
    """
    # get unique repetition times
    uContrasts = np.unique(constrasts, axis=0)

    return uContrasts


def _make_nifti_affine(shape, position, orientation, resolution):
    """
    Return affine transform between voxel coordinates and mm coordinates.

    Args:
        shape: volume shape (nz, ny, nx).
        resolution: image resolution in mm (dz, dy, dz).
        position: position of each slice (3, nz).
        orientation: image orientation.

    Returns:
        affine matrix describing image position and orientation.

    Ref: https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting
    """
    # get image size
    nz, ny, nx = shape

    # get resoluzion
    dz, dy, dx = resolution

    # common parameters
    T = position
    T1 = T[:, 0].round(4)

    F = orientation
    dr, dc = np.asarray([dy, dx]).round(4)

    if nz == 1:  # single slice case
        n = _get_plane_normal(orientation)
        ds = float(dz)

        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append(-ds * n, 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    else:  # multi slice case
        N = nz
        TN = T[:, -1].round(4)
        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append((TN - T1) / (N - 1), 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    # sign of affine matrix
    A0[:2, :] *= -1

    # reorient
    A = _reorient(shape, A0, "LAS")

    return A.astype(np.float32)


def _initialize_series_tag(json):
    """
    Initialize common DICOM series tags.

    Adapted from https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2dicom.py

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # change the hook
        dicomDset = DicomMRI("nii2dcm_dicom_mri.dcm").ds

    # ----- Update DICOM header from NIfTI JSON -----
    if "PatientName" in json:
        dicomDset.PatientName = json["PatientName"]
    if "PatientWeight" in json:
        dicomDset.PatientWeight = json["PatientWeight"]
    if "PatientID" in json:
        dicomDset.PatientID = json["PatientID"]
    if "PatientBirthDate" in json:
        dicomDset.PatientBirthDate = json["PatientBirthDate"]
    if "PatientAge" in json:
        dicomDset.PatientAge = json["PatientAge"]
    if "PatientSex" in json:
        dicomDset.PatientSex = json["PatientSex"]

    if "StudyDate" in json:
        dicomDset.StudyDate = json["StudyDate"]
    if "StudyTime" in json:
        dicomDset.StudyTime = json["StudyTime"]
    if "AccessionNumber" in json:
        dicomDset.AccessionNumber = json["AccessionNumber"]
    if "ReferringPhysicianName" in json:
        dicomDset.ReferringPhysicianName = json["ReferringPhysicianName"]
    if "StudyDescription" in json:
        dicomDset.StudyDescription = json["StudyDescription"]
    if "StudyInstanceUID" in json:
        dicomDset.StudyInstanceUID = json["StudyInstanceUID"]

    if "SeriesDate" in json:
        dicomDset.SeriesDate = json["SeriesDate"]
    if "SeriesTime" in json:
        dicomDset.SeriesTime = json["SeriesTime"]
    if "PatientPosition" in json:
        dicomDset.PatientPosition = json["PatientPosition"]
    if "SequenceName" in json:
        dicomDset.SequenceName = json["SequenceName"]
    if "FrameOfReferenceUID" in json:
        dicomDset.FrameOfReferenceUID = json["FrameOfReferenceUID"]

    if "Manufacturer" in json:
        dicomDset.Manufacturer = json["Manufacturer"]
    if "ManufacturerModelName" in json:
        dicomDset.ManufacturerModelName = json["ManufacturerModelName"]
    if "MagneticFieldStrength" in json:
        dicomDset.MagneticFieldStrength = json["MagneticFieldStrength"]
    if "InstitutionName" in json:
        dicomDset.InstitutionName = json["InstitutionName"]
    if "StationName" in json:
        dicomDset.StationName = json["StationName"]

    return dicomDset


def _initialize_json_dict(dicomDset):
    """
    Initialize Json dictionary.

    """

    json = {}

    if "PatientName" in dicomDset:
        json["PatientName"] = str(dicomDset.PatientName)
    if "PatientWeight" in dicomDset:
        json["PatientWeight"] = str(dicomDset.PatientWeight)
    if "PatientID" in dicomDset:
        json["PatientID"] = str(dicomDset.PatientID)
    if "PatientBirthDate" in dicomDset:
        json["PatientBirthDate"] = str(dicomDset.PatientBirthDate)
    if "PatientAge" in dicomDset:
        json["PatientAge"] = str(dicomDset.PatientAge)
    if "PatientSex" in dicomDset:
        json["PatientSex"] = str(dicomDset.PatientSex)

    if "StudyDate" in dicomDset:
        json["StudyDate"] = str(dicomDset.StudyDate)
    if "StudyTime" in dicomDset:
        json["StudyTime"] = str(dicomDset.StudyTime)
    if "AccessionNumber" in dicomDset:
        json["AccessionNumber"] = str(dicomDset.AccessionNumber)
    if "ReferringPhysicianName" in dicomDset:
        json["ReferringPhysicianName"] = str(dicomDset.ReferringPhysicianName)
    if "StudyDescription" in dicomDset:
        json["StudyDescription"] = str(dicomDset.StudyDescription)
    if "StudyInstanceUID" in dicomDset:
        json["StudyInstanceUID"] = str(dicomDset.StudyInstanceUID)

    if "SeriesDate" in dicomDset:
        json["SeriesDate"] = str(dicomDset.SeriesDate)
    if "SeriesTime" in dicomDset:
        json["SeriesTime"] = str(dicomDset.SeriesTime)
    if "PatientPosition" in dicomDset:
        json["PatientPosition"] = str(dicomDset.PatientPosition)
    if "SequenceName" in dicomDset:
        json["SequenceName"] = str(dicomDset.SequenceName)
    if "FrameOfReferenceUID" in dicomDset:
        json["FrameOfReferenceUID"] = str(dicomDset.FrameOfReferenceUID)

    if "Manufacturer" in dicomDset:
        json["Manufacturer"] = str(dicomDset.Manufacturer)
    if "ManufacturerModelName" in dicomDset:
        json["ManufacturerModelName"] = str(dicomDset.ManufacturerModelName)
    if "MagneticFieldStrength" in dicomDset:
        json["MagneticFieldStrength"] = str(dicomDset.MagneticFieldStrength)
    if "InstitutionName" in dicomDset:
        json["InstitutionName"] = str(dicomDset.InstitutionName)
    if "StationName" in dicomDset:
        json["StationName"] = str(dicomDset.StationName)

    return json
