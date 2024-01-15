"""ISMRMRD header utils."""


import warnings

import numpy as np

from ...external.nii2dcm.dcm import DicomMRI


def _find_in_user_params(userField, *keys):
    """
    Find field in MRDHeader UserParameters.
    """
    # find names
    names = [field.name for field in userField]

    # find positions
    idx = [names.index(k) for k in keys if k in names]
    values = [userField[i].value for i in idx]

    if len(keys) == len(values):
        return dict(zip(keys, values))
    else:
        return None  # One or more keys not found


def _get_slice_locations(acquisitions):
    """
    Return array of unique slice locations and slice location index for each acquisition in acquisitions.
    """
    # get orientation and position
    orientation = _get_image_orientation(acquisitions)
    position = _get_position(acquisitions)

    # get unique slice locations
    sliceLocs = _get_relative_slice_position(orientation, position).round(decimals=4)
    uSliceLocs, firstVolumeIdx = np.unique(sliceLocs, return_index=True)

    # get indexes
    sliceIdx = np.zeros(sliceLocs.shape, dtype=np.int16)

    for n in range(len(uSliceLocs)):
        sliceIdx[sliceLocs == uSliceLocs[n]] = n

    return uSliceLocs, firstVolumeIdx, sliceIdx


def _get_relative_slice_position(orientation, position):
    """
    Return array of slice coordinates along the normal to imaging plane.
    """
    z = _get_plane_normal(orientation)
    return z @ position


def _get_plane_normal(orientation):
    """
    Return array of normal to imaging plane, as the cross product
    between x and y plane versors.
    """
    x, y = orientation
    return np.cross(x, y)


def _get_first_volume(acquisitions, index):
    """
    Get first volume in a multi-contrast series.
    """
    out = [acquisitions[idx] for idx in index]

    return out


def _get_shape(geom):
    """
    Return image shape.
    """
    shape = geom.matrixSize
    nz, ny, nx = shape.z, shape.y, shape.x
    shape = (nz, ny, nx)
    return shape


def _get_spacing(user, geom, shape):
    """
    Return slice spacing.
    """
    nz = shape[0]
    tmp = _find_in_user_params(
        user.userParameterDouble, "SliceThickness", "SpacingBetweenSlices"
    )
    if tmp is not None:
        dz, spacing = tmp["SliceThickness"], tmp["SpacingBetweenSlices"]
    else:
        if nz != 1:
            warnings.warn(
                "Slice thickness and spacing info not found; assuming contiguous"
                " slices!",
                UserWarning,
            )
        rz = geom.fieldOfView_mm.z
        dz = rz / nz
        spacing = dz
    spacing = round(float(spacing), 2)
    return spacing, dz


def _get_resolution(geom, shape, dz):
    """
    Return image resolution.
    """
    _, ny, nx = shape
    ry, rx = geom.fieldOfView_mm.y, geom.fieldOfView_mm.x
    dy, dx = ry / ny, rx / nx
    resolution = (dz, dy, dx)
    return resolution


def _get_image_orientation(acquisitions):
    """
    Return image orientation matrix.
    """
    tmp = acquisitions[0].getHead()
    dircosX = np.asarray(tmp.read_dir)
    dircosY = np.asarray(tmp.phase_dir)
    orientation = (
        dircosX[0],
        dircosX[1],
        dircosX[2],
        dircosY[0],
        dircosY[1],
        dircosY[2],
    )

    orientation = np.asarray(orientation).reshape(2, 3)

    return np.around(orientation, 4)


def _get_position(acquisitions):
    """
    Return matrix of image position of size (3, nslices).
    """
    return np.stack(
        [
            np.asarray([acq.position[0], acq.position[1], acq.position[2]])
            for acq in acquisitions
        ],
        axis=1,
    )


def _get_origin(acquisitions):
    """
    Return image origin.
    """
    pos = [np.asarray(acq.getHead().position) for acq in acquisitions]
    pos = np.stack(pos, axis=0)
    origin = tuple(pos.mean(axis=0))
    return origin


def _get_flip_angles(header):
    """
    Return array of flip angles for each for each volume.
    """
    try:
        flipAngles = header.sequenceParameters.FA
    except:
        flipAngles = None

    return np.asarray(flipAngles)


def _get_echo_times(header):
    """
    Return array of echo times for each for each volume.
    """
    try:
        echoTimes = header.sequenceParameters.TE
    except:
        echoTimes = None

    return np.asarray(echoTimes)


def _get_repetition_times(header):
    """
    Return array of repetition times for each volume.
    """
    try:
        repetitionTimes = header.sequenceParameters.TR
    except:
        repetitionTimes = None

    return np.asarray(repetitionTimes)


def _get_inversion_times(header):
    """
    Return array of inversion times for each volume.
    """
    try:
        inversionTimes = header.sequenceParameters.TI
    except:
        inversionTimes = None

    return np.asarray(inversionTimes)


def _initialize_series_tag(mrdHead):
    """
    Initialize common DICOM series tags.

    Adapted from https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2dicom.py

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # change the hook
        dicomDset = DicomMRI("nii2dcm_dicom_mri.dcm").ds

    # ----- Update DICOM header from MRD header -----
    try:
        if mrdHead.subjectInformation is None:
            pass
        else:
            if mrdHead.subjectInformation.patientName is not None:
                dicomDset.PatientName = mrdHead.subjectInformation.patientName
            if mrdHead.subjectInformation.patientWeight_kg is not None:
                dicomDset.PatientWeight = mrdHead.subjectInformation.patientWeight_kg
            if mrdHead.subjectInformation.patientID is not None:
                dicomDset.PatientID = mrdHead.subjectInformation.patientID
            if mrdHead.subjectInformation.patientBirthdate is not None:
                dicomDset.PatientBirthDate = mrdHead.subjectInformation.patientBirthdate
            if mrdHead.subjectInformation.patientGender is not None:
                dicomDset.PatientSex = mrdHead.subjectInformation.patientGender

    except Exception:
        print(
            "Error setting header information from MRD header's subjectInformationType section"
        )

    try:
        if mrdHead.studyInformation is None:
            pass
        else:
            if mrdHead.studyInformation.studyDate is not None:
                dicomDset.StudyDate = mrdHead.studyInformation.studyDate
            if mrdHead.studyInformation.studyTime is not None:
                dicomDset.StudyTime = mrdHead.studyInformation.studyTime
            if mrdHead.studyInformation.accessionNumber is not None:
                dicomDset.AccessionNumber = mrdHead.studyInformation.accessionNumber
            if mrdHead.studyInformation.referringPhysicianName is not None:
                dicomDset.ReferringPhysicianName = (
                    mrdHead.studyInformation.referringPhysicianName
                )
            if mrdHead.studyInformation.studyInstanceUID is not None:
                dicomDset.StudyInstanceUID = mrdHead.studyInformation.studyInstanceUID

    except Exception:
        print(
            "Error setting header information from MRD header's studyInformationType section"
        )

    try:
        if mrdHead.measurementInformation is None:
            pass
        else:
            if mrdHead.measurementInformation.seriesDate is not None:
                dicomDset.SeriesDate = mrdHead.measurementInformation.seriesDate
            if mrdHead.measurementInformation.seriesTime is not None:
                dicomDset.SeriesTime = mrdHead.measurementInformation.seriesTime
            if mrdHead.measurementInformation.patientPosition is not None:
                dicomDset.PatientPosition = (
                    mrdHead.measurementInformation.patientPosition.name
                )
            if mrdHead.measurementInformation.relativeTablePosition is not None:
                dicomDset.IsocenterPosition = (
                    mrdHead.measurementInformation.relativeTablePosition
                )
            if mrdHead.measurementInformation.initialSeriesNumber is not None:
                dicomDset.SeriesNumber = (
                    mrdHead.measurementInformation.initialSeriesNumber
                )
            if mrdHead.measurementInformation.protocolName is not None:
                dicomDset.SeriesDescription = (
                    mrdHead.measurementInformation.protocolName
                )
            if mrdHead.measurementInformation.sequenceName is not None:
                dicomDset.SequenceName = mrdHead.measurementInformation.sequenceName
            if mrdHead.measurementInformation.frameOfReferenceUID is not None:
                dicomDset.FrameOfReferenceUID = (
                    mrdHead.measurementInformation.frameOfReferenceUID
                )

    except Exception:
        print(
            "Error setting header information from MRD header's measurementInformation section"
        )

    try:
        if mrdHead.acquisitionSystemInformation.systemVendor is not None:
            dicomDset.Manufacturer = mrdHead.acquisitionSystemInformation.systemVendor
        if mrdHead.acquisitionSystemInformation.systemModel is not None:
            dicomDset.ManufacturerModelName = (
                mrdHead.acquisitionSystemInformation.systemModel
            )
        if mrdHead.acquisitionSystemInformation.systemFieldStrength_T is not None:
            dicomDset.MagneticFieldStrength = (
                mrdHead.acquisitionSystemInformation.systemFieldStrength_T
            )
        if mrdHead.acquisitionSystemInformation.institutionName is not None:
            dicomDset.InstitutionName = (
                mrdHead.acquisitionSystemInformation.institutionName
            )
        if mrdHead.acquisitionSystemInformation.stationName is not None:
            dicomDset.StationName = mrdHead.acquisitionSystemInformation.stationName
    except Exception:
        print(
            "Error setting header information from MRD header's acquisitionSystemInformation section"
        )

    return dicomDset
