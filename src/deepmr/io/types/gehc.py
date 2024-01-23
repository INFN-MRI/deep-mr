"""GEHC header utils."""

import warnings

from ...external.nii2dcm.dcm import DicomMRI


def _initialize_series_tag(head):
    """
    Initialize common DICOM series tags.

    Adapted from https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2dicom.py

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # change the hook
        dicomDset = DicomMRI("nii2dcm_dicom_mri.dcm").ds

    # ----- Update DICOM header from NIfTI JSON -----
    if "PatientName" in head:
        dicomDset.PatientName = head["PatientName"]
    if "PatientWeight" in head:
        dicomDset.PatientWeight = head["PatientWeight"]
    if "PatientID" in head:
        dicomDset.PatientID = head["PatientID"]
    if "PatientBirthDate" in head:
        dicomDset.PatientBirthDate = head["PatientBirthDate"]
    if "PatientSex" in head:
        dicomDset.PatientSex = head["PatientSex"]

    if "StudyDate" in head:
        dicomDset.StudyDate = head["StudyDate"]
    if "StudyTime" in head:
        dicomDset.StudyTime = head["StudyTime"]
    if "AccessionNumber" in head:
        dicomDset.AccessionNumber = head["AccessionNumber"]
    if "ReferringPhysicianName" in head:
        dicomDset.ReferringPhysicianName = head["ReferringPhysicianName"]
    if "StudyDescription" in head:
        dicomDset.StudyDescription = head["StudyDescription"]
    if "StudyInstanceUID" in head:
        dicomDset.StudyInstanceUID = head["StudyInstanceUID"]

    if "SeriesDate" in head:
        dicomDset.SeriesDate = head["SeriesDate"]
    if "SeriesTime" in head:
        dicomDset.SeriesTime = head["SeriesTime"]
    if "PatientPosition" in head:
        dicomDset.PatientPosition = head["PatientPosition"]
    if "SequenceName" in head:
        dicomDset.SequenceName = head["SequenceName"]
    if "FrameOfReferenceUID" in head:
        dicomDset.FrameOfReferenceUID = head["FrameOfReferenceUID"]

    if "Manufacturer" in head:
        dicomDset.Manufacturer = head["Manufacturer"]
    if "ManufacturerModelName" in head:
        dicomDset.ManufacturerModelName = head["ManufacturerModelName"]
    if "MagneticFieldStrength" in head:
        dicomDset.MagneticFieldStrength = head["MagneticFieldStrength"]
    if "InstitutionName" in head:
        dicomDset.InstitutionName = head["InstitutionName"]
    if "StationName" in head:
        dicomDset.StationName = head["StationName"]

    return dicomDset
