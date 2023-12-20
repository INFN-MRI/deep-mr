"""Subroutines for DICOM files reading and sorting."""
import copy
import glob
import json
import os

import nibabel as nib
import numpy as np

from deepmr.io.mrd import header
from deepmr.io.mrd.constants import *
from deepmr.io.utils.geometry import *


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
    img, head, affine = _nifti_read(nifti_path)

    return img, {"head": head, "affine": affine, "json": json_list}


def _nifti_read(file_path):
    """
    Wrapper to nibabel to handle multi-file datasets.
    """
    if isinstance(file_path, (list, tuple)):
        # convert to array
        file_path = np.array(file_path)

        # check for complex images
        # phase
        idx = np.argwhere(np.array(["phase" in name for name in file_path])).squeeze()
        files_phase = file_path[idx]
        if isinstance(files_phase, str):
            files_phase = np.array([files_phase])
            if files_phase.size > 0:
                img_phase = [nib.load(file) for file in files_phase]
                data_phase = np.stack(
                    [d.get_fdata() for d in img_phase], axis=-1
                ).squeeze()
                affine = img_phase[0].affine
                head = img_phase[0].header
            else:
                img_phase = np.array([])

        # real
        idx = np.argwhere(np.array(["real" in name for name in file_path])).squeeze()
        files_real = file_path[idx]
        if isinstance(files_real, str):
            files_real = np.array([files_real])
            if files_real.size > 0:
                img_real = [nib.load(file) for file in files_real]
                data_real = np.stack(
                    [d.get_fdata() for d in img_real], axis=-1
                ).squeeze()
                affine = img_real[0].affine
                head = img_real[0].header
            else:
                files_real = np.array([])

        # imaginary
        idx = np.argwhere(np.array(["imag" in name for name in file_path])).squeeze()
        files_imag = file_path[idx]
        if isinstance(files_imag, str):
            files_imag = np.array([files_imag])
            if files_imag.size > 0:
                img_imag = [nib.load(file) for file in files_imag]
                data_imag = np.stack(
                    [d.get_fdata() for d in img_imag], axis=-1
                ).squeeze()
                affine = img_imag[0].affine
                head = img_imag[0].header
            else:
                img_imag = np.array([])

        # magnitude
        idx = np.argwhere(np.array(["mag" in name for name in file_path])).squeeze()
        files_mag = file_path[idx]
        if isinstance(files_mag, str):
            files_mag = np.array([files_mag])

            # remove imag
            tmp = np.concatenate((files_phase, files_real, files_imag)).tolist()
            s = set(tmp)
            files_mag = np.array([file for file in file_path if file not in s])

            if files_mag.size > 0:
                img_mag = [nib.load(file) for file in files_mag]
                data = np.stack([d.get_fdata() for d in img_mag], axis=-1).squeeze()
                affine = img_mag[0].affine
                head = img_mag[0].header
            else:
                img_mag = np.array([])

        # cast to complex image
        if files_mag.shape[0] != 0 and files_phase.shape[0] != 0:
            scale = 2 * np.pi / 4095
            offset = -np.pi
            data = data * np.exp(1j * scale * data_phase + offset)

        if files_real.shape[0] != 0 and files_imag.shape[0] != 0:
            data = data_real + 1j * data_imag

    else:
        file_path = [os.path.normpath(os.path.abspath(file_path))]
        img = nib.load(file_path[0])
        data = img.get_fdata()
        affine = img.affine
        head = img.header

    return img, head, affine


def _json_read(file_path):
    """
    Wrapper to handle multi-file json.
    """
    if not isinstance(file_path, (tuple, list)):
        file_path = [file_path]

    for json_path in file_path:
        with open(json_path) as json_file:
            json_list = json.loads(json_file.read())

    return json_list


# %% paths
def _get_json_paths(input):
    """
    Get path to all sidecar JSONs.
    """
    if isinstance(input, (list, tuple)):
        json_path = [path for path in input]
        for path in json_path:
            path = path.split(".nii")[0] + ".json"
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


# %% sequence parameters
def _get_flip_angles(dsets):
    """
    Return array of flip angles for each dataset in dsets.
    """
    # get flip angles
    flipAngles = np.array([float(dset.FlipAngle) for dset in dsets])

    return flipAngles


def _get_echo_times(dsets):
    """
    Return array of echo times for each dataset in dsets.
    """
    # get unique echo times
    echoTimes = np.array([float(dset.EchoTime) for dset in dsets])

    return echoTimes


def _get_echo_numbers(dsets):
    """
    Return array of echo numbers for each dataset in dsets.
    """
    # get unique echo times
    echoNumbers = np.array([int(dset.EchoNumbers) for dset in dsets])

    return echoNumbers


def _get_repetition_times(dsets):
    """
    Return array of repetition times for each dataset in dsets.
    """
    # get unique repetition times
    repetitionTimes = np.array([float(dset.RepetitionTime) for dset in dsets])

    return repetitionTimes


def _get_inversion_times(dsets):
    """
    Return array of inversion times for each dataset in dsets.
    """
    try:
        # get unique repetition times
        inversionTimes = np.array([float(dset.InversionTime) for dset in dsets])
    except:
        inversionTimes = np.zeros(len(dsets)) + np.inf

    return inversionTimes


def _get_unique_contrasts(constrasts):
    """
    Return ndarray of unique contrasts and contrast index for each dataset in dsets.
    """
    # get unique repetition times
    uContrasts = np.unique(constrasts, axis=0)

    # get indexes
    contrastIdx = np.zeros(constrasts.shape[0], dtype=np.int16)

    for n in range(uContrasts.shape[0]):
        contrastIdx[(constrasts == uContrasts[n]).all(axis=-1)] = n

    return uContrasts, contrastIdx


# %% header parsing
# def _mrd2dcm(data, head):
#     """Create DICOM from MRD header and image array."""
#     ####################################################################

#     # fill encoding
#     enc = header.Encoding()

#     # encoding and reconstruction space
#     encSpace = header.EncodingSpace()

#     # matrix
#     nslices = dset[0x0021, 0x104f].value
#     encSpace.matrixSize = header.MatrixSize()
#     encSpace.matrixSize.x = dset.Columns
#     encSpace.matrixSize.y = dset.Rows
#     encSpace.matrixSize.z = nslices

#     # FOV
#     encSpace.fieldOfView_mm = header.FieldOfView_mm()
#     if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
#         encSpace.fieldOfView_mm.x = dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]*dset.Rows
#         encSpace.fieldOfView_mm.y = dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[1]*dset.Columns
#         encSpace.fieldOfView_mm.z = (nslices - 1) * float(dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices) + float(dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
#         encSpace.fieldOfView_mm.z = round(encSpace.fieldOfView_mm.z, 2)
#     else:
#         encSpace.fieldOfView_mm.x = dset.PixelSpacing[0]*dset.Rows
#         encSpace.fieldOfView_mm.y = dset.PixelSpacing[1]*dset.Columns
#         encSpace.fieldOfView_mm.z = (nslices - 1) * float(dset.SpacingBetweenSlices)  + float(dset.SliceThickness)
#         encSpace.fieldOfView_mm.z = round(encSpace.fieldOfView_mm.z, 2)
#     enc.encodedSpace = encSpace
#     enc.reconSpace = encSpace

#     # encoding limits
#     enc.encodingLimits = header.EncodingLimits()
#     enc.encodingLimits.kspace_encoding_step_1.maximum = shape[-1]
#     enc.encodingLimits.kspace_encoding_step_1.center = shape[-1] // 2
#     enc.encodingLimits.kspace_encoding_step_2.maximum = shape[-2]
#     enc.encodingLimits.kspace_encoding_step_2.center = shape[-2] // 2
#     enc.encodingLimits.slice.maximum = shape[-3]
#     enc.encodingLimits.slice.center = shape[-3] // 2

#     if shape[0] != 1:
#         nvolumes = np.prod(shape[:-3])
#         enc.encodingLimits.contrast.maximum = nvolumes
#         enc.encodingLimits.contrast.center = nvolumes // 2

#     # trajectory
#     enc.trajectory = header.Trajectory('cartesian')

#     # parallel imaging
#     enc.parallelImaging = header.ParallelImaging()
#     enc.parallelImaging.accelerationFactor = header.AccelerationFactor()
#     if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
#         enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorInPlane
#         enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorOutOfPlane
#     else:
#         enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1 / dset[0x0043, 0x1083].value[0]
#         enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1 / dset[0x0043, 0x1083].value[1]

#     # append encoding
#     mrdHead.encoding = [enc]
#     ####################################################################

#     # parse
#     mrdHead = head["head"]
#     mrdImg = head["meta"]

#     # initailize dicom
#     dicomDset = pydicom.dataset.Dataset()

#     # Enforce explicit little endian for written DICOM files
#     dicomDset.file_meta = pydicom.dataset.FileMetaDataset()
#     dicomDset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
#     dicomDset.file_meta.MediaStorageSOPClassUID = pynetdicom.sop_class.MRImageStorage
#     dicomDset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
#     dicomDset.dataset.validate_file_meta(dicomDset.file_meta)

#     # FileMetaInformationGroupLength is still missing?
#     dicomDset.is_little_endian = True
#     dicomDset.is_implicit_VR = False

#     # ----- Set some mandatory default values -----
#     dicomDset.SamplesPerPixel = 1
#     dicomDset.PhotometricInterpretation = 'MONOCHROME2'
#     dicomDset.PixelRepresentation = 0  # Unsigned integer
#     dicomDset.ImageType = ['ORIGINAL', 'PRIMARY', 'M']

#     # ----- Update DICOM header from MRD header -----
#     # fill subject information
#     dicomDset.PatientName = mrdHead.subjectInformation.patientName
#     dicomDset.PatientWeight = mrdHead.subjectInformation.patientWeight_kg
#     dicomDset.PatientID = mrdHead.subjectInformation.patientID
#     dicomDset.PatientBirthDate = mrdHead.subjectInformation.patientBirthdate
#     dicomDset.PatientSex = mrdHead.subjectInformation.patientGender

#     # fill study information
#     dicomDset.StudyDate = mrdHead.studyInformation.studyDate
#     dicomDset.StudyTime = mrdHead.studyInformation.studyTime
#     dicomDset.StudyID = mrdHead.studyInformation.studyID
#     dicomDset.AccessionNumber = mrdHead.studyInformation.accessionNumber
#     dicomDset.ReferringPhysicianName = mrdHead.studyInformation.referringPhysicianName
#     dicomDset.StudyInstanceUID = mrdHead.studyInformation.studyInstanceUID
#     dicomDset.BodyPartExamined = mrdHead.studyInformation.bodyPartExamined

#     # fill measurement information
#     dicomDset.SeriesInstanceUID = mrdHead.measurementInformation.measurementID
#     dicomDset.SeriesDate = mrdHead.measurementInformation.seriesDate
#     dicomDset.SeriesTime = mrdHead.measurementInformation.seriesTime
#     dicomDset.PatientPosition = mrdHead.measurementInformation.patientPosition.name
#     dicomDset[0x0019, 0x107f].value = mrdHead.measurementInformation.relativeTablePosition
#     dicomDset.ProtocolName = mrdHead.measurementInformation.protocolName
#     dicomDset.SeriesDescription = mrdHead.measurementInformation.seriesDescription
#     dicomDset.SeriesInstanceUID = mrdHead.measurementInformation.seriesInstanceUIDRoot
#     dicomDset.FrameOfReferenceUID = mrdHead.measurementInformation.frameOfReferenceUID
#     dicomDset.ReferencedImageSequence = []
#     for ref in mrdHead.measurementInformation.referencedImageSequence:
#         tmp = pydicom.dataset.Dataset()
#         tmp[0x008, 0x1155].value = ref
#         dicomDset.ReferencedImageSequence.append(tmp)

#     # fill acquisition system information
#     dicomDset.Manufacturer = mrdHead.acquisitionSystemInformation.systemVendor
#     dicomDset.ManufacturerModelName = mrdHead.acquisitionSystemInformation.systemModel
#     dicomDset.MagneticFieldStrength = mrdHead.acquisitionSystemInformation.systemFieldStrength_T
#     dicomDset.InstitutionName = mrdHead.acquisitionSystemInformation.institutionName
#     dicomDset.StationName = mrdHead.acquisitionSystemInformation.stationName
#     dicomDset.DeviceSerialNumber = mrdHead.acquisitionSystemInformation.deviceSerialNumber

#     # Set mrdImg pixel data from MRD mrdImg
#     dicomDset.Rows = data.shape[-2]
#     dicomDset.Columns = data.shape[-1]
#     dicomDset.PixelSpacing = [float(mrdImg.field_of_view[0]) / mrdImg.data.shape[2], float(mrdImg.field_of_view[1]) / mrdImg.data.shape[3]]
#     dicomDset.SliceThickness = mrdImg.field_of_view[2]
#     dicomDset.ImagePositionPatient = [mrdImg.position[0], mrdImg.position[1], mrdImg.position[2]]
#     dicomDset.ImageOrientationPatient = [mrdImg.read_dir[0], mrdImg.read_dir[1], mrdImg.read_dir[2], mrdImg.phase_dir[0], mrdImg.phase_dir[1], mrdImg.phase_dir[2]]

#     dicomDset.AcquisitionTime = mrdImg.acquisition_time_stamp
#     dicomDset.TriggerTime = mrdImg.physiology_time_stamp[0] / 2.5

#     # set bits
#     if mrdImg.data.dtype == 'uint16' or mrdImg.data.dtype == 'int16':
#         dicomDset.BitsAllocated = 16
#         dicomDset.BitsStored = 16
#         dicomDset.HighBit = 15
#     elif mrdImg.data.dtype == 'uint32' or mrdImg.data.dtype == 'int' or mrdImg.data.dtype == 'float32':
#         dicomDset.BitsAllocated = 32
#         dicomDset.BitsStored = 32
#         dicomDset.HighBit = 31
#     elif mrdImg.data.dtype == 'float64':
#         dicomDset.BitsAllocated = 64
#         dicomDset.BitsStored    = 64
#         dicomDset.HighBit       = 63
#     else:
#         print(f"Unsupported data type: {mrdImg.data.dtype}")

#     dicomDset.SeriesNumber = mrdImg.image_series_index
#     dicomDset.InstanceNumber = mrdImg.image_index

#     # ----- Update DICOM header from MRD ImageHeader -----
#     dicomDset.ImageType[2] = imtype_map[mrdImg.image_type]

#     return dset


def _dcm2mrd(data, dcminfo):
    """Create MRD header from a DICOM file"""
    # reshape
    data = data.reshape(-1, *data.shape[-3:])

    if len(data.shape) != 4:
        data = data[None, ...]

    # get info from image
    shape = data.shape

    # initialize header
    mrdHead = header.MRDHeader(version=1)

    # get first dset in info
    dset = dcminfo[0]

    # fill subject information
    mrdHead.subjectInformation = header.SubjectInformation()
    mrdHead.subjectInformation.patientName = dset.PatientName
    mrdHead.subjectInformation.patientWeight_kg = dset.PatientWeight
    try:
        mrdHead.subjectInformation.patientHeight_m = dset.PatientHeight
    except:
        pass
    mrdHead.subjectInformation.patientID = dset.PatientID
    mrdHead.subjectInformation.patientBirthdate = dset.PatientBirthDate
    mrdHead.subjectInformation.patientGender = dset.PatientSex

    # fill study information
    mrdHead.studyInformation = header.StudyInformation()
    mrdHead.studyInformation.studyDate = dset.StudyDate
    mrdHead.studyInformation.studyTime = dset.StudyTime
    mrdHead.studyInformation.studyID = dset.StudyID
    mrdHead.studyInformation.accessionNumber = dset.AccessionNumber
    mrdHead.studyInformation.referringPhysicianName = dset.ReferringPhysicianName
    # mrdHead.studyInformation.studyDescription =
    mrdHead.studyInformation.studyInstanceUID = dset.StudyInstanceUID
    mrdHead.studyInformation.bodyPartExamined = dset.BodyPartExamined

    # fill measurement information
    mrdHead.measurementInformation = header.MeasurementInformation()
    mrdHead.measurementInformation.measurementID = dset.SeriesInstanceUID
    mrdHead.measurementInformation.seriesDate = dset.SeriesDate
    mrdHead.measurementInformation.seriesTime = dset.SeriesTime
    mrdHead.measurementInformation.patientPosition = dset.PatientPosition
    mrdHead.measurementInformation.relativeTablePosition = header.ThreeDimensionalFloat(
        0, 0, float(dset[0x0019, 0x107F].value)
    )
    mrdHead.measurementInformation.initialSeriesNumber = 1
    mrdHead.measurementInformation.protocolName = dset.ProtocolName
    mrdHead.measurementInformation.seriesDescription = dset.SeriesDescription
    # mrdHead.measurementInformation.measurementDependency =
    mrdHead.measurementInformation.seriesInstanceUIDRoot = dset.SeriesInstanceUID
    mrdHead.measurementInformation.frameOfReferenceUID = dset.FrameOfReferenceUID

    mrdHead.measurementInformation.referencedImageSequence = []
    for ref in dset.ReferencedImageSequence:
        mrdHead.measurementInformation.referencedImageSequence.append(
            ref[0x008, 0x1155].value
        )

    # fill acquisition system information
    mrdHead.acquisitionSystemInformation = header.AcquisitionSystemInformation()
    mrdHead.acquisitionSystemInformation.systemVendor = dset.Manufacturer
    mrdHead.acquisitionSystemInformation.systemModel = dset.ManufacturerModelName
    mrdHead.acquisitionSystemInformation.systemFieldStrength_T = float(
        dset.MagneticFieldStrength
    )
    # mrdHead.acquisitionSystemInformation.relativeReceiverNoiseBandwidth =
    # mrdHead.acquisitionSystemInformation.receiverChannels =
    # mrdHead.acquisitionSystemInformation.coilLabel =
    mrdHead.acquisitionSystemInformation.institutionName = dset.InstitutionName
    try:
        mrdHead.acquisitionSystemInformation.stationName = dset.StationName
    except:
        pass
    # mrdHead.acquisitionSystemInformation.deviceID =
    mrdHead.acquisitionSystemInformation.deviceSerialNumber = dset.DeviceSerialNumber

    # fill experimental condiction
    mrdHead.experimentalConditions = header.ExperimentalConditions()
    mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(
        dset.MagneticFieldStrength * 4258e4
    )

    # fill encoding
    enc = header.Encoding()

    # encoding and reconstruction space
    encSpace = header.EncodingSpace()

    # matrix
    nslices = dset[0x0021, 0x104F].value
    encSpace.matrixSize = header.MatrixSize()
    encSpace.matrixSize.x = dset.Columns
    encSpace.matrixSize.y = dset.Rows
    encSpace.matrixSize.z = nslices

    # FOV
    encSpace.fieldOfView_mm = header.FieldOfView_mm()
    if dset.SOPClassUID.name == "Enhanced MR Image Storage":
        encSpace.fieldOfView_mm.x = (
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[0]
            * dset.Rows
        )
        encSpace.fieldOfView_mm.y = (
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[1]
            * dset.Columns
        )
        encSpace.fieldOfView_mm.z = (nslices - 1) * float(
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .SpacingBetweenSlices
        ) + float(
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .SliceThickness
        )
        encSpace.fieldOfView_mm.z = round(encSpace.fieldOfView_mm.z, 2)
    else:
        encSpace.fieldOfView_mm.x = dset.PixelSpacing[0] * dset.Rows
        encSpace.fieldOfView_mm.y = dset.PixelSpacing[1] * dset.Columns
        encSpace.fieldOfView_mm.z = (nslices - 1) * float(
            dset.SpacingBetweenSlices
        ) + float(dset.SliceThickness)
        encSpace.fieldOfView_mm.z = round(encSpace.fieldOfView_mm.z, 2)
    enc.encodedSpace = encSpace
    enc.reconSpace = encSpace

    # encoding limits
    enc.encodingLimits = header.EncodingLimits()
    enc.encodingLimits.kspace_encoding_step_1.maximum = shape[-1]
    enc.encodingLimits.kspace_encoding_step_1.center = shape[-1] // 2
    enc.encodingLimits.kspace_encoding_step_2.maximum = shape[-2]
    enc.encodingLimits.kspace_encoding_step_2.center = shape[-2] // 2
    enc.encodingLimits.slice.maximum = shape[-3]
    enc.encodingLimits.slice.center = shape[-3] // 2

    if shape[0] != 1:
        nvolumes = np.prod(shape[:-3])
        enc.encodingLimits.contrast.maximum = nvolumes
        enc.encodingLimits.contrast.center = nvolumes // 2

    # trajectory
    enc.trajectory = header.Trajectory("cartesian")

    # parallel imaging
    enc.parallelImaging = header.ParallelImaging()
    enc.parallelImaging.accelerationFactor = header.AccelerationFactor()
    if dset.SOPClassUID.name == "Enhanced MR Image Storage":
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = (
            dset.SharedFunctionalGroupsSequence[0]
            .MRModifierSequence[0]
            .ParallelReductionFactorInPlane
        )
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = (
            dset.SharedFunctionalGroupsSequence[0]
            .MRModifierSequence[0]
            .ParallelReductionFactorOutOfPlane
        )
    else:
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = (
            1 / dset[0x0043, 0x1083].value[0]
        )
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = (
            1 / dset[0x0043, 0x1083].value[1]
        )

    # append encoding
    mrdHead.encoding = [enc]

    # sequence parameters
    mrdHead.sequenceParameters = header.SequenceParameters()

    # user parameters
    mrdHead.userParameters = header.UserParameters()
    mrdHead.userParameters.userParameterDouble = []

    # keep slice thickness
    tmpUserDouble = header.UserParameterDouble()
    tmpUserDouble.name = "SliceThickness"
    tmpUserDouble.value = float(dset.SliceThickness)
    mrdHead.userParameters.userParameterDouble.append(tmpUserDouble)
    tmpUserDouble = header.UserParameterDouble()
    tmpUserDouble.name = "SpacingBetweenSlices"
    tmpUserDouble.value = float(dset.SpacingBetweenSlices)
    mrdHead.userParameters.userParameterDouble.append(tmpUserDouble)

    # calculate affine
    affine = Affine.from_dicom(dcminfo, shape[-3:])

    # image header
    # templateHead = image.ImageHeader(data_type=data.dtype)
    # templateHead.image_type = dtype
    # templateHead.field_of_view = (encSpace.fieldOfView_mm.x, encSpace.fieldOfView_mm.y, encSpace.fieldOfView_mm.z)
    # templateHead.position = tuple(np.stack(dset.ImagePositionPatient))
    # templateHead.read_dir = tuple(np.stack(dset.ImageOrientationPatient[0:3]))
    # templateHead.phase_dir = tuple(np.stack(dset.ImageOrientationPatient[3:7]))
    # templateHead.slice_dir = tuple(np.cross(np.stack(dset.ImageOrientationPatient[0:3]), np.stack(dset.ImageOrientationPatient[3:7])))
    # templateHead.patient_table_position = (0.0, 0.0, mrdHead.measurementInformation.relativeTablePosition.z)
    # templateHead.image_series_index = int(dset.SeriesNumber)
    # templateHead.acquisition_time_stamp = dset.AcquisitionTime

    # # update contrast and slice
    # imgHead = []
    # count = 1
    # for m in range(shape[0]):
    #     for n in range(shape[1]):
    #         tmpHead = copy.deepcopy(templateHead)
    #         tmpHead.image_index = count
    #         tmpHead.slice = n
    #         tmpHead.contrast = m
    #         imgHead.append(tmpHead)
    #         count += 1

    return {"head": mrdHead, "affine": affine}


def _get_dicom_info(dsets, index):
    """Get DICOM info structure (remove pixel data)."""
    dcminfo = []

    # SeriesNumber = dsets[index[0]].SeriesNumber

    for n in range(len(index)):
        dset = copy.deepcopy(dsets[index[n]])

        dset.pixel_array[:] = 0.0
        dset.PixelData = dset.pixel_array.tobytes()

        dset.WindowWidth = None
        dset.WindowCenter = None

        # dset.SeriesDescription = None
        # dset.SeriesNumber = SeriesNumber
        # dset.SeriesInstanceUID = None
        # dset.SOPInstanceUID = None

        # dset.InstanceNumber = None

        # try:
        #     dset.ImagesInAcquisition = None
        # except:
        #     pass
        # try:
        #     dset[0x0025, 0x1007].value = None
        # except:
        #     pass
        # try:
        #     dset[0x0025, 0x1019].value = None
        # except:
        #     pass
        # try:
        #     dset[0x2001, 0x9000][0][0x2001, 0x1068][0][0x0028, 0x1052].value = '0.0'
        # except:
        #     pass
        # try:
        #     dset[0x2001, 0x9000][0][0x2001, 0x1068][0][0x0028, 0x1053].value = '1.0'
        # except:
        #     pass
        # try:
        #     dset[0x2005, 0x100e].value = 1.0
        # except:
        #     pass
        # try:
        #     dset[0x0040, 0x9096][0][0x0040,0x9224].value = 0.0
        # except:
        #     pass
        # try:
        #     dset[0x0040, 0x9096][0][0x0040,0x9225].value = 1.0
        # except:
        #     pass

        # dset[0x0018, 0x0086].value = '1' # Echo Number
        # dset.InversionTime = '0'
        # dset.EchoTime = '0'
        # dset.EchoNumbers = '1'
        # dset.EchoTrainLength = '1'
        # dset.RepetitionTime = '0'
        # dset.FlipAngle = '0'

        dcminfo.append(dset)

    return dcminfo
