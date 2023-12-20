"""
This module contain a implementations of ISMRMRD xml and acquisition header.
For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""
# __all__ = ["MRDHeader"]

import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Union, get_args

import numpy as np
from dacite import Config, from_dict

from .base import Field
from .. import remove_none,  xmltodict

# %% aliases
ReferencedImageSequence = str
CalibrationMode = str
DiffusionDimension = str
InterleavingDimension = str
MultibandCalibration = str
PatientGender = str
PatientPosition = str
Trajectory = str
WaveformType = str


# %% restricted types
valid_genders = ["M", "F", "O"]
valid_positions = ["HFP", "HFS", "HFDR", "HFDL", "FFP", "FFS", "FFDR", "FFDL"]
valid_trajectories = ["cartesian", "epi", "radial", "goldenangle", "spiral", "other"]
valid_diffusion_dimensions = [
    "average",
    "contrast",
    "phase",
    "repetition",
    "set",
    "segment",
    "user_0",
    "user_1",
    "user_2",
    "user_3",
    "user_4",
    "user_5",
    "user_6",
    "user_7",
]
valid_calibration_modes = ["embedded", "interleaved", "separate", "external", "other"]
valid_interleaving_dimensions = ["average", "contrast", "phase", "repetition", "other"]
valid_mb_calibration_modes = ["separable2D", "full3D", "other"]
valid_waveform_types = [
    "ecg",
    "pulse",
    "respiratory",
    "trigger",
    "gradientwaveform",
    "other",
]


# %% utility functions
def _get_type(input):
    try:
        input = get_args(input)[0]
    except Exception:
        pass
    return input


def _is_builtin_type(obj):
    builtin_types = {
        np.uint16,
        np.int64,
        np.float32,
        np.float64,
        complex,
        bool,
        str,
        bytes,
        bytearray,
        memoryview,
        list,
        tuple,
        range,
        set,
        frozenset,
        dict,
    }
    return isinstance(obj, tuple(builtin_types))


def _cast_to(input, type):
    if input is not None and _is_builtin_type(input) and type is not str:
        if isinstance(input, List):
            return [type(element) for element in input]
        else:
            return type(input)
    return input


def _custom_factory(input, value=0):
    return field(default_factory=lambda: _init_annotated(input, value))


def _init_annotated(input, value):
    # get length and type
    tmp, length = get_args(input)
    type = get_args(tmp)[0]

    return [type(value) for n in range(length)]


# %% 'low-level' helper classes
@dataclass
class CoilLabel(Field):
    coilNumber: np.uint16 = None
    coilName: str = None


@dataclass
class ExperimentalConditions(Field):
    H1resonanceFrequency_Hz: np.int64 = None


@dataclass
class FieldOfView_mm(Field):
    x: np.float32 = None
    y: np.float32 = None
    z: np.float32 = None


@dataclass
class GradientDirection(Field):
    rl: np.float32 = None
    ap: np.float32 = None
    fh: np.float32 = None


@dataclass
class Limit(Field):
    minimum: np.uint16 = None
    maximum: np.uint16 = None
    center: np.uint16 = None


@dataclass
class MatrixSize(Field):
    x: np.uint16 = None
    y: np.uint16 = None
    z: np.uint16 = None


@dataclass
class MeasurementDependency(Field):
    dependencyType: str = None
    measurementID: str = None


@dataclass
class MultibandSpacing(Field):
    dZ: np.float32 = None


@dataclass
class ThreeDimensionalFloat(Field):
    x: np.float32 = None
    y: np.float32 = None
    z: np.float32 = None


@dataclass
class UserParameterLong(Field):
    name: str = None
    value: np.int64 = None


@dataclass
class UserParameterDouble(Field):
    name: str = None
    value: np.float32 = None


@dataclass
class UserParameterString(Field):
    name: str = None
    value: str = None


@dataclass
class UserParameterBase64(Field):
    name: str = None
    value: str = None  # type: base64

    def __post_init__(self):
        self.value = base64.b64decode(self.value)


@dataclass
class UserParameters(Field):
    userParameterLong: List[UserParameterLong] = None
    userParameterDouble: List[UserParameterDouble] = None
    userParameterString: List[UserParameterString] = None
    userParameterBase64: List[UserParameterBase64] = None

    def asdict(self):
        out = {}

        # iterate over all
        if self.userParameterLong is not None:
            for el in self.userParameterLong:
                out[el.name] = el.value

        if self.userParameterDouble is not None:
            for el in self.userParameterDouble:
                out[el.name] = el.value

        if self.userParameterString is not None:
            for el in self.userParameterString:
                out[el.name] = el.value

        if self.userParameterBase64 is not None:
            for el in self.userParameterBase64:
                out[el.name] = el.value

        return out


# %% 'mid-level' helper classes
@dataclass
class AccelerationFactor(Field):
    kspace_encoding_step_0: np.uint16 = (
        None  # MC (wave-CAIPI accelerate over x as well)
    )
    kspace_encoding_step_1: np.uint16 = None
    kspace_encoding_step_2: np.uint16 = None


@dataclass
class Diffusion(Field):
    gradientDirection: GradientDirection = None
    bvalue: np.float32 = None


@dataclass
class EncodingSpace(Field):
    matrixSize: MatrixSize = None
    fieldOfView_mm: FieldOfView_mm = None

    def asdict(self):
        matrixSize = [self.matrixSize.z, self.matrixSize.y, self.matrixSize.x]
        fieldOfView_mm = [
            self.fieldOfView_mm.z,
            self.fieldOfView_mm.y,
            self.fieldOfView_mm.x,
        ]

        return {"matrixSize": matrixSize, "fieldOfView_mm": fieldOfView_mm}


@dataclass
class EncodingLimits(Field):
    kspace_encoding_step_0: Limit = None
    kspace_encoding_step_1: Limit = None
    kspace_encoding_step_2: Limit = None
    average: Limit = None
    slice: Limit = None
    contrast: Limit = None
    phase: Limit = None
    repetition: Limit = None
    set: Limit = None
    segment: Limit = None
    user_0: Limit = None
    user_1: Limit = None
    user_2: Limit = None
    user_3: Limit = None
    user_4: Limit = None
    user_5: Limit = None
    user_6: Limit = None
    user_7: Limit = None

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                setattr(self, f.name, Limit(0, 0, 0))


@dataclass
class Multiband(Field):
    spacing: MultibandSpacing = None
    deltaKz: np.float32 = None
    multiband_factor: np.int64 = None
    calibration: MultibandCalibration = None
    calibration_encoding: np.int64 = None

    def __post_init__(self):
        if self.calibration is not None:
            self.calibration = self.calibration.lower()
            valid = valid_mb_calibration_modes
            assert (
                self.calibration in valid
            ), f"Error! Invalid calibration mode. Allowed modes are {valid}"


@dataclass
class ParallelImaging(Field):
    accelerationFactor: AccelerationFactor = None
    calibrationMode: CalibrationMode = (
        None  # [embedded, interleaved, separate, external, other]
    )
    interleavingDimension: InterleavingDimension = (
        None  # [phase, repetition, contrast, average, other]
    )
    multiband: Multiband = None

    def __post_init__(self):
        if self.calibrationMode is not None:
            self.calibrationMode = self.calibrationMode.lower()
            valid = valid_calibration_modes
            assert (
                self.calibrationMode in valid_calibration_modes
            ), f"Error! Invalid calibration mode. Allowed modes are {valid}"
        if self.interleavingDimension is not None:
            self.interleavingDimension = self.interleavingDimension.lower()
            valid = valid_interleaving_dimensions
            assert (
                self.interleavingDimension in valid
            ), f"Error! Invalid interleaving dimension. Allowed dimensions are {valid}"
        super().__post_init__()


@dataclass
class TrajectoryDescription(Field):
    identifier: str = None
    userParameterLong: List[UserParameterLong] = None
    userParameterDouble: List[UserParameterDouble] = None
    comment: str = None


# %% 'high-level' helper classes
@dataclass
class SubjectInformation(Field):
    patientName: str = None
    patientWeight_kg: np.float32 = None
    patientHeight_m: np.float32 = None
    patientID: str = None
    patientBirthdate: str = None
    patientGender: PatientGender = None

    def __post_init__(self):
        if self.patientGender is not None:
            self.patientGender = self.patientGender.upper()[0]
            valid = valid_genders
            assert (
                self.patientGender in valid
            ), f"Error! Invalid patient gender. Valid genders are {valid}"
        super().__post_init__()


@dataclass
class StudyInformation(Field):
    studyDate: str = None
    studyTime: str = None
    studyID: np.float32 = None
    accessionNumber: str = None
    referringPhysicianName: str = None
    studyDescription: str = None
    studyInstanceUID: str = None
    bodyPartExamined: str = None


@dataclass
class MeasurementInformation(Field):
    measurementID: str = None
    seriesDate: str = None
    seriesTime: str = None
    patientPosition: PatientPosition = (
        None  # [HFP, HFS, HFDR, HFDL, FFP, FFS, FFDR, FFDL]
    )
    relativeTablePosition: ThreeDimensionalFloat = None
    initialSeriesNumber: np.int64 = None
    protocolName: str = None
    seriesDescription: str = None
    measurementDependency: MeasurementDependency = None
    seriesInstanceUIDRoot: str = None
    frameOfReferenceUID: str = None
    referencedImageSequence: ReferencedImageSequence = None

    def __post_init__(self):
        if self.patientPosition is not None:
            self.patientPosition = self.patientPosition.upper()
            valid = valid_positions
            assert (
                self.patientPosition in valid
            ), f"Error! Invalid patient position. Valid positions are {valid}"
        super().__post_init__()


@dataclass
class AcquisitionSystemInformation(Field):
    systemVendor: str = None
    systemModel: str = None
    systemFieldStrength_T: np.float32 = None
    relativeReceiverNoiseBandwidth: np.float32 = None
    receiverChannels: np.uint16 = None
    coilLabel: Union[CoilLabel, List[CoilLabel]] = None
    institutionName: str = None
    stationName: str = None
    deviceID: str = None
    deviceSerialNumber: str = None


@dataclass
class Encoding(Field):
    encodedSpace: EncodingSpace = None
    reconSpace: EncodingSpace = None
    encodingLimits: EncodingLimits = None
    trajectory: Trajectory = (
        None  # [cartesian, epi, radial, goldenangle, spiral, other]
    )
    trajectoryDescription: TrajectoryDescription = None
    parallelImaging: ParallelImaging = None
    echoTrainLength: np.int64 = None

    def __post_init__(self):
        if self.trajectory is not None:
            self.trajectory = self.trajectory.lower()
            valid = valid_trajectories
            assert (
                self.trajectory in valid
            ), f"Error! Invalid trajectory type. Valid trajectories are {valid}"
        super().__post_init__()


@dataclass
class SequenceParameters(Field):
    TR: Union[np.float32, List[np.float32]] = None
    TE: Union[np.float32, List[np.float32]] = None
    TI: Union[np.float32, List[np.float32]] = None
    flipAngle_deg: Union[np.float32, List[np.float32]] = None
    rfPhase_deg: Union[np.float32, List[np.float32]] = None  # MC
    rfPhaseInc_deg: np.float32 = None  # MC
    sequence_type: str = None
    echo_spacing: np.float32 = None
    diffusionDimension: DiffusionDimension = None
    diffusion: Diffusion = None
    diffusionScheme: str = None

    def __post_init__(self):
        assert (
            self.rfPhase_deg is None or self.rfPhaseInc_deg is None
        ), "Error! Specify either rf phase or rf phase increment, not both."
        if self.diffusionDimension is not None:
            self.diffusionDimension = self.diffusionDimension.lower()
            valid = valid_diffusion_dimensions
            assert (
                self.diffusionDimension in valid
            ), f"Error! Invalid diffusion dimension. Valid dimensions are {valid}"
        super().__post_init__()


@dataclass
class WaveformInformation(Field):
    waveformName: str = None
    waveformType: WaveformType = None
    userParameter: UserParameters = None

    def __post_init__(self):
        if self.waveformType is not None:
            self.waveformType = self.waveformType.lower()
            valid = valid_waveform_types
            assert (
                self.waveformType in valid
            ), f"Error! Invalid waveform type. Allowed types are {valid}"


# %% actual implementation
@dataclass
class MRDHeader(Field):
    """
    Adaptation of ISMRMRD xml schema.
    """

    version: np.int64 = None
    subjectInformation: SubjectInformation = None
    studyInformation: StudyInformation = None
    measurementInformation: MeasurementInformation = None
    acquisitionSystemInformation: AcquisitionSystemInformation = None
    experimentalConditions: ExperimentalConditions = None
    encoding: Encoding = None
    sequenceParameters: SequenceParameters = None
    userParameters: UserParameters = None
    waveformInformation: WaveformInformation = None

    def serialize(self) -> Dict[str, Any]:
        """
        Dump MRDHeader instance while recursively removing
        None values.
        """
        # initialize output dict
        output_dict = OrderedDict()
        output_dict["@xmlns"] = "http://www.ismrm.org/ISMRMRD"
        output_dict["@xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        output_dict["@xmlns:xs"] = "http://www.w3.org/2001/XMLSchema"
        output_dict["@xsi:schemaLocation"] = "http://www.ismrm.org/ISMRMRD ismrmrd.xsd"

        # convert instance to dictionary and filter None
        attribute_dict = OrderedDict(remove_none(asdict(self)))
        output_dict = OrderedDict(
            list(output_dict.items()) + list(attribute_dict.items())
        )

        # convert xml dictionary to string
        xmlstr = xmltodict.unparse({"ismrmrdHeader": output_dict})

        return np.asarray([xmlstr], dtype="O")

    @classmethod
    def deserialize(cls, input: np.ndarray):
        """
        Load MRDHeader instance.
        """
        # get xml string
        xmlstr = input[0]

        # convert xml string to dict
        xmldict = xmltodict.parse(xmlstr)["ismrmrdHeader"]

        # prepare XMLHeader
        return from_dict(cls, xmldict, config=Config(check_types=False))
