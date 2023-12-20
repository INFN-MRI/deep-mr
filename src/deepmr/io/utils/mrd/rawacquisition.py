"""
This module contain a implementations of ISMRMRD RawAcquisitionData.

For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""
__all__ = ["RawAcquisitionData", "AcquisitionHeader", "EncodingCounters"]

from dataclasses import asdict, dataclass, field
from typing import Annotated, Dict, List, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt

from .base import Data, Field
from .constants import *
from .. import (
    arraylike_factory,
    get_dict_from_structured_array,
    get_structured_array_dtype,
    get_structured_array_values,
)


@dataclass
class EncodingCounters(Field):
    kspace_encode_step_0: np.uint16 = 0  # MC
    kspace_encode_step_1: np.uint16 = 0
    kspace_encode_step_2: np.uint16 = 0
    average: np.uint16 = 0
    slice: np.uint16 = 0
    contrast: np.uint16 = 0
    phase: np.uint16 = 0
    repetition: np.uint16 = 0
    set: np.uint16 = 0
    segment: np.uint16 = 0
    user: Annotated[List[np.uint16], 8] = arraylike_factory(
        Annotated[List[np.uint16], 8]
    )


@dataclass
class AcquisitionHeader(Field):
    version: np.uint16 = 1
    flags: np.uint64 = 0
    measurement_uid: np.uint32 = 0
    scan_counter: np.uint32 = 0
    acquisition_time_stamp: np.uint32 = 0
    physiology_time_stamp: Annotated[List[np.uint32], PHYS_STAMPS] = arraylike_factory(
        Annotated[List[np.uint32], PHYS_STAMPS]
    )
    number_of_samples: np.uint16 = 0
    available_channels: np.uint16 = 0
    active_channels: np.uint16 = 0
    channel_mask: Annotated[List[np.uint64], CHANNEL_MASKS] = arraylike_factory(
        Annotated[List[np.uint64], CHANNEL_MASKS]
    )
    discard_pre: np.uint16 = 0
    discard_post: np.uint16 = 0
    center_sample: np.uint16 = 0
    encoding_space_ref: np.uint16 = 0
    trajectory_dimensions: np.uint16 = 0
    sample_time_us: np.float32 = 0
    position: Annotated[List[np.float32], POSITION_LENGTH] = arraylike_factory(
        Annotated[List[np.float32], POSITION_LENGTH]
    )
    read_dir: Annotated[List[np.float32], DIRECTION_LENGTH] = arraylike_factory(
        Annotated[List[np.float32], DIRECTION_LENGTH]
    )
    phase_dir: Annotated[List[np.float32], DIRECTION_LENGTH] = arraylike_factory(
        Annotated[List[np.float32], DIRECTION_LENGTH]
    )
    slice_dir: Annotated[List[np.float32], DIRECTION_LENGTH] = arraylike_factory(
        Annotated[List[np.float32], DIRECTION_LENGTH]
    )
    patient_table_position: Annotated[
        List[np.float32], POSITION_LENGTH
    ] = arraylike_factory(Annotated[List[np.float32], POSITION_LENGTH])
    idx: EncodingCounters = field(default_factory=EncodingCounters)
    user_int: Annotated[List[np.int32], USER_INTS] = arraylike_factory(
        Annotated[List[np.int32], USER_INTS]
    )
    user_float: Annotated[List[np.float32], USER_FLOATS] = arraylike_factory(
        Annotated[List[np.float32], USER_FLOATS]
    )

    def __post_init__(self):
        if isinstance(self.idx, dict):
            self.idx = EncodingCounters(**self.idx)

    def serialize(self) -> Dict:
        """Dump AcquisitionHeader instance."""
        return asdict(self)


@dataclass
class RawAcquisitionData(Data):
    """ISMRMRD RawAcquisition class."""

    head: AcquisitionHeader = None
    data: npt.NDArray[np.complex64] = None
    traj: Optional[npt.NDArray[np.float32]] = None

    @property
    def dtype(self):
        """Return class dtype for serialization to NumPy structured array."""
        names = ["head", "traj", "data"]
        formats = [
            get_structured_array_dtype(self.head.serialize()),
            h5py.vlen_dtype(np.dtype("float32")),
            h5py.vlen_dtype(np.dtype("float32")),
        ]
        return np.dtype({"names": names, "formats": formats})

    def serialize(self) -> Tuple:
        """Dump RawAcquisition instance."""
        head = get_structured_array_values(self.head.serialize())
        if self.traj is not None:
            traj = self.traj.ravel()
        else:
            traj = None
        if self.data is not None:
            data = self.data.ravel().view(np.float32)
        else:
            data = None
        return head, traj, data

    @classmethod
    def deserialize(cls, input: np.ndarray):
        """Load RawAcquisition instance."""
        # get header
        head = get_dict_from_structured_array(input[0])
        head = AcquisitionHeader(**head)

        # get trajectory
        traj = (
            input[1]
            .view(np.float32)
            .reshape(head.number_of_samples, head.trajectory_dimensions)
        )

        # get data
        data = (
            input[2]
            .view(np.complex64)
            .reshape(head.active_channels, head.number_of_samples)
        )

        # deserialize
        return cls(head, data, traj)
