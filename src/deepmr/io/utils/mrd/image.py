"""
This module contain a implementations of ISMRMRD ImageData.

For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""

__all__ = ["ImageHeader"]

from dataclasses import asdict, dataclass
from typing import Annotated, Dict

import numpy as np

from .base import Field
from .constants import * # noqa
from .. import arraylike_factory

# %% utils
dtype_mapping = { 
    DATATYPE_USHORT: np.dtype("uint16"), # noqa
    DATATYPE_SHORT: np.dtype("int16"), # noqa
    DATATYPE_UINT: np.dtype("uint32"), # noqa
    DATATYPE_INT: np.dtype("int"), # noqa
    DATATYPE_FLOAT: np.dtype("float32"), # noqa
    DATATYPE_DOUBLE: np.dtype("float64"), # noqa
    DATATYPE_CXFLOAT: np.dtype("complex64"), # noqa
    DATATYPE_CXDOUBLE: np.dtype("complex128"), # noqa
}
inverse_dtype_mapping = {dtype_mapping.get(k): k for k in dtype_mapping}


def get_dtype_from_data_type(val):
    dtype = dtype_mapping.get(val)
    if dtype is None:
        raise TypeError("Unknown image data type: " + str(val))
    return dtype


def get_data_type_from_dtype(dtype):
    type = inverse_dtype_mapping.get(dtype)
    if type is None:
        raise TypeError("Datatype not supported: " + str(dtype))
    return type


# %% implementation
@dataclass
class ImageHeader(Field):
    version: np.uint16 = 1.0
    data_type: np.uint64 = 0
    measurement_uid: np.uint32 = 0
    matrix_size: Annotated[list[np.uint16], POSITION_LENGTH] = arraylike_factory(
        Annotated[list[np.uint16], POSITION_LENGTH]
    )
    field_of_view: Annotated[list[np.uint16], POSITION_LENGTH] = arraylike_factory(
        Annotated[list[np.uint16], POSITION_LENGTH]
    )
    channels: np.uint16 = 0
    position: Annotated[list[np.float32], POSITION_LENGTH] = arraylike_factory(
        Annotated[list[np.float32], POSITION_LENGTH]
    )
    read_dir: Annotated[list[np.float32], DIRECTION_LENGTH] = arraylike_factory(
        Annotated[list[np.float32], DIRECTION_LENGTH]
    )
    phase_dir: Annotated[list[np.float32], DIRECTION_LENGTH] = arraylike_factory(
        Annotated[list[np.float32], DIRECTION_LENGTH]
    )
    slice_dir: Annotated[list[np.float32], DIRECTION_LENGTH] = arraylike_factory(
        Annotated[list[np.float32], DIRECTION_LENGTH]
    )
    patient_table_position: Annotated[
        list[np.float32], POSITION_LENGTH
    ] = arraylike_factory(Annotated[list[np.float32], POSITION_LENGTH])
    average: np.uint16 = 0
    slice: np.uint16 = 0
    contrast: np.uint16 = 0
    phase: np.uint16 = 0
    repetition: np.uint16 = 0
    set: np.uint16 = 0
    acquisition_time_stamp: np.uint32 = 0
    physiology_time_stamp: Annotated[list[np.uint32], PHYS_STAMPS] = arraylike_factory(
        Annotated[list[np.uint32], PHYS_STAMPS]
    )
    image_type: np.uint16 = 0
    image_index: np.uint16 = 0
    image_series_index: np.uint16 = 0
    user_int: Annotated[list[np.int32], USER_INTS] = arraylike_factory(
        Annotated[list[np.int32], USER_INTS]
    )
    user_float: Annotated[list[np.float32], USER_FLOATS] = arraylike_factory(
        Annotated[list[np.float32], USER_FLOATS]
    )
    attribute_string_len: np.uint32 = 0

    def __post_init__(self):
        self.data_type = get_data_type_from_dtype(self.data_type)

    def serialize(self) -> Dict:
        """Dump AcquisitionHeader instance."""
        return asdict(self)
