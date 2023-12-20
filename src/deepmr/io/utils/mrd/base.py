"""
This module contain an abstract class for ISMRMRD Data.

Examples are "RawAcquisitionData", "WaveformData" and "ImageData".

For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""
__all__ = ["Data", "Field"]

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import List

from .. import cast_to, get_type

# %% module export


# %% implementation
@dataclass
class Data(ABC):
    """Base class for ISMRMRD Data structures."""

    @property
    @abstractmethod
    def dtype(self):
        """Return class dtype for serialization to NumPy structured array."""

    @abstractmethod
    def serialize(self):
        """Dump Data instance."""

    @classmethod
    @abstractmethod
    def deserialize(self):
        """Load Data instance."""


@dataclass
class Field:
    """Base class for fields of ISMRMRD Data structures and headers."""

    def __post_init__(self):
        for f in fields(self):  # iterate over class fields
            fvalue = getattr(self, f.name)  # get current value
            ftype = get_type(f.type)  # get underlying datatype
            ftype = get_type(ftype)
            fvalue = cast_to(fvalue, ftype)  # cast
            setattr(self, f.name, fvalue)
