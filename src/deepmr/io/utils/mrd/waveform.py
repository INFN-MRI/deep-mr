"""
This module contain a implementations of ISMRMRD waveform.
For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""

__all__ = ["Waveform"]


from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from .base import Data
from .header import valid_waveform_types

@dataclass
class WaveformHeader:
    """
    Waveform header included in a Waveform object.
    """

    version: np.uint16 = 0
    flags: np.uint64 = 0
    measurement_uid: np.uint32 = 0
    scan_counter: np.uint32 = 0
    time_stamp: np.uint32 = 0
    number_of_samples: np.uint16 = 0
    channels: np.uint16 = 0
    sample_time_us: np.float32 = 0
    waveform_id: str = None

    def __post_init__(self):
        if self.waveform_id is not None:
            self.waveform_id = self.waveform_id.lower()
            valid = valid_waveform_types
            assert (
                self.waveform_id in valid
            ), f"Error! Invalid waveform type. Allowed types are {valid}"

    def serialize(self) -> Dict[str, Any]:
        """
        Dump AcquisitionHeader instance to dictionary.
        """
        # convert instance to dictionary
        return OrderedDict(asdict(self))


@dataclass
class Waveform(Data):
    head: WaveformHeader
    data: npt.NDArray[np.uint32]
