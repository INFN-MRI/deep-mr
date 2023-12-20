"""
This module contain writing routines for ISMRMRD data format.

For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""

__all__ = ["write"]

import os.path
from typing import List

from .header import MRDHeader
from .rawacquisition import RawAcquisitionData
from .. import serialize, hdf

def write(filename, data, head, overwrite=False):
    """
    ISMRMRD raw data writing function.

    Write MRDHeader and list of RawAcquisitionData objects to
    the `filename` ISMRMRD file.


    Args:
        filename (str): Path on disk of the ISMRMRD dataset.
        data (List[RawAcquisitionData]): list of RawAcquisitionData objects.
        head (XMLHeader); MRDHeader object.
        overwrite (bool): If True, force overwrite of existing 'filename' file.
            Defaults to False.

    References:
        Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
            Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
            P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
            T.S. and Hansen, M.S. (2017),
            ISMRM Raw data format: A proposed standard for MRI raw datasets.
            Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

    """
    # initialize structure
    filedict = {"dataset": {}}

    # dump xmlheader
    filedict["dataset"]["xml"] = serialize(head)

    # dump acquisitions
    filedict["dataset"]["data"] = serialize(data)

    # check if file exists
    if os.path.isfile(filename) and overwrite:
        try:
            os.remove(filename)
        except ValueError:
            print(f"Unable to overwrite {filename}")
            return None
    elif os.path.isfile(filename):
        print(f"{filename} already exists.")
        return None
    
    # actual saving
    hdf.dump(filedict, filename)
