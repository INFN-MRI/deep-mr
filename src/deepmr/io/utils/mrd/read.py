"""
This module contain reading routines for ISMRMRD data format.

For more info, refer to the corresponding paper:

    [1] Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
        Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
        P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
        T.S. and Hansen, M.S. (2017),
        ISMRM Raw data format: A proposed standard for MRI raw datasets.
        Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

"""

__all__ = ["read"]


from .header import MRDHeader
from .rawacquisition import RawAcquisitionData
from .. import deserialize, hdf

def read(filename):
    """
    ISMRMRD raw data reading function.

    Reads the `ISMRMRD` filename and stores returns the corresponding
    MRDHeader and list of RawAcquisitionData objects.


    Args:
        filename (str): Path on disk of the ISMRMRD dataset.

    Returns:
        Tuple[List[RawAcquisitionData], MRDHeader]:
            List of RawAcquisitionData and corresponding MRDHeader.

    References:
        Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V.,
            Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., Kellman,
            P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen,
            T.S. and Hansen, M.S. (2017),
            ISMRM Raw data format: A proposed standard for MRI raw datasets.
            Magn. Reson. Med., 77: 411-421. https://doi.org/10.1002/mrm.26089

    """
    # load from h5 to dict
    filedict = hdf.load(filename)["dataset"]

    # load xmlheader
    head = deserialize(MRDHeader, filedict["xml"])

    # load acquisitions
    data = deserialize(RawAcquisitionData, filedict["data"])

    return data, head
