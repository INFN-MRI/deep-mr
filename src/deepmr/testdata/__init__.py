"""Sub-package containing test data."""

__all__ = ["testdata"]

from os.path import dirname
from os.path import join as pjoin


validnames = [
    "bart",
    "dicom",
    "gehc",
    "gehc::pfile",
    "gehc::archive",
    "mrd",
    "nifti",
    "siemens",
]


def testdata(name):
    """
    Test dataset for I/O routines.

    Parameters
    ----------
    name : str
        File type to be tested.
        Valid entries are "bart", "dicom", "gehc", "gehc::pfile",
        "gehc::archive", "mrd", "nifti" and "siemens".

    Returns
    -------
    path: str
        file path on disk.

    """
    if name == "bart":
        return pjoin(dirname(__file__), "bart", "ME-SE")
    if name == "dicom":
        return pjoin(dirname(__file__), "dicom")
    if name == "gehc":
        return pjoin(dirname(__file__), "gehc", "*")
    if name == "gehc::pfile":
        return pjoin(dirname(__file__), "gehc", "P20480_GRE.7")
    if name == "gehc::archive":
        return pjoin(dirname(__file__), "gehc", "ScanArchive_GRE.h5")
    if name == "mrd":
        return pjoin(dirname(__file__), "mrd", "spiral.h5")
    if name == "nifti":
        return pjoin(dirname(__file__), "nifti", "*.nii")
    if name == "siemens":
        return pjoin(dirname(__file__), "siemens", "gre.dat")
    raise RuntimeError(f"Name not recognized! valid names are {validnames}")
