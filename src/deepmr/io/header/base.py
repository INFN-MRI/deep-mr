"""Base I/O routines for acquisition headers."""

__all__ = ["read_base_acqheader", "write_base_acqheader"]

import dacite
from dacite import Config

from dataclasses import asdict

from ..generic import hdf5
from ..generic.pathlib import get_filepath

from ..types.header import Header


def read_base_acqheader(filepath):
    """
    Read acquistion header from hdf5 file.

    Parameters
    ----------
    filepath : str
        Path to the file on disk.

    Returns
    -------
    head : deepmr.Header
        Deserialized acqusition header.

    """
    # get full path
    filepath = get_filepath(filepath, True, "h5")

    # load dictionary
    hdict = hdf5.read_hdf5(filepath)

    # initialize header
    head = dacite.from_dict(Header, hdict, config=Config(check_types=False))

    return head


def write_base_acqheader(head, filepath):
    """
    Write acquistion header to hdf5 file.

    Parameters
    ----------
    head: deepmr.Header
        Structure containing trajectory of shape (ncontrasts, nviews, npts, ndim)
        and meta information (shape, resolution, spacing, etc).
    filepath : str
        Path to mrd file.

    """
    hdict = asdict(head)
    hdf5.write_hdf5(hdict, filepath)
