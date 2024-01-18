"""Common utils for I/O."""

__all__ = ["get_filepath"]

import glob
import os
import warnings


def get_filepath(
    filename: str = None, pick_largest: bool = False, *extensions: str
) -> str:
    """
    Retrieve sorted absolute path for given filename.

    If name is not provided, search for all files of given extension.
    If multiple extensions are specified and folder contains files with
    multiple extensions, pick the first one.

    Parameters
    ----------
    filename : str, optional
        (partial) name of the file on disk. The default is None.
    pick_largest: bool, optional
        If True and multiple files with the same
        extension are found, pick the largest. The default if False.
    *extensions : str
        Files extensions to be searched for.

    Returns
    -------
    str
        Sorted absolute path of corresponding to input filename.

    """
    assert len(extensions) > 0, "Please provide file extension."

    # default files
    if filename is None:
        # get paths
        filename = ["*." + ext for ext in extensions]
        filename = [glob.glob(f) for f in filename]

        # select non-empty lists
        filename = [f for f in filename if len(f) > 0]

        # check resulting lists
        if len(filename) > 1:
            warnings.warn(f"Multiple filetypes; picking {extensions[0]}.", UserWarning)
        filename = filename[0]
    else:
        filename = glob.glob(filename)

    # pick largest possible file
    if len(filename) > 1 and pick_largest is True:
        warnings.warn("Multiple files; picking largest.", UserWarning)
        filename = max(filename, key=lambda x: os.stat(x).st_size)

    # get full path
    if isinstance(filename, list):
        filename = [os.path.normpath(os.path.abspath(file)) for file in filename]
        filename.sort()
    else:
        filename = os.path.normpath(os.path.abspath(filename))

    # fix length-1 lists
    if isinstance(filename, list) and len(filename) == 1:
        filename = filename[0]

    return filename
