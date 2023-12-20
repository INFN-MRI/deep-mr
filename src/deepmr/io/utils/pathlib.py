"""Common utils for I/O."""

__all__ = ["get_filepath"]

import glob
import os
import warnings


def get_filepath(filename: str = None, *extensions: str) -> str:
    """
    Retrieve absolute path for given filename.

    If name is not provided, search for all files of given extension.
    If multiple extensions are specified and folder contains files with
    multiple extensions, pick the first one. If multiple files with the same
    extension are found, pick the largest.

    Args:
        filename (str, optional): Filename. Defaults to None.
        *extensions (str): Files extensions to be searched for.

    Returns:
        (str): absolute path of corresponding to input filename.

    """
    # default files
    if filename is None:
        assert len(extensions) > 0, "Please provide at least file extensions."
        filename = ["*." + ext for ext in extensions]
    else:
        filename = [filename]

    # get paths
    filename = [glob.glob(f) for f in filename]

    # select non-empty lists
    filename = [f for f in filename if len(f) > 0]

    # check resulting lists
    if len(filename) > 1:
        warnings.warn(f"Multiple filetypes; picking {extensions[0]}.", UserWarning)
    filename = filename[0]

    # pick largest possible file
    if len(filename) > 1:
        warnings.warn("Multiple files; picking largest.", UserWarning)
        filename = max(filename, key=lambda x: os.stat(x).st_size)
    else:
        filename = filename[0]

    # get full path
    filename = os.path.normpath(os.path.abspath(filename))

    return filename
