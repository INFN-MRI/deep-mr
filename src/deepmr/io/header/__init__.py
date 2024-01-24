"""Acquisition Header IO routines."""

import copy
import os
import time

import numpy as np

from . import base as _base

# from . import bart as _bart
from . import matlab as _matlab
from . import mrd as _mrd

__all__ = ["read_acquisition_header", "write_acquisition_header"]


def read_acquisition_header(filepath, device="cpu", verbose=False, *args):
    """
    Read acquisition header from file.

    Parameters
    ----------
    filepath : str
        Path to acquisition header file.
    device : str, optional
        Computational device for internal attributes. The default is "cpu".
    verbose : int, optional
        Verbosity level (0=Silent, 1=Less, 2=More). The default is 0.


    Args (matfiles)
    ---------------
    dcfpath : str, optional
        Path to the dcf file.
        The default is None.
    methodpath : str, optional
        Path to the schedule description file.
        The default is None.
    sliceprofpath : str, optional
        Path to the slice profile file.
        The default is None.


    Returns
    -------
    head : deepmr.Header
        Deserialized acquisition header.

    """
    tstart = time.time()
    if verbose >= 1:
        print(f"Reading acquisition header from file {filepath}...", end="\t")

    done = False

    # mrd
    if filepath.endswith(".h5"):
        try:
            head = _mrd.read_mrd_acqhead(filepath)
            done = True
        except Exception:
            pass

    # matfile
    if filepath.endswith(".mat") and not (done):
        try:
            head = _matlab.read_matlab_acqhead(filepath, *args)
            done = True
        except Exception:
            pass

    # bart
    # if filepath.endswith(".cfl") and not(done):
    #     try:
    #         head = _bart.read_bart_acqhead(filepath)
    #         done = True
    #     except Exception:
    #         pass

    # fallback
    if filepath.endswith(".h5") and not (done):
        try:
            done = True
            head = _base.read_base_acqheader(filepath)
        except Exception:
            raise RuntimeError(f"File (={filepath}) not recognized!")

    # check if we loaded data
    if not (done):
        raise RuntimeError(f"File (={filepath}) not recognized!")

    # normalize trajectory
    if head.traj is not None:
        traj_max = ((head.traj**2).sum(axis=-1) ** 0.5).max()
        head.traj = head.traj / (2 * traj_max)  # normalize to (-0.5, 0.5)

    # cast
    head.torch(device)

    # final report
    if verbose == 2:
        print(f"Readout time: {round(float(head.t[-1]), 2)} ms")
        if head.traj is not None:
            print(
                f"Trajectory shape: (ncontrasts={head.traj.shape[0]}, nviews={head.traj.shape[1]}, nsamples={head.traj.shape[2]}, ndim={head.traj.shape[-1]})"
            )
        if head.dcf is not None:
            print(
                f"DCF shape: (ncontrasts={head.dcf.shape[0]}, nviews={head.dcf.shape[1]}, nsamples={head.dcf.shape[2]})"
            )
    if head.FA is not None:
        if len(np.unique(head.FA)) > 1:
            if verbose == 2:
                print(f"Flip Angle train length: {len(head.FA)}")
        else:
            FA = float(np.unique(head.FA)[0])
            head.FA = FA
            if verbose == 2:
                print(f"Constant FA: {round(abs(FA), 2)} deg")
    if head.TR is not None:
        if len(np.unique(head.TR)) > 1:
            if verbose == 2:
                print(f"TR train length: {len(head.TR)}")
        else:
            TR = float(np.unique(head.TR)[0])
            head.TR = TR
            if verbose == 2:
                print(f"Constant TR: {round(TR, 2)} ms")
    if head.TE is not None:
        if len(np.unique(head.TE)) > 1:
            if verbose == 2:
                print(f"Echo train length: {len(head.TE)}")
        else:
            TE = float(np.unique(head.TE)[0])
            head.TE = TE
            if verbose == 2:
                print(f"Constant TE: {round(TE, 2)} ms")
    if head.TI is not None and np.allclose(head.TI, 0.0) is False:
        if len(np.unique(head.TI)) > 1:
            if verbose == 2:
                print(f"Inversion train length: {len(head.TI)}")
        else:
            TI = float(np.unique(head.TI)[0])
            head.TI = TI
            if verbose == 2:
                print(f"Constant TI: {round(TI, 2)} ms")

    tend = time.time()
    if verbose >= 1:
        print(f"done! Elapsed time: {round(tend-tstart, 2)} s...")

    return head


def write_acquisition_header(filename, head, filepath="./", dataformat="hdf5"):
    """
    Write acquisition header to file.

    Parameters
    ----------
    filename : str
        Name of the file.
    head: deepmr.Header
        Structure containing trajectory of shape (ncontrasts, nviews, npts, ndim)
        and meta information (shape, resolution, spacing, etc).
    filepath : str, optional
        Path to file. The default is "./".
    dataformat : str, optional
        Available formats ('mrd' or 'hdf5'). The default is 'hdf5'.

    """
    head = copy.deepcopy(head)
    head.ref_dicom = None
    if dataformat == "hdf5":
        _base.write_base_acqheader(head, os.path.join(filepath, filename))
    elif dataformat == "mrd":
        _mrd.write_mrd_acqhead(head, os.path.join(filepath, filename))
    else:
        raise RuntimeError(
            f"Data format = {dataformat} not recognized! Please use 'mrd' or 'hdf5'"
        )
