"""I/O routines for MRD acquisition headers."""

__all__ = ["read_mrd_acqhead", "write_mrd_acqhead"]

import os
import warnings

import numpy as np
import numba as nb

import ismrmrd

from ..generic import mrd
from ..._types.mrd import _numpy_to_bytes


def read_mrd_acqhead(filepath):
    """
    Read acquistion header from mrd file.

    Parameters
    ----------
    filepath : str
        Path to the file on disk.

    Returns
    -------
    head : deepmr.Header
        Deserialized acquiistion header.

    """
    _, head = mrd.read_mrd(filepath, external=True)

    return head


def write_mrd_acqhead(head, filepath):
    """
    Write acquistion header to mrd file.

    Parameters
    ----------
    head: deepmr.Header
        Structure containing trajectory of shape (ncontrasts, nviews, npts, ndim)
        and meta information (shape, resolution, spacing, etc).
    filepath : str
        Path to mrd file.

    """
    # prepare path
    if filepath.endswith(".h5"):
        filepath = filepath[:-3]
    filepath += ".h5"
    filepath = os.path.realpath(filepath)

    if os.path.exists(filepath):
        raise RuntimeError(f"{filepath} already existing!")

    # initialize xml header
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xmlhead = _create_hdr(head)

    # get dwell time
    dt = int(np.diff(head.t)[0] * 1e3)  # ms to us

    # prepare trajecoty and dcf
    traj = head.traj.astype(np.float32)
    if head.dcf is None:
        dcf = np.ones(traj.shape[:-1], dtype=np.float32)
    else:
        dcf = head.dcf.astype(np.float32)
    traj = np.concatenate((traj, dcf[..., None]), axis=-1)

    # get number of acquisitions
    nacq = np.prod(traj.shape[:-2])

    # loop and prepare acquisitions
    if "ordering" in head.user:
        ordering = head.user["ordering"]
    else:
        ordering = None

    # preallocate inverse ordering
    if traj.shape[-1] - 1 == 2:
        ncontrasts, nslices, nviews, npts = (
            traj.shape[0],
            head.shape[0],
            traj.shape[1],
            traj.shape[-2],
        )
    elif traj.shape[-1] - 1 == 3:
        ncontrasts, nslices, nviews, npts = (
            traj.shape[0],
            1,
            traj.shape[1],
            traj.shape[-2],
        )

    iordering = np.zeros((ncontrasts, nslices, nviews), dtype=int)
    if ordering is not None:
        icontrast, iz, iview = ordering
        _invert_ordering(iordering, nacq, icontrast, iz, iview)
    else:
        # assume (from innermost to outermost) contrast -> views -> slices
        _initialize_ordering(iordering, ncontrasts, nslices, nviews)
        icontrast, iview, iz = np.broadcast_arrays(
            np.arange(ncontrasts),
            np.arange(nviews)[:, None],
            np.arange(nslices)[:, None, None],
        )
        icontrast, iview, iz = icontrast.ravel(), iview.ravel(), iz.ravel()

    # reshape trajectory and prepare dummy data
    traj = traj.reshape(nacq, npts, -1)
    dummy = np.zeros((1, npts), dtype=np.complex64)

    # prepare
    prot = ismrmrd.Dataset(filepath)
    prot.write_xml_header(xmlhead)
    for n in range(nacq):
        # add acquisitions to metadata
        acq = ismrmrd.Acquisition.from_array(dummy, traj[n])
        acq.sample_time_us = dt
        acq.idx.kspace_encode_step_1 = iview[n]
        acq.idx.slice = iz[n]
        acq.idx.contrast = icontrast[n]
        prot.append_acquisition(acq)

    # finalize
    prot.close()


# %% utils
def _create_hdr(head):
    hdr = ismrmrd.xsd.ismrmrdHeader()

    # encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType("other")

    # set fov and matrix size
    fov = np.asarray(head.shape) * np.asarray(
        [head._spacing] + list(head.resolution)[1:]
    )

    efov = ismrmrd.xsd.fieldOfViewMm()
    efov.z, efov.y, efov.x = fov

    rfov = ismrmrd.xsd.fieldOfViewMm()
    rfov.z, rfov.y, rfov.x = fov

    ematrix = ismrmrd.xsd.matrixSizeType()
    ematrix.z, ematrix.y, ematrix.x = head.shape

    rmatrix = ismrmrd.xsd.matrixSizeType()
    rmatrix.z, rmatrix.y, rmatrix.x = head.shape

    # set encoded and recon spaces
    espace = ismrmrd.xsd.encodingSpaceType()
    espace.matrixSize = ematrix
    espace.fieldOfView_mm = efov

    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix
    rspace.fieldOfView_mm = rfov

    encoding.encodedSpace = espace
    encoding.reconSpace = rspace

    # encoding limits
    if head.traj.shape[-1] - 1 == 2:
        nslices, ncontrasts, nviews = (
            head.shape[0],
            head.traj.shape[0],
            head.traj.shape[1],
        )
    elif head.traj.shape[-1] - 1 == 3:
        nslices, ncontrasts, nviews = 1, head.traj.shape[0], head.traj.shape[1]

    limits = ismrmrd.xsd.encodingLimitsType()
    limits.slice = ismrmrd.xsd.limitType()
    limits.slice.minimum = 0
    limits.slice.maximum = nslices - 1
    limits.slice.center = int(nslices // 2)

    limits.contrast = ismrmrd.xsd.limitType()
    limits.contrast.minimum = 0
    limits.contrast.maximum = ncontrasts - 1
    limits.contrast.center = int(ncontrasts // 2)

    limits.kspace_encoding_step_1 = ismrmrd.xsd.limitType()
    limits.kspace_encoding_step_1.minimum = 0
    limits.kspace_encoding_step_1.maximum = nviews - 1
    limits.kspace_encoding_step_1.center = int(nviews // 2)

    encoding.encodingLimits = limits

    # append encoding
    hdr.encoding.append(encoding)

    # user parameters
    hdr.userParameters = ismrmrd.xsd.userParametersType()
    slice_thickness = ismrmrd.xsd.userParameterDoubleType()
    slice_thickness.name = "SliceThickness"
    slice_thickness.value = head.resolution[0]
    hdr.userParameters.userParameterDouble.append(slice_thickness)

    spacing = ismrmrd.xsd.userParameterDoubleType()
    spacing.name = "SpacingBetweenSlices"
    spacing.value = head._spacing
    hdr.userParameters.userParameterDouble.append(spacing)

    if "slice_profile" in head.user:
        slice_profile = ismrmrd.xsd.userParameterStringType()
        slice_profile.name = "slice_profile"
        slice_profile.value = _numpy_to_bytes(
            head.user["slice_profile"].astype(np.float32)
        )
        hdr.userParameters.userParameterString.append(slice_profile)
        head.user.pop("slice_profile", None)

    if "basis" in head.user:
        basis = ismrmrd.xsd.userParameterStringType()
        basis.name = "basis"
        basis.value = _numpy_to_bytes(head.user["basis"].astype(np.complex64))
        hdr.userParameters.userParameterString.append(basis)
        head.user.pop("basis", None)

    if "mode" in head.user:
        mode = ismrmrd.xsd.userParameterStringType()
        mode.name = "mode"
        mode.value = head.user["mode"]
        hdr.userParameters.userParameterString.append(mode)
        head.user.pop("mode", None)

    if "separable" in head.user:
        mode = ismrmrd.xsd.userParameterStringType()
        mode.name = "separable"
        mode.value = str(head.user["separable"])
        hdr.userParameters.userParameterString.append(mode)
        head.user.pop("separable", None)

    for k in head.user:
        value = head.user[k]
        if np.issubdtype(type(value), int):
            tmp = ismrmrd.xsd.userParameterLongType()
            tmp.name = k
            tmp.value = int(value)
            hdr.userParameters.userParameterLong.append(tmp)
        if np.issubdtype(type(value), np.floating):
            tmp = ismrmrd.xsd.userParameterDoubleType()
            tmp.name = k
            tmp.value = float(value)
            hdr.userParameters.userParameterDouble.append(tmp)
        if isinstance(value, str):
            tmp = ismrmrd.xsd.userParameterStringType()
            tmp.name = k
            tmp.value = value
            hdr.userParameters.userParameterString.append(tmp)

    # sequence parameters
    sequence = ismrmrd.xsd.sequenceParametersType()

    # append sequence parameters
    if head.FA is not None:
        if np.iscomplexobj(head.FA):
            head.FA = head.FA.astype(np.complex64)
            rf_phase = np.angle(head.FA)
            tmp = ismrmrd.xsd.userParameterStringType()
            tmp.name = "rf_phase"
            tmp.value = _numpy_to_bytes(np.rad2deg(rf_phase).astype(np.float32))
            hdr.userParameters.userParameterString.append(tmp)
            head.FA = abs(head.FA).astype(np.float32)

    for n in range(ncontrasts):
        if head.FA is not None:
            sequence.flipAngle_deg.append(head.FA[n])
        if head.TI is not None:
            sequence.TI.append(head.TI[n])
        if head.TE is not None:
            sequence.TE.append(head.TE[n])
        if head.TR is not None:
            sequence.TR.append(head.TR[n])

    # append
    hdr.sequenceParameters = sequence

    return hdr.toXML("utf-8")


@nb.njit(cache=True)
def _invert_ordering(output, nframes, echo_num, slice_num, view_num):
    # actual reordering
    for n in range(nframes):
        iecho = echo_num[n]
        islice = slice_num[n]
        iview = view_num[n]
        output[iecho, islice, iview] = n


@nb.njit(cache=True)
def _initialize_ordering(output, echo_num, slice_num, view_num):
    nframes = 0
    for z in range(slice_num):
        for v in range(view_num):
            for c in range(echo_num):
                output[c, z, v] = nframes
                nframes += 1
