"""Image IO API."""

__all__ = ["read_image", "write_image"]

import time

import numpy as np
import torch

from . import dicom as _dicom
from . import nifti as _nifti

# from .dicom import *  # noqa
# from .nifti import *  # noqa


def read_image(filepath, acqheader=None, device="cpu", verbose=0):
    """
    Read image data from file.

    Supported formats are ``DICOM`` and ``NIfTI``.

    Parameters
    ----------
    filepath : str
        Path to image file. Supports wildcard (e.g., ``/path-to-dicom-exam/*``, ``/path-to-BIDS/*.nii``).
    acqheader : Header, optional
        Acquisition header loaded from trajectory.
        If not provided, assume Cartesian acquisition and infer from data.
        The default is ``None``.
    device : str, optional
        Computational device for internal attributes. The default is ``cpu``.
    verbose : int, optional
        Verbosity level ``(0=Silent, 1=Less, 2=More)``. The default is ``0``.

    Returns
    -------
    image : torch.tensor
        Complex image data.
    head : Header
        Metadata for image reconstruction.

    Example
    -------
    >>> import deepmr

    Get the filenames for exemplary DICOM and NIfTI files.

    >>> dcmpath = deepmr.testdata("dicom")
    >>> niftipath = deepmr.testdata("nifti")

    Load the file contents.

    >>> image_dcm, head_dcm = deepmr.io.read_image(dcmpath)
    >>> image_nii, head_nii = deepmr.io.read_image(niftipath)

    The result is a image/header pair. ``Image`` contains image-space data.
    Here, it represents a 2D cartesian acquisition with 3 echoes, 2 slices and 192x192 matrix size.

    >>> image_dcm.shape
    torch.Size([3, 2, 192, 192])
    >>> image_nii.shape
    torch.Size([3, 2, 192, 192])

    ``Head`` contains the acquisition information. We can inspect the image shape and resolution:

    >>> head_dcm.shape
    tensor([  2, 192, 192])
    >>> head_nii.shape
    tensor([  2, 192, 192])
    >>> head_dcm.ref_dicom.SpacingBetweenSlices
    '10.5'
    >>> head_nii.ref_dicom.SpacingBetweenSlices
    '10.5'
    >>> head_dcm.ref_dicom.SliceThickness
    '7.0'
    >>> head_nii.ref_dicom.SliceThickness
    '7.0'
    >>> head_dcm.ref_dicom.PixelSpacing
    [0.67, 0.67]
    >>> head_nii.ref_dicom.PixelSpacing
    [0.67,0.67]

    ``Head`` also contains contrast information (for forward simulation and parameter inference):

    >>> head_dcm.FA
    180.0
    >>> head_nii.FA
    180.0
    >>> head_dcm.TE
    tensor([  20.0, 40.0, 60.0])
    >>> head_nii.TE
    tensor([  20.0, 40.0, 60.0])
    >>> head_dcm.TR
    3000.0
    >>> head_nii.TR
    3000.0

    Notes
    -----
    The returned ``image`` tensor contains image space data. Dimensions are defined as following:

        * **2D:** ``(ncontrasts, nslices, ny, nx)``.
        * **3D:** ``(ncontrasts, nz, ny, nx)``.

    The returned ``head`` (:func:`deepmr.io.Header`) is a structure with the following fields:

        * shape (torch.Tensor):
            This is the expected image size of shape ``(nz, ny, nx)``.
        * resolution (torch.Tensor):
            This is the expected image resolution in mm of shape ``(dz, dy, dx)``.
        * t (torch.Tensor):
            This is the readout sampling time ``(0, t_read)`` in ``ms``.
            with shape ``(nsamples,)``.
        * traj (torch.Tensor):
            This is the k-space trajectory normalized as ``(-0.5, 0.5)``
            with shape ``(ncontrasts, nviews, nsamples, ndims)``.
        * dcf (torch.Tensor):
            This is the k-space sampling density compensation factor
            with shape ``(ncontrasts, nviews, nsamples)``.
        * FA (torch.Tensor, float):
            This is either the acquisition flip angle in degrees or the list
            of flip angles of shape ``(ncontrasts,)`` for each image in the series.
        * TR (torch.Tensor, float):
            This is either the repetition time in ms or the list
            of repetition times of shape ``(ncontrasts,)`` for each image in the series.
        * TE (torch.Tensor, float):
            This is either the echo time in ms or the list
            of echo times of shape ``(ncontrasts,)`` for each image in the series.
        * TI (torch.Tensor, float):
            This is either the inversion time in ms or the list
            of inversion times of shape ``(ncontrasts,)`` for each image in the series.
        * user (dict):
            User parameters. Some examples are:

                * ordering (torch.Tensor):
                    Indices for reordering (acquisition to reconstruction)
                    of acquired k-space data, shaped ``(3, nslices * ncontrasts * nview)``, whose rows are
                    ``contrast_index``, ``slice_index`` and ``view_index``, respectively.
                * mode (str):
                    Acquisition mode (``2Dcart``, ``3Dcart``, ``2Dnoncart``, ``3Dnoncart``).
                * separable (bool):
                    Whether the acquisition can be decoupled by fft along ``slice`` / ``readout`` directions
                    (3D stack-of-noncartesian / 3D cartesian, respectively) or not (3D noncartesian and 2D acquisitions).
                * slice_profile (torch.Tensor):
                    Flip angle scaling along slice profile of shape ``(nlocs,)``.
                * basis (torch.Tensor):
                    Low rank subspace basis for subspace reconstruction of shape ``(ncoeff, ncontrasts)``.
        * affine (np.ndarray):
            Affine matrix describing image spacing, orientation and origin of shape ``(4, 4)``.
        * ref_dicom (pydicom.Dataset):
            Template dicom for image export.

    """
    tstart = time.time()
    if verbose >= 1:
        print(f"Reading image from file {filepath}...", end="\t")

    done = False

    # convert header to numpy
    if acqheader is not None:
        acqheader.numpy()

    # dicom
    if verbose == 2:
        t0 = time.time()
    try:
        image, head = _dicom.read_dicom(filepath)
        done = True
    except Exception as e:
        msg0 = e

    # nifti
    if verbose == 2:
        t0 = time.time()
    try:
        image, head = _nifti.read_nifti(filepath)
        done = True
    except Exception as e:
        msg1 = e

    if not (done):
        raise RuntimeError(
            f"File (={filepath}) not recognized! Error:\nDICOM {msg0}\nNIfTI {msg1}"
        )
    if verbose == 2:
        t1 = time.time()
        print(f"done! Elapsed time: {round(t1-t0, 2)} s")

    # load trajectory info from acqheader if present
    if acqheader is not None:
        if acqheader.traj is not None:
            head.traj = acqheader.traj
        if acqheader.dcf is not None:
            head.dcf = acqheader.dcf
        if acqheader.t is not None:
            head.t = acqheader.t
        if acqheader.user is not None:
            head.user = acqheader.user
        if acqheader.FA is not None:
            head.FA = acqheader.FA
        if acqheader.TR is not None:
            head.TR = acqheader.TR
        if acqheader.TE is not None:
            head.TE = acqheader.TE
        if acqheader.TI is not None:
            head.TI = acqheader.TI

    # remove flip and transpose
    head.transpose = None
    head.flip = None

    # final report
    if verbose == 2:
        if len(image.shape) == 3:
            print(
                f"Image shape: (nz={image.shape[-3]}, ny={image.shape[-2]},  nx={image.shape[-1]})"
            )
        else:
            print(
                f"Image shape: (ncontrasts={image.shape[0]}, nz={image.shape[-3]}, ny={image.shape[-2]},  nx={image.shape[-1]})"
            )
        if head.t is not None:
            print(f"Readout time: {round(float(head.t[-1]), 2)} ms")
        if head.traj is not None:
            print(f"Trajectory range: ({head.traj.min()},{head.traj.max()})")
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

    # cast
    image = torch.as_tensor(
        np.ascontiguousarray(image), dtype=torch.complex64, device=device
    )
    head.torch(device)

    tend = time.time()
    if verbose == 1:
        print(f"done! Elapsed time: {round(tend-tstart, 2)} s")
    elif verbose == 2:
        print(f"Total elapsed time: {round(tend-tstart, 2)} s")

    return image, head


def write_image(
    filename,
    image,
    head=None,
    dataformat="nifti",
    filepath="./",
    series_description="",
    series_number_offset=0,
    series_number_scale=1000,
    rescale=False,
    anonymize=False,
    verbose=False,
):
    """
    Write image to disk.

    Parameters
    ----------
    filename : str
        Name of the file.
    image : np.ndarray
        Complex image data of shape ``(ncontrasts, nslices, ny, n)``.
        See ``'Notes'`` for additional information.
    filepath : str, optional
        Path to file. The default is ``./``.
    head : Header, optional
        Structure containing trajectory of shape ``(ncontrasts, nviews, npts, ndim)``
        and meta information (shape, resolution, spacing, etc). If None,
        assume 1mm isotropic resolution, contiguous slices and axial orientation.
        The default is None.
    dataformat : str, optional
        Available formats (``dicom`` or ``nifti``). The default is ``nifti``.
    series_description : str, optional
        Custom series description. The default is ``""`` (empty string).
    series_number_offset : int, optional
        Series number offset with respect to the acquired one.
        Final series number is ``series_number_scale * acquired_series_number + series_number_offset``.
        he default is ``0``.
    series_number_scale : int, optional
        Series number multiplicative scaling with respect to the acquired one.
        Final series number is ``series_number_scale * acquired_series_number + series_number_offset``.
        The default is ``1000``.
    rescale : bool, optional
        If true, rescale image intensity between ``0`` and ``int16_max``.
        Beware! Avoid this if you are working with quantitative maps.
        The default is ``False``.
    anonymize : bool, optional
        If True, remove sensible info from header. The default is ``False``.
    verbose : bool, optional
        Verbosity flag. The default is ``False``.

    Example
    -------
    >>> import deepmr
    >>> import tempfile

    Get the filenames for an example DICOM file.

    >>> filepath = deepmr.testdata("dicom")

    Load the file contents.

    >>> image_orig, head_orig = deepmr.io.read_image(filepath)
    >>> with tempfile.TemporaryDirectory() as tempdir:
    >>>     dcmpath = os.path.join(tempdir, "dicomtest")
    >>>     niftipath = os.path.join(tempdir, "niftitest.nii")
    >>>     deepmr.io.write_image(dcmpath, image_orig, head_orig, dataformat="dicom")
    >>>     deepmr.io.write_image(niftipath, image_orig, head_orig, dataformat="nifti")
    >>>     deepmr.io.write_image(dcmpath, image_orig, head_orig, dataformat="dicom")
    >>>     deepmr.io.write_image(niftipath, image_orig, head_orig, dataformat="nifti")
    >>>     image_dcm, head_dcm = deepmr.io.read_image(dcmpath)
    >>>     image_nii, head_nii = deepmr.io.read_image(niftipath)

    The result is a image/header pair. ``Image`` contains image-space data.
    Here, it represents a 2D cartesian acquisition with 3 echoes, 2 slices and 192x192 matrix size.

    >>> image_dcm.shape
    torch.Size([3, 2, 192, 192])
    >>> image_nii.shape
    torch.Size([3, 2, 192, 192])

    ``Head`` contains the acquisition information. We can inspect the image shape and resolution:

    >>> head_dcm.shape
    tensor([  2, 192, 192])
    >>> head_nii.shape
    tensor([  2, 192, 192])
    >>> head_dcm.ref_dicom.SpacingBetweenSlices
    '10.5'
    >>> head_nii.ref_dicom.SpacingBetweenSlices
    '10.5'
    >>> head_dcm.ref_dicom.SliceThickness
    '7.0'
    >>> head_nii.ref_dicom.SliceThickness
    '7.0'
    >>> head_dcm.ref_dicom.PixelSpacing
    [0.67, 0.67]
    >>> head_nii.ref_dicom.PixelSpacing
    [0.67,0.67]

    ``Head`` also contains contrast information (for forward simulation and parameter inference):

    >>> head_dcm.FA
    180.0
    >>> head_nii.FA
    180.0
    >>> head_dcm.TE
    tensor([  20.0, 40.0, 60.0])
    >>> head_nii.TE
    tensor([  20.0, 40.0, 60.0])
    >>> head_dcm.TR
    3000.0
    >>> head_nii.TR
    3000.0


    Notes
    -----
    When the image to be written is the result of a reconstruction performed on k-space data loaded using :func:`deepmr.io.read_rawdata`,
    axis order depends on acquisition mode:

        * **2Dcart:** ``(nslices, ncontrasts, ny, nx)``
        * **2Dnoncart:** ``(nslices, ncontrasts, ny, nx)``
        * **3Dcart:** ``(ncontrasts, nz, ny, nx)``
        * **3Dnoncart:** ``(nx, ncontrasts, nz, ny)``

    In this case, image should be transposed to ``(ncontrasts, nslices, ny, nx)`` or ``(ncontrasts, nz, ny, nx)`` for 2D/3D acquisitions, respectively.
    If provided, ``head`` will contain the appropriate permutation order (:func:`head.transpose`):

        * **2Dcart:** ``head.transpose = [1, 0, 2, 3]``
        * **2Dnoncart:** ``head.transpose = [1, 0, 2, 3]``
        * **3Dcart:** ``head.transpose = [0, 1, 2, 3]``
        * **3Dnoncart:** ``head.transpose = [1, 2, 3, 0]``

    If ``head`` is not provided, the user shoudl manually transpose the image tensor to match the required shape.

    """
    if dataformat == "dicom":
        _dicom.write_dicom(
            filename,
            image,
            filepath,
            head,
            series_description,
            series_number_offset,
            series_number_scale,
            rescale,
            anonymize,
            verbose,
        )
    elif dataformat == "nifti":
        _nifti.write_nifti(
            filename,
            image,
            filepath,
            head,
            series_description,
            series_number_offset,
            series_number_scale,
            rescale,
            anonymize,
            verbose,
        )
    else:
        raise RuntimeError(
            f"Data format = {dataformat} not recognized! Please use 'dicom' or 'nifti'"
        )
