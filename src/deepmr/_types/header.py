"""Data header structure."""

__all__ = ["Header"]

import copy
from datetime import date

from dataclasses import dataclass
from dataclasses import field

import warnings

import numpy as np
import pydicom
import torch

from .._external.nii2dcm.dcm import DicomMRI

from . import dicom
from . import gehc
from . import mrd
from . import nifti


@dataclass
class Header:
    """
    Acquisition Header containing sequence description.

    The header info (e.g., k-space trajectory, shape) can be used to
    simulate acquisitions or to inform raw data loading (e.g., via ordering)
    to reshape from acquisition to reconstruction ordering and image post-processing
    (transposition, flipping) and exporting.

    Attributes
    ----------
    shape : torch.Tensor
        This is the expected image size of shape (nz, ny, nx).
    resolution : torch.Tensor
        This is the expected image resolution in mm of shape (dz, dy, dx).
    t : torch.Tensor
        This is the readout sampling time (0, t_read) in ms.
        with shape (nsamples,).
    traj : torch.Tensor
        This is the k-space trajectory normalized as (-0.5, 0.5)
        with shape (ncontrasts, nviews, nsamples, ndims).
    dcf : torch.Tensor
        This is the k-space sampling density compensation factor
        with shape (ncontrasts, nviews, nsamples).
    FA : torch.Tensor, float
        This is either the acquisition flip angle in degrees or the list
        of flip angles of shape (ncontrasts,) for each image in the series.
    TR : torch.Tensor, float
        This is either the repetition time in ms or the list
        of repetition times of shape (ncontrasts,) for each image in the series.
    TE  : torch.Tensor, float
        This is either the echo time in ms or the list
        of echo times of shape (ncontrasts,) for each image in the series.
    TI : torch.Tensor, float
        This is either the inversion time in ms or the list
        of inversion times of shape (ncontrasts,) for each image in the series.
    user : dict
        User parameters. Some examples are:

        * ordering (torch.Tensor):
            Indices for reordering (acquisition to reconstruction)
            of acquired k-space data, shaped (3, nslices * ncontrasts * nview), whose rows are
            'contrast_index', 'slice_index' and 'view_index', respectively.
        * mode (str):
            Acquisition mode ('2Dcart', '3Dcart', '2Dnoncart', '3Dnoncart').
        * separable (bool):
            Whether the acquisition can be decoupled by fft along slice / readout directions
            (3D stack-of-noncartesian / 3D cartesian, respectively) or not (3D noncartesian and 2D acquisitions).
        * slice_profile (torch.Tensor):
            Flip angle scaling along slice profile of shape (nlocs,).
        * basis (torch.Tensor):
            Low rank subspace basis for subspace reconstruction of shape (ncoeff, ncontrasts).

    affine : np.ndarray
        Affine matrix describing image spacing, orientation and origin of shape (4, 4).
    ref_dicom : pydicom.Dataset
        Template dicom for image export.
    flip : list
        List of spatial axis to be flipped after image reconstruction.
        The default is an empty list (no flipping).
    transpose : list
        Permutation of image dimensions after reconstruction, depending on acquisition mode:

        * **2Dcart:** reconstructed image has (nslices, ncontrasts, ny, nx) -> transpose = [1, 0, 2, 3]
        * **2Dnoncart:** reconstructed image has (nslices, ncontrasts, ny, nx) -> transpose = [1, 0, 2, 3]
        * **3Dcart:** reconstructed image has (ncontrasts, nz, ny, nx) -> transpose = [0, 1, 2, 3]
        * **3Dnoncart:** reconstructed image has (nx, ncontrasts, nz, ny) -> transpose = [1, 2, 3, 0]

        The default is an empty list (no transposition).

    """

    ## public attributes
    # recon
    shape: tuple  # (z, y, x)
    resolution: tuple = (1.0, 1.0, 1.0)  # mm (z, y, x)
    t: np.ndarray = None  # sampling time in ms
    traj: np.ndarray = None  # (ncontrasts, nviews, nsamples, ndims)
    dcf: np.ndarray = None  # ([ncontrasts, nviews], nsamples)

    # image post processing
    flip: list = None
    transpose: list = None

    # image export
    affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    ref_dicom: pydicom.Dataset = None

    # contrast parameters
    FA: np.ndarray = None
    TR: np.ndarray = None
    TE: np.ndarray = None
    TI: np.ndarray = None
    user: dict = field(default_factory=lambda: {})  # mainly (slice_profile , basis)

    ## private attributes
    _adc: np.ndarray = None
    _shift: tuple = (0.0, 0.0, 0.0)
    _spacing: float = None
    _orientation: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def torch(self, device="cpu"):
        """
        Cast internal attributes to Pytorch.

        Parameters
        ----------
        device : str, optional
            Computational device for internal attributes. The default is "cpu".

        """
        self.shape = torch.as_tensor(
            copy.deepcopy(self.shape), dtype=int, device=device
        )
        self.resolution = torch.as_tensor(self.resolution, dtype=float, device=device)
        if self.traj is not None:
            self.traj = torch.as_tensor(
                np.ascontiguousarray(self.traj), dtype=torch.float32, device=device
            )
        if self.dcf is not None:
            self.dcf = torch.as_tensor(
                np.ascontiguousarray(self.dcf), dtype=torch.float32, device=device
            )
        if self.FA is not None:
            if np.isscalar(self.FA):
                if np.isreal(self.FA):
                    self.FA = torch.as_tensor(
                        self.FA, dtype=torch.float32, device=device
                    )
                else:
                    self.FA = torch.as_tensor(
                        self.FA, dtype=torch.complex64, device=device
                    )
            else:
                if np.isreal(self.FA).all():
                    self.FA = torch.as_tensor(
                        self.FA, dtype=torch.float32, device=device
                    )
                else:
                    self.FA = torch.as_tensor(
                        self.FA, dtype=torch.complex64, device=device
                    )
        if self.TR is not None:
            self.TR = torch.as_tensor(self.TR, dtype=torch.float32, device=device)
        if self.TE is not None:
            self.TE = torch.as_tensor(self.TE, dtype=torch.float32, device=device)
        if self.TI is not None:
            self.TI = torch.as_tensor(self.TI, dtype=torch.float32, device=device)
        if "slice_profile" in self.user:
            self.user["slice_profile"] = torch.as_tensor(
                self.user["slice_profile"], dtype=torch.float32, device=device
            )
        if "basis" in self.user:
            if np.isreal(self.user["basis"]).all():
                self.user["basis"] = torch.as_tensor(
                    self.user["basis"], dtype=torch.float32, device=device
                )
            else:
                self.user["basis"] = torch.as_tensor(
                    self.user["basis"], dtype=torch.complex64, device=device
                )

    def B0(self):
        """
        Get B0 intensity and direction.

        Returns
        -------
        B0 : float
            Field intensity in T.
        B0vec : np.ndarray
            Field direction (x, y, z).

        """
        B0 = float(self.ref_dicom.MagneticFieldStrength)
        orient = np.asarray(self._orientation).reshape(2, 3)
        B0vec = np.cross(orient[0], orient[1])
        return B0, B0vec

    def numpy(self):
        """Cast internal attributes to Numpy."""
        if isinstance(self.shape, torch.Tensor):
            self.shape = self.shape.numpy()
        if isinstance(self.resolution, torch.Tensor):
            self.resolution = self.resolution.numpy()
        if self.traj is not None and isinstance(self.traj, torch.Tensor):
            self.traj = self.traj.numpy()
        if self.dcf is not None and isinstance(self.dcf, torch.Tensor):
            self.dcf = self.dcf.numpy()
        if self.FA is not None and isinstance(self.FA, torch.Tensor):
            self.FA = self.FA.numpy()
        if self.TR is not None and isinstance(self.TR, torch.Tensor):
            self.TR = self.TR.numpy()
        if self.TE is not None and isinstance(self.TE, torch.Tensor):
            self.TE = self.TE.numpy()
        if self.TI is not None and isinstance(self.TI, torch.Tensor):
            self.TI = self.TI.numpy()
        if "slice_profile" in self.user and isinstance(
            self.user["slice_profile"], torch.Tensor
        ):
            self.user["slice_profile"] = self.user["slice_profile"].numpy()
        if "basis" in self.user and isinstance(self.user["basis"], torch.Tensor):
            self.user["basis"] = self.user["basis"].numpy()

    def __post_init__(self):  # noqa
        # cast
        if self.TI is not None:
            self.TI = np.asarray(self.TI, dtype=np.float32)
        if self.TE is not None:
            self.TE = np.asarray(self.TE, dtype=np.float32)
        if self.TR is not None:
            self.TR = np.asarray(self.TR, dtype=np.float32)
        if self.FA is not None:
            if np.iscomplexobj(self.FA):
                self.FA = np.asarray(self.FA, dtype=np.complex64)
            else:
                self.FA = np.asarray(self.FA, dtype=np.float32)

        # fix spacing
        if self._spacing is None:
            self._spacing = self.resolution[0]

        # convert orientation to tuple
        if isinstance(self._orientation, np.ndarray):
            self._orientation = self._orientation.ravel()
        if isinstance(self._orientation, list) is False:
            self._orientation = list(self._orientation)

        # prepare Series tags
        if self.ref_dicom is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # change the hook
                self.ref_dicom = DicomMRI("nii2dcm_dicom_mri.dcm").ds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # support saving of real data only for now
            self.ref_dicom.ImageType.append("M")
            self.ref_dicom.Rows = self.shape[-1]
            self.ref_dicom.Columns = self.shape[-2]
            self.ref_dicom.PixelSpacing = [
                np.round(float(self.resolution[-1]), 2),
                np.round(float(self.resolution[-2]), 2),
            ]
            self.ref_dicom.SliceThickness = np.round(self.resolution[0], 2)
            self.ref_dicom.SpacingBetweenSlices = np.round(self._spacing, 2)
            self.ref_dicom.ImageOrientationPatient = self._orientation
            self.ref_dicom.AcquisitionMatrix = [self.shape[-1], self.shape[-2]]

            # make sure SeriesInstanceUID is unique
            self.ref_dicom.SeriesInstanceUID = pydicom.uid.generate_uid()

            # fill Patient Age
            if self.ref_dicom.PatientAge == "":
                if (
                    self.ref_dicom.PatientBirthDate != ""
                    and self.ref_dicom.StudyDate != ""
                ):
                    self.ref_dicom.PatientAge = _calculate_age(
                        self.ref_dicom.PatientBirthDate, self.ref_dicom.StudyDate
                    )

            try:
                self.ref_dicom.ImagesInAcquisition = ""
            except Exception:
                pass
            try:
                self.ref_dicom[0x0025, 0x1007].value = ""
            except Exception:
                pass
            try:
                self.ref_dicom[0x0025, 0x1019].value = ""
            except Exception:
                pass
            try:
                self.ref_dicom[0x2001, 0x9000][0][0x2001, 0x1068][0][
                    0x0028, 0x1052
                ].value = "0.0"
            except Exception:
                pass
            try:
                self.ref_dicom[0x2001, 0x9000][0][0x2001, 0x1068][0][
                    0x0028, 0x1053
                ].value = "1.0"
            except Exception:
                pass
            try:
                self.ref_dicom[0x2005, 0x100E].value = 1.0
            except Exception:
                pass
            try:
                self.ref_dicom[0x0040, 0x9096][0][0x0040, 0x9224].value = 0.0
            except Exception:
                pass
            try:
                self.ref_dicom[0x0040, 0x9096][0][0x0040, 0x9225].value = 1.0
            except Exception:
                pass

            # get constant contrast info and number of contrasts
            contrasts = [self.FA, self.TE, self.TI, self.TR]
            contrasts = [contrast for contrast in contrasts if contrast is not None]
            if contrasts:
                try:
                    ncontrasts = len(contrasts[0])
                except Exception:
                    ncontrasts = 1
            else:
                ncontrasts = 1
            self.ref_dicom.EchoTrainLength = str(ncontrasts)

            if self.TI is None:
                self.ref_dicom.InversionTime = ""
            elif len(np.unique(self.TI)) == 1:
                TI = float(np.unique(self.TI)[0])
                self.ref_dicom.InversionTime = str(round(TI, 2))
            if self.TE is None:
                self.ref_dicom.EchoTime = "0"
            elif len(np.unique(self.TE)) == 1:
                TE = float(np.unique(self.TE)[0])
                self.ref_dicom.EchoTime = str(round(TE, 2))
            if self.TR is None:
                self.ref_dicom.RepetitionTime = "1000"
            elif len(np.unique(self.TR)) == 1:
                TR = float(np.unique(self.TR)[0])
                self.ref_dicom.RepetitionTime = str(round(TR, 2))
            if self.FA is None:
                self.ref_dicom.FlipAngle = "90"
            elif len(np.unique(self.FA)) == 1:
                FA = float(np.unique(self.FA)[0])
                self.ref_dicom.FlipAngle = str(round(abs(FA), 2))

    @classmethod
    def from_mrd(cls, header, acquisitions, firstVolumeIdx, external):
        """
        Construct Header from MRD data.

        Parameters
        ----------
        header : ismsmrd.XMLHeader
            XMLHeader instance loaded from MRD file.
        acquisitions : list(ismsmrd.Acquisition)
            List of Acquisitions loaded from MRD file.
        firstVolumeIdx : int
            Index in acquisitions corresponding to (contrast=0, slice=0, view=0).
        external : bool
            If True, assume we are loading the Sequence description only,
            i.e., no position / orientation info.

        """
        # get other relevant info from header
        geom = header.encoding[0].encodedSpace
        user = header.userParameters

        # calculate geometry parameters
        shape = mrd._get_shape(geom)
        spacing, dz = mrd._get_spacing(user, geom, shape)
        resolution = mrd._get_resolution(geom, shape, dz)

        # get reference dicom
        ref_dicom = mrd._initialize_series_tag(header)

        # get dwell time
        dt = float(acquisitions[0]["head"]["sample_time_us"]) * 1e-3  # ms
        t = np.arange(acquisitions[0]["head"]["number_of_samples"]) * dt

        if external:
            return cls(shape, resolution, t, ref_dicom=ref_dicom, _spacing=spacing)
        else:
            acquisitions = mrd._get_first_volume(acquisitions, firstVolumeIdx)
            orientation = mrd._get_image_orientation(acquisitions)
            position = mrd._get_position(acquisitions)
            affine = nifti._make_nifti_affine(shape, position, orientation, resolution)

            return cls(
                shape,
                resolution,
                t,
                affine=affine,
                ref_dicom=ref_dicom,
                _spacing=spacing,
                _orientation=orientation,
            )

    @classmethod
    def from_gehc(cls, header):
        """
        Construct Header GEHC MRD data.

        Parameters
        ----------
        header : dict
            Dictionary with Header parameters loaded from GEHC data.

        """
        # image reconstruction
        shape = header["shape"]
        t = header["t"]
        traj = header["traj"]
        dcf = header["dcf"]

        # image post processing
        flip = header["flip"]
        transpose = header["transpose"]

        # affine
        spacing = header["spacing"]
        resolution = header["resolution"]
        orientation = header["orientation"]
        position = header["position"]
        affine = nifti._make_nifti_affine(shape, position, orientation, resolution)

        # get reference dicom
        ref_dicom = gehc._initialize_series_tag(header["meta"])

        # get sequence time
        FA = header["FA"]
        TR = header["TR"]
        TE = header["TE"]
        TI = header["TI"]
        user = header["user"]

        # reconstruction options
        adc = header["adc"]
        shift = header["shift"]

        return cls(
            shape,
            resolution,
            t,
            traj,
            dcf,
            flip,
            transpose,
            affine,
            ref_dicom,
            FA,
            TR,
            TE,
            TI,
            user,
            adc,
            shift,
            spacing,
            orientation,
        )

    # @classmethod
    # def from_siemens(cls):
    #     print("Not Implemented")

    # @classmethod
    # def from_philips(cls):
    #     print("Not Implemented")

    @classmethod
    def from_dicom(cls, dsets, firstVolumeIdx):
        """
        Construct Header from DICOM data.

        Parameters
        ----------
        dsets : list(pydicom.Dataset)
            List of pydicom.Dataset objects containing info for each file in DICOM dataset.
        firstVolumeIdx : int
            Index in acquisitions corresponding to (contrast=0, slice=0, view=0).

        """
        # first, get dsets for the first contrast and calculate slice pos
        dsets = dicom._get_first_volume(dsets, firstVolumeIdx)
        position = dicom._get_position(dsets)

        # calculate geometry parameters
        resolution = dicom._get_resolution(dsets)
        orientation = np.around(dicom._get_image_orientation(dsets), 4)
        shape = dicom._get_shape(dsets, position)
        spacing = dicom._get_spacing(dsets)
        affine = nifti._make_nifti_affine(shape, position, orientation, resolution)

        # get reference dicom
        ref_dicom = dicom._initialize_series_tag(copy.deepcopy(dsets[0]))

        # get dwell time
        try:
            dt = float(dsets[0][0x0019, 0x1018].value) * 1e-6  # ms
            t = np.arange(shape[-1]) * dt
        except Exception:
            t = None

        return cls(
            shape,
            resolution,
            t=t,
            affine=affine,
            ref_dicom=ref_dicom,
            _spacing=spacing,
            _orientation=orientation.ravel(),
        )

    @classmethod
    def from_nifti(cls, img, header, affine, json):
        """
        Construct Header from NIfTI data.

        Parameters
        ----------
        img : np.ndarray
            Image array of shape (nz, ny, nx).
        header : np.ndarray
            NIfTI header.
        affine : np.ndarray
            NIfTI affine matrix.
        json : dict
            Deserialized BIDS NIfTI sidecar.

        """
        # first, reorient affine
        A = nifti._reorient(img.shape[-3:], affine, "LPS")
        A[:2, :] *= -1

        # calculate parameters
        shape = nifti._get_shape(img)
        resolution = nifti._get_resolution(header, json)
        spacing = nifti._get_spacing(header)
        orientation = nifti._get_image_orientation(resolution, A)
        affine = np.around(affine, 4).astype(np.float32)

        # get reference dicom
        ref_dicom = nifti._initialize_series_tag(json)

        # get dwell time
        try:
            dt = float(json["DwellTime"]) * 1e3  # ms
            t = np.arange(shape[-1]) * dt
        except Exception:
            t = None

        return cls(
            shape,
            resolution,
            t=t,
            affine=affine,
            ref_dicom=ref_dicom,
            _spacing=spacing,
            _orientation=orientation,
        )

    # def to_dicom(self):
    #     pass

    # def to_nifti(self):
    #     pass


# %% subroutines
def _calculate_age(start_date, stop_date):
    start_date = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:]))
    stop_date = date(int(stop_date[:4]), int(stop_date[4:6]), int(stop_date[6:]))

    # get years
    years = stop_date.year - start_date.year
    months = stop_date.month - start_date.month
    days = stop_date.day - start_date.day

    if years < 1 and months < 2:
        age = str(days).zfill(3) + "D"
    elif years < 3:
        age = str(months).zfill(3) + "M"
    else:
        delta = months >= 6
        age = str(years + delta).zfill(3) + "Y"

    return age
