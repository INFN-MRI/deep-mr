"""Header class with docstring only for cleaner documentation."""

__all__ = ["Header"]


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

    def torch(self, device="cpu"):
        """
        Cast internal attributes to Pytorch.

        Parameters
        ----------
        device : str, optional
            Computational device for internal attributes. The default is "cpu".

        """
        pass

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
        pass

    def numpy(self):
        """Cast internal attributes to Numpy."""
        pass

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
        pass

    @classmethod
    def from_gehc(cls, header):
        """
        Construct Header GEHC MRD data.

        Parameters
        ----------
        header : dict
            Dictionary with Header parameters loaded from GEHC data.

        """
        pass

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
        pass

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
        pass
