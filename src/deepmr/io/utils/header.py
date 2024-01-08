"""Data header structure."""

__all__ = ["Header"]

import copy
from dataclasses import dataclass
from dataclasses import field

import warnings

import numpy as np
import pydicom

from ...external.nii2dcm.dcm import DicomMRI

from . import dicom
from . import mrd
from . import nifti

@dataclass
class Header:
    """
    """
    # geometry
    shape: tuple
    resolution: tuple = (1.0, 1.0, 1.0)
    spacing: float = 1.0
    orientation: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    affine: np.ndarray = field(default_factory=lambda: np.eye(shape=(4, 4), dtype=np.float32))
    
    # meta
    ref_dicom: pydicom.Dataset = None
    
    # sequence
    TI: np.ndarray = None
    TE: np.ndarray = None
    TR: np.ndarray = None
    FA: np.ndarray = None
    dt: float = None
        
    def __post_init__(self):
        
        # convert orientation to tuple
        if isinstance(self.orientation, np.ndarray):
            self.orientation = self.orientation.ravel()
        if isinstance(self.orientation, list) is False:
            self.orientation = list(self.orientation)
        
        # transfer Series tags from NIfTI
        self.ref_dicom.Rows = self.shape[2]
        self.ref_dicom.Columns = self.shape[1]
        self.ref_dicom.PixelSpacing = [round(self.resolution[2], 2), round(self.resolution[1], 2)]
        self.ref_dicom.SliceThickness = round(self.resolution[0], 2)
        self.ref_dicom.SpacingBetweenSlices = round(self.spacing, 2)
        self.ref_dicom.ImageOrientationPatient = self.orientation
        self.ref_dicom.AcquisitionMatrix = [self.shape[2], self.shape[1], self.shape[0]]
    
    @classmethod
    def from_mrd(cls, header, acquisitions, firstVolumeIdx):
        
        # first, get acquisition for the first contrast
        acquisitions = mrd._get_first_volume(acquisitions, firstVolumeIdx)
        
        # get other relevant info from header
        geom = header.encoding[0].encodedSpace
        user = header.userParameters
        
        # calculate geometry parameters
        shape = mrd._get_shape(geom)
        spacing, dz = mrd._get_spacing(user, geom, shape)
        resolution = mrd._get_resolution(geom, shape, dz)
        orientation = mrd._get_image_orientation(acquisitions)
        position = mrd._get_position(acquisitions)
        affine = nifti._make_nifti_affine(shape, position, orientation, resolution)
    
        # get reference dicom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # change the hook
            ref_dicom = DicomMRI('nii2dcm_dicom_mri.dcm').ds

        return cls(shape, resolution, spacing, orientation, affine, ref_dicom)
    
    @classmethod
    def from_gehc(cls, header):
        print("Not Implemented")

    @classmethod
    def from_siemens(cls):
        print("Not Implemented")

    @classmethod
    def from_philips(cls):
        print("Not Implemented")
    
    @classmethod
    def from_dicom(cls, dsets, firstVolumeIdx):
        
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
        ref_dicom = copy.deepcopy(dsets[0])
                            
        return cls(shape, resolution, spacing, orientation.ravel(), affine, ref_dicom)
    
    @classmethod
    def from_nifti(cls, img, header, affine, json):

        # first, reorient affine
        A = nifti._reorient(img.shape[-3:], affine, "LPS")
        A[:2, :] *= -1
                        
        # calculate parameters
        shape = nifti._get_shape(img)
        resolution = nifti._get_resolution(header, json)
        spacing = nifti._get_spacing(header)        
        origin = nifti._get_origin(shape, A)
        orientation = nifti._get_image_orientation(resolution, A)
        affine = np.around(affine, 4).astype(np.float32)
                
        # get reference dicom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # change the hook
            ref_dicom = DicomMRI('nii2dcm_dicom_mri.dcm').ds
                    
        return cls(shape, resolution, spacing, orientation, affine, ref_dicom)
            
    def to_dicom(self):
        pass
    
    def to_nifti(self):
        pass
    

# %% mrd utils
