"""Data header structure."""

__all__ = ["Header"]

from dataclasses import dataclass

import warnings

import numpy as np
import pydicom

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
    origin: tuple = (0.0, 0.0, 0.0)
    
    # sequence
    TI: np.ndarray = None
    TE: np.ndarray = None
    TR: np.ndarray = None
    FA: np.ndarray = None
    dt: float = None
        
    # meta
    ref_dicom: pydicom.Dataset = None
    
    @classmethod
    def from_mrd(cls, header, acquisitions):
        
        # first, get relevant fields from mrdheader
        geom = header.encoding[0].encodedSpace
        user = header.userParameters
        
        # calculate geometry parameters
        shape = mrd._get_shape(geom)
        spacing, dz = mrd._get_spacing(user, geom, shape)
        resolution = mrd._get_resolution(geom, shape, dz)
        orientation = mrd._get_image_orientation(acquisitions)
        origin = mrd._get_origin(acquisitions)

        return cls(shape, resolution, spacing, orientation, origin)
    
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
    def from_dicom(cls, dsets):
        
        # first, get position
        position = dicom._get_position(dsets)

        # calculate geometry parameters
        resolution = dicom._get_resolution(dsets)
        origin = dicom._get_origin(position)
        orientation = dicom._get_image_orientation(dsets, True)
        shape = dicom._get_shape(dsets, position)
        spacing = dicom._get_spacing(dsets)
                            
        return cls(shape, resolution, spacing, orientation, origin)
    
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
            
        return cls(shape, resolution, spacing, orientation, origin)
            
    def to_dicom(self):
        pass
    
    def to_nifti(self):
        pass
    

# %% mrd utils
