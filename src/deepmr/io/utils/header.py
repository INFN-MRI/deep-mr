"""Data header structure."""

__all__ = ["Header"]

from dataclasses import dataclass

import warnings

import numpy as np
import pydicom

from . import dicom
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
    EC: np.ndarray = None
    TR: np.ndarray = None
    FA: np.ndarray = None
    
    # meta
    ref_dicom: pydicom.Dataset = None
    
    
    @classmethod
    def from_mrd(cls, header, acquisitions):
        
        # get relevant fields from mrdheader
        geom = header.encoding[0].encodedSpace
        user = header.userParameters
        
        # get number of slices, resolution and spacing
        shape = geom.matrixSize
        nz, ny, nx = shape.z, shape.y, shape.x
        shape = (nz, ny, nx)

        # calculate slice spacing
        tmp = _find_in_user_params(user.userParameterDouble, "SliceThickness", "SpacingBetweenSlices")
        if tmp is not None:
            dz, spacing = tmp["SpacingBetweenSlices"], tmp["SliceThickness"]
        else:
            if nz != 1:
                warnings.warn(
                    "Slice thickness and spacing info not found; assuming contiguous"
                    " slices!",
                    UserWarning,
                )
            rz = geom.fieldOfView_mm.z
            dz = rz / nz
            spacing = dz
        spacing = round(float(spacing), 2)

        # calculate resolution
        ry, rx = geom.fieldOfView_mm.y, geom.fieldOfView_mm.x 
        dy, dx = ry / ny, rx / nx
        resolution = (dz, dy, dx)
        
        # calculate orientation
        tmp = acquisitions[0].getHead()
        dircosX = np.asarray(tmp.read_dir)
        dircosY = np.asarray(tmp.phase_dir)
        orientation = (
            dircosX[0],
            dircosX[1],
            dircosX[2],
            dircosY[0],
            dircosY[1],
            dircosY[2],
        )
        
        # calculate origin
        pos = [np.asarray(acq.getHead().position) for acq in acquisitions]
        pos = np.stack(pos, axis=0)
        origin = tuple(pos.mean(axis=0))
        
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
def _find_in_user_params(userField, *keys):
    
    # find names
    names = [field.name for field in userField]

    # find positions    
    idx = [names.index(k) for k in keys if k in names]
    values = [userField[i] for i in idx]
    
    if len(keys) == len(values):
        return dict(zip(keys, values))
    else:
        return None # One or more keys not found