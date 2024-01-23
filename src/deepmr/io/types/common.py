"""Common types subroutines"""

import numpy as np
import nibabel as nib

from nibabel.orientations import axcodes2ornt, ornt_transform

def _reorient(shape, affine, orientation):
    """
    Reorient input image to desired orientation.
    """
    # get input orientation
    orig_ornt = nib.io_orientation(affine)

    # get target orientation
    targ_ornt = axcodes2ornt(orientation)

    # estimate transform
    transform = ornt_transform(orig_ornt, targ_ornt)

    # reorient
    tmp = np.ones(shape[-3:], dtype=np.float32)
    tmp = nib.Nifti1Image(tmp, affine)
    tmp = tmp.as_reoriented(transform)

    return tmp.affine

def _get_plane_normal(orientation):
    """
    Return array of normal to imaging plane, as the cross product
    between x and y plane versors.
    """
    x, y = orientation
    return np.cross(x, y)