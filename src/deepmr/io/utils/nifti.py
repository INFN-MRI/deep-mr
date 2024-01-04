"""NIfTI header utils."""

import numpy as np
import nibabel as nib


def _get_shape(img):
    """
    Return image shape.
    """
    return img.shape[-3:]


def _get_resolution(header, json):
    """
    Return image resolution.
    """ 
    if json:
        resolution = (float(json["SliceThickness"]), header["pixdim"][2], header["pixdim"][1])
    else:
        resolution = (header["pixdim"][3], header["pixdim"][2], header["pixdim"][1])
        
    return resolution


def _get_origin(shape, affine):
    """
    Return image origin.
    """
    N = shape[0]
    if N > 1:
        T1 = affine[:-1, -1]
        TN = affine[:-1, 2] * (N-1) + T1
        origin = (T1 + TN) / 2
    else:
        origin = affine[:-1, -1]
    return affine


def _get_image_orientation(resolution, affine):
    """
    Return image orientation matrix.
    """
    # orientation
    dy, dx = resolution[1], resolution[2]

    # direction cosines & position parameters
    dircosX = affine[:3, 0] / dx
    dircosY = affine[:3, 1] / dy
    orientation = (
        dircosX[0],
        dircosX[1],
        dircosX[2],
        dircosY[0],
        dircosY[1],
        dircosY[2],
    )
    
    return orientation


def _get_spacing(header):
    """
    Return slice spacing.
    """
    return header["pixdim"][3]


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


def _make_nifti_affine(shape, position, orientation, resolution):
    """
    Return affine transform between voxel coordinates and mm coordinates.

    Args:
        shape: volume shape (nz, ny, nx).
        resolution: image resolution in mm (dz, dy, dz).
        position: position of each slice (3, nz).
        orientation: image orientation.

    Returns:
        affine matrix describing image position and orientation.

    Ref: https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting
    """
    # get image size
    nz, ny, nx = shape

    # get resoluzion
    dz, dy, dx = resolution

    # common parameters
    T = position
    T1 = T[:, 0].round(4)

    F = orientation
    dr, dc = np.asarray([dy, dx]).round(4)

    if nz == 1:  # single slice case
        n = _get_plane_normal(orientation)
        ds = float(dz)

        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append(-ds * n, 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    else:  # multi slice case
        N = nz
        TN = T[:, -1].round(4)
        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append((TN - T1) / (N - 1), 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    # sign of affine matrix
    A0[:2, :] *= -1

    # reorient
    A = _reorient(shape, A0, "LAS")

    return A
