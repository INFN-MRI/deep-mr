"""Utilities for image orientation calculation."""

__all__ = ["Affine", "_get_slice_locations"]

import warnings

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform


class Affine:
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

    @staticmethod
    def from_nifti(nii):
        """
        Return affine transform between voxel coordinates and mm coordinates.
        """
        return nii.affine

    @staticmethod
    def from_dicom(dsets, shape):
        """
        Return affine transform between voxel coordinates and mm coordinates as
        described in https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting
        """
        position = _get_position(dsets)
        orientation = _get_image_orientation(dsets)
        resolution = _get_resolution(dsets)

        return _make_nifti_affine(shape, position, orientation, resolution)

    @staticmethod
    def from_mrd(dsets, shape):
        """
        Return affine transform between voxel coordinates and mm coordinates.
        """

    @staticmethod
    def to_dicom(affine, head):
        """
        Creates DICOM geometry tags from nifti affine.

        Args:
            ds: Existing Pydicom DataSet object
            nii: nifti containing affine
            sliceNumber: slice number (counting from 1)

        Ref: https://gist.github.com/tomaroberts/8deebaa0ae204d7ae32fecd1e6efdb51
        """
        # get relevant fields from header
        geom = head.encoding[0].encodedSpace.asdict()
        user = head.userParameters.asdict()

        # get number of slices, resolution and spacing
        shape = geom["matrixSize"]
        nz, ny, nx = shape

        # calculate slice spacing
        if "SliceThickness" in user and "SpacingBetweenSlices" in user:
            dz = user["SpacingBetweenSlices"]
            spacing = user["SpacingBetweenSlices"]
        else:  # attempt to estimate
            warnings.warn(
                "Slice thickness and spacing info not found; assuming contiguous"
                " slices!",
                UserWarning,
            )
            rz = geom["fieldOfView_mm"][0]
            dz = rz / nz
            spacing = dz
        spacing = round(float(spacing), 2)

        # calculate resolution
        _, ry, rx = geom["fieldOfView_mm"]
        dy, dx = ry / ny, rx / nx
        resolution = [dz, dy, dx]

        return _make_geometry_tags(affine, shape, resolution, spacing)


def _make_geometry_tags(affine, shape, resolution, spacing):
    """
    Creates DICOM geometry tags from nifti affine.

    Args:
        ds: Existing Pydicom DataSet object
        nii: nifti containing affine
        sliceNumber: slice number (counting from 1)

    Ref: https://gist.github.com/tomaroberts/8deebaa0ae204d7ae32fecd1e6efdb51
    """
    # reorient affine
    A = _reorient(shape, affine, "LPS")

    # sign of affine matrix
    A[:2, :] *= -1

    # data parameters & pixel dimensions
    nz, ny, nx = shape
    dz, dy, dx = resolution

    # direction cosines & position parameters
    dircosX = A[:3, 0] / dx
    dircosY = A[:3, 1] / dy
    orientation = [
        dircosX[0],
        dircosX[1],
        dircosX[2],
        dircosY[0],
        dircosY[1],
        dircosY[2],
    ]
    orientation = np.asarray(orientation)

    # calculate position
    n = np.arange(nz)
    zero = 0 * n
    one = zero + 1
    arr = np.stack((zero, zero, n, one), axis=0)
    position = A @ arr
    position = position[:3, :]

    # calculate slice location
    slice_loc = _get_relative_slice_position(orientation.reshape(2, 3), position)

    return spacing, orientation, position.transpose(), slice_loc.round(4)


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


def _get_slice_locations(dsets):
    """
    Return array of unique slice locations and slice location index for each dataset in dsets.
    """
    # get orientation and position
    orientation = _get_image_orientation(dsets)
    position = _get_position(dsets)

    # get unique slice locations
    sliceLocs = _get_relative_slice_position(orientation, position).round(decimals=4)
    uSliceLocs, firstSliceIdx = np.unique(sliceLocs, return_index=True)

    # get indexes
    sliceIdx = np.zeros(sliceLocs.shape, dtype=np.int16)

    for n in range(len(uSliceLocs)):
        sliceIdx[sliceLocs == uSliceLocs[n]] = n

    return uSliceLocs, firstSliceIdx, sliceIdx


# %% nifti utils
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


# %% dicom utils
def _get_relative_slice_position(orientation, position):
    """
    Return array of slice coordinates along the normal to imaging plane.
    """
    z = _get_plane_normal(orientation)
    return z @ position


def _get_image_orientation(dsets):
    """
    Return image orientation matrix.
    """
    F = np.asarray(dsets[0].ImageOrientationPatient).reshape(2, 3)

    return F


def _get_plane_normal(orientation):
    """
    Return array of normal to imaging plane, as the cross product
    between x and y plane versors.
    """
    x, y = orientation
    return np.cross(x, y)


def _get_position(dsets):
    """
    Return matrix of image position of size (3, nslices).
    """
    return np.stack([dset.ImagePositionPatient for dset in dsets], axis=1)


def _get_resolution(dsets):
    """
    Return image resolution.
    """
    return np.asarray(
        [dsets[0].SliceThickness, dsets[0].PixelSpacing[0], dsets[0].PixelSpacing[1]]
    )
