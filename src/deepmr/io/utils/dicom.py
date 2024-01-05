"""DICOM header utils."""

import numpy as np

# def _make_geometry_tags(affine, shape, resolution, spacing):
#     """
#     Creates DICOM geometry tags from nifti affine.

#     Args:
#         ds: Existing Pydicom DataSet object
#         nii: nifti containing affine
#         sliceNumber: slice number (counting from 1)

#     Ref: https://gist.github.com/tomaroberts/8deebaa0ae204d7ae32fecd1e6efdb51
#     """
#     # reorient affine
#     A = _reorient(shape, affine, "LPS")

#     # sign of affine matrix
#     A[:2, :] *= -1

#     # data parameters & pixel dimensions
#     nz, ny, nx = shape
#     dz, dy, dx = resolution

#     # direction cosines & position parameters
#     dircosX = A[:3, 0] / dx
#     dircosY = A[:3, 1] / dy
#     orientation = [
#         dircosX[0],
#         dircosX[1],
#         dircosX[2],
#         dircosY[0],
#         dircosY[1],
#         dircosY[2],
#     ]
#     orientation = np.asarray(orientation)

#     # calculate position
#     n = np.arange(nz)
#     zero = 0 * n
#     one = zero + 1
#     arr = np.stack((zero, zero, n, one), axis=0)
#     position = A @ arr
#     position = position[:3, :]

#     # calculate slice location
#     slice_loc = _get_relative_slice_position(orientation.reshape(2, 3), position)

#     return spacing, orientation, position.transpose(), slice_loc.round(4)


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


def _get_relative_slice_position(orientation, position):
    """
    Return array of slice coordinates along the normal to imaging plane.
    """
    z = _get_plane_normal(orientation)
    return z @ position


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
    return (float(dsets[0].SliceThickness), float(dsets[0].PixelSpacing[0]), float(dsets[0].PixelSpacing[1]))


def _get_origin(position):
    """
    Return image origin.
    """
    return tuple(position.mean(axis=-1))


def _get_shape(dsets, position):
    """
    Return image shape.
    """
    nz = np.unique(position, axis=-1).shape[-1]
    return (nz, dsets[0].Columns, dsets[0].Rows)


def _get_image_orientation(dsets, astuple=False):
    """
    Return image orientation matrix.
    """
    F = np.asarray(dsets[0].ImageOrientationPatient).reshape(2, 3)
    
    if astuple:
        F = tuple(F.ravel())

    return F


def _get_spacing(dsets):
    """
    Return slice spacing.
    """
    return float(dsets[0].SpacingBetweenSlices)


def _get_flip_angles(dsets):
    """
    Return array of flip angles for each dataset in dsets.
    """
    # get flip angles
    flipAngles = np.array([float(dset.FlipAngle) for dset in dsets])

    return flipAngles


def _get_echo_times(dsets):
    """
    Return array of echo times for each dataset in dsets.
    """
    # get unique echo times
    echoTimes = np.array([float(dset.EchoTime) for dset in dsets])

    return echoTimes


def _get_echo_numbers(dsets):
    """
    Return array of echo numbers for each dataset in dsets.
    """
    # get unique echo times
    echoNumbers = np.array([int(dset.EchoNumbers) for dset in dsets])

    return echoNumbers


def _get_repetition_times(dsets):
    """
    Return array of repetition times for each dataset in dsets.
    """
    # get unique repetition times
    repetitionTimes = np.array([float(dset.RepetitionTime) for dset in dsets])

    return repetitionTimes


def _get_inversion_times(dsets):
    """
    Return array of inversion times for each dataset in dsets.
    """
    try:
        # get unique repetition times
        inversionTimes = np.array([float(dset.InversionTime) for dset in dsets])
    except:
        inversionTimes = np.zeros(len(dsets)) + np.inf

    return inversionTimes


def _get_unique_contrasts(constrasts):
    """
    Return ndarray of unique contrasts and contrast index for each dataset in dsets.
    """
    # get unique repetition times
    uContrasts = np.unique(constrasts, axis=0)

    # get indexes
    contrastIdx = np.zeros(constrasts.shape[0], dtype=np.int16)

    for n in range(uContrasts.shape[0]):
        contrastIdx[(constrasts == uContrasts[n]).all(axis=-1)] = n

    return uContrasts, contrastIdx

