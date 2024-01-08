"""ISMRMRD header utils."""


import warnings
import numpy as np


def _find_in_user_params(userField, *keys):
    """
    Find field in MRDHeader UserParameters.
    """
    # find names
    names = [field.name for field in userField]

    # find positions    
    idx = [names.index(k) for k in keys if k in names]
    values = [userField[i] for i in idx]
    
    if len(keys) == len(values):
        return dict(zip(keys, values))
    else:
        return None # One or more keys not found
    

def _get_slice_locations(acquisitions):
    """
    Return array of unique slice locations and slice location index for each acquisition in acquisitions.
    """
    # get orientation and position
    orientation = _get_image_orientation(acquisitions)
    position = _get_position(acquisitions)

    # get unique slice locations
    sliceLocs = _get_relative_slice_position(orientation, position).round(decimals=4)
    uSliceLocs, firstVolumeIdx = np.unique(sliceLocs, return_index=True)

    # get indexes
    sliceIdx = np.zeros(sliceLocs.shape, dtype=np.int16)

    for n in range(len(uSliceLocs)):
        sliceIdx[sliceLocs == uSliceLocs[n]] = n

    return uSliceLocs, firstVolumeIdx, sliceIdx


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


def _get_first_volume(acquisitions, index):
    """
    Get first volume in a multi-contrast series.
    """
    out = [acquisitions[idx] for idx in index]
    
    return out


def _get_shape(geom):
    """
    Return image shape.
    """
    shape = geom.matrixSize
    nz, ny, nx = shape.z, shape.y, shape.x
    shape = (nz, ny, nx)
    return shape


def _get_spacing(user, geom, shape):
    """
    Return slice spacing.
    """
    nz = shape[0]
    tmp = _find_in_user_params(user.userParameterDouble, "SliceThickness", "SpacingBetweenSlices")
    if tmp is not None:
        dz, spacing = tmp["SliceThickness"], tmp["SpacingBetweenSlices"]
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
    return spacing, dz


def _get_resolution(geom, shape, dz):
    """
    Return image resolution.
    """
    _, ny, nx = shape
    ry, rx = geom.fieldOfView_mm.y, geom.fieldOfView_mm.x 
    dy, dx = ry / ny, rx / nx
    resolution = (dz, dy, dx)
    return resolution


def _get_image_orientation(acquisitions):
    """
    Return image orientation matrix.
    """
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
    
    orientation = np.asarray(orientation).reshape(2, 3)
    
    return np.around(orientation, 4)


def _get_position(acquisitions):
    """
    Return matrix of image position of size (3, nslices).
    """
    return np.stack([np.asarray([acq.position[0], acq.position[1], acq.position[2]]) for acq in acquisitions], axis=1)


def _get_origin(acquisitions):
    """
    Return image origin.
    """
    pos = [np.asarray(acq.getHead().position) for acq in acquisitions]
    pos = np.stack(pos, axis=0)
    origin = tuple(pos.mean(axis=0))
    return origin


def _get_flip_angles(header):
    """
    Return array of flip angles for each for each volume.
    """
    try:
        flipAngles = header.sequenceParameters.FA
    except:
        flipAngles = None    

    return np.asarray(flipAngles)


def _get_echo_times(header):
    """
    Return array of echo times for each for each volume.
    """
    try:
        echoTimes = header.sequenceParameters.TE
    except:
        echoTimes = None    
    
    return np.asarray(echoTimes)


def _get_repetition_times(header):
    """
    Return array of repetition times for each volume.
    """
    try:
        repetitionTimes = header.sequenceParameters.TR
    except:
        repetitionTimes = None    
    
    return np.asarray(repetitionTimes)


def _get_inversion_times(header):
    """
    Return array of inversion times for each volume.
    """
    try:
        inversionTimes = header.sequenceParameters.TI
    except:
        inversionTimes = None    
    
    return np.asarray(inversionTimes)