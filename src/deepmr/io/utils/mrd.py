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
    return orientation


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