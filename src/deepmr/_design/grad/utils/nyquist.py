"""Nyquist limits for different trajectory types."""

__all__ = ["squared_fov", "cubic_fov", "radial_fov", "cylindrical_fov", "spherical_fov"]

import numpy as np


# Cartesian
def squared_fov(fov, resolution):
    """Calculate Nyquist criterion for a squared FOV."""
    # calculate
    npix, dk = _nyquist(fov, resolution, 2)

    # unpack
    nx, ny = npix
    dkx, dky = dk

    return (nx, dkx), (ny, dky)


def cubic_fov(fov, resolution):
    """Calculate Nyquist criterion for a cubic FOV."""
    # calculate
    npix, dk = _nyquist(fov, resolution, 3)

    # unpack
    nx, ny, nz = npix
    dkx, dky, dkz = dk

    return (nx, dkx), (ny, dky), (nz, dkz)


# Non Cartesian
def radial_fov(fov, resolution):
    """Calculate Nyquist criterion for a circular FOV."""
    assert np.isscalar(fov), "Error! 2D Non Cartesian sequence must have isotropic fov"
    assert np.isscalar(
        resolution
    ), "Error! 2D Non Cartesian sequence must have isotropic resolution"

    # calculate
    npix, dk = _nyquist(fov, resolution, 2)

    # unpack
    nr, ntheta = npix
    ntheta = np.ceil(0.5 * np.pi * ntheta)

    dkr, _ = dk
    dktheta = np.pi / ntheta

    return (nr, dkr), (int(ntheta), dktheta)


def cylindrical_fov(fov, resolution):
    """Calculate Nyquist criterion for a cylindrical FOV."""
    if np.isscalar(fov) is False:
        assert (
            len(fov) == 2
        ), "Please specify FOV as scalar (isotropic) or (fov_plane, fov_z)"
        fov = [fov[0], fov[0], fov[1]]

    if np.isscalar(resolution) is False:
        assert len(resolution) == 2, (
            "Please specify resolution as scalar (isotropic) or (inplane_res,"
            " slice_thickness)"
        )
        resolution = [resolution[0], resolution[0], resolution[1]]

    # calculate
    npix, dk = _nyquist(fov, resolution, 3)

    # unpack
    nr, ntheta, nz = npix
    ntheta = np.ceil(np.pi * ntheta)

    dkr, _, dkz = dk
    dktheta = 2 * np.pi / ntheta

    return (nr, dkr), (int(ntheta), dktheta), (nz, dkz)


def spherical_fov(fov, resolution):
    """Calculate Nyquist criterion for a spherical FOV."""
    assert np.isscalar(
        fov
    ), "Error! 3D projection Non Cartesian sequence must have isotropic fov"
    assert np.isscalar(
        resolution
    ), "Error! 3D projection Non Cartesian sequence must have isotropic resolution"

    # calculate
    npix, dk = _nyquist(fov, resolution, 2)

    # unpack
    nr, ntheta = npix
    ntheta = np.ceil(np.pi * ntheta)
    nphi = ntheta

    dkr, _ = dk
    dktheta = 2 * np.pi / ntheta
    dkphi = dktheta

    return (nr, dkr), (int(ntheta), dktheta), (int(nphi), dkphi)


# %% local utils
def _nyquist(fov, resolution, ndim):
    # process args
    if ndim > 1:
        if np.isscalar(fov):
            fov = ndim * [fov]

        if np.isscalar(resolution):
            resolution = ndim * [resolution]

        fov = np.array(fov)
        resolution = np.array(resolution)

    # calculate matrix size
    npix = np.ceil(fov / resolution).astype(int)

    # get dk (sampling density)
    dk = 2 * np.pi / fov  # rad / m

    return npix, dk
