"""Two-dimensional spiral sampling."""

__all__ = ["spiral"]

import numpy as np

# this is for stupid Sphinx
try:
    from ... import _design
except Exception:
    pass

from ..._types import Header


def spiral(shape, accel=1, nintl=1, **kwargs):
    r"""
    Design a constant- or multi-density spiral.

    Parameters
    ----------
    shape : Iterable[int]
        Matrix shape ``(in-plane, contrasts=1)``.
    accel : int, optional
        In-plane acceleration. Ranges from ``1`` (fully sampled) to ``nintl``.
        The default is ``1``.
    nintl : int, optional
        Number of interleaves to fully sample a plane.
        The default is ``1``.

    Keyword Arguments
    -----------------
    moco_shape : int
        Matrix size for inner-most (motion navigation) spiral.
        The default is ``None``.
    acs_shape : int
        Matrix size for intermediate inner (coil sensitivity estimation) spiral.
        The default is ``None``.
    acs_nintl : int
        Number of interleaves to fully sample intermediate inner spiral.
        The default is ``1``.
    variant : str
        Type of spiral. Allowed values are:
            
        * ``center-out``: starts at the center of k-space and ends at the edge (default).
        * ``reverse``: starts at the edge of k-space and ends at the center.
        * ``in-out``: starts at the edge of k-space and ends on the opposite side (two 180° rotated arms back-to-back).

    Returns
    -------
    head : Header
        Acquisition header corresponding to the generated spiral.

    Notes
    -----
    The returned ``head`` (:func:`deepmr.Header`) is a structure with the following fields:

    * shape (torch.Tensor):
        This is the expected image size of shape ``(nz, ny, nx)``.
    * t (torch.Tensor):
        This is the readout sampling time ``(0, t_read)`` in ``ms``.
        with shape (nsamples,).
    * traj (torch.Tensor):
        This is the k-space trajectory normalized as ``(-0.5 * shape, 0.5 * shape)``
        with shape ``(ncontrasts, nviews, nsamples, ndims)``.
    * dcf (torch.Tensor):
        This is the k-space sampling density compensation factor
        with shape ``(ncontrasts, nviews, nsamples)``.

    """
    # expand shape if needed
    if np.isscalar(shape):
        shape = [shape, 1]
        
    # assume 1mm iso
    fov = shape[0]

    # design single interleaf spiral
    tmp, _ = _design.spiral(fov, shape[0], accel, nintl, **kwargs)

    # rotate
    ncontrasts = shape[1]
    nviews = max(int(nintl // accel), 1)

    # generate angles
    dphi = (1 - 233 / 377) * 360.0
    phi = np.arange(ncontrasts * nviews) * dphi  # angles in degrees
    phi = np.deg2rad(phi)  # angles in radians

    # build rotation matrix
    rot = _design.angleaxis2rotmat(phi, "z")

    # get trajectory
    traj = tmp["kr"] * tmp["mtx"]
    traj = _design.projection(traj[0].T, rot)
    traj = traj.swapaxes(-2, -1).T
    traj = traj.reshape(ncontrasts, nviews, *traj.shape[-2:])

    # get dcf
    dcf = tmp["dcf"]

    # get shape
    shape = tmp["mtx"]

    # get time
    t = tmp["t"]

    # extra args
    user = {}
    user["moco_shape"] = tmp["moco"]["mtx"]
    user["acs_shape"] = tmp["acs"]["mtx"]
    user["min_te"] = float(tmp["te"][0])

    # get indexes
    head = Header(shape, t=t, traj=traj, dcf=dcf, user=user)
    head.torch()

    return head
