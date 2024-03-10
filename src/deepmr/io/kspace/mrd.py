"""I/O routines for MRD raw data."""

__all__ = ["read_mrd_rawdata"]

from ..generic import mrd


def read_mrd_rawdata(filepath):
    """
    Read kspace data from MRD file.

    Parameters
    ----------
    filepath : str
        Path to MRD file.

    Returns
    -------
    data : np.ndarray
        Complex k-space data of shape (ncoils, ncontrasts, nslices, nview, npts).
    head : deepmr.Header
        Metadata for image reconstruction.
    """
    data, head = mrd.read_mrd(filepath)

    # normalize trajectory
    if head.traj is not None:
        ndim = head.traj.shape[-1]
        traj_max = ((head.traj**2).sum(axis=-1) ** 0.5).max()
        head.traj = head.traj / (2 * traj_max)  # normalize to (-0.5, 0.5)
        head.traj = head.traj * head.shape[-ndim:]

    return data, head
