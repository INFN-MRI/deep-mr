"""Two-dimensional spiral sampling."""

def spiral(fov, shape, accel=1, nintl=1, **kwargs):
    r"""
    Design a constant- or multi-density spiral.

    Parameters
    ----------
    fov : float
        Field of view in ``[mm]``.
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
    tilt_type : str
        Tilt of the shots. The default is ``uniform`` (see ``Notes``).
    tilt_contrasts : bool
        If ``True``, each contrast have a different sampling pattern.
        In this case, ``tilt_type`` is forced to ``golden``.
        The default is ``False``.
    acs_shape : int
        Matrix size for inner spiral.
        The default is ``None``.
    acs_nintl : int
        Number of interleaves to fully sample inner spiral.
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
    The following values are accepted for the tilt name, with :math:`N` the number of
    partitions:

    * "uniform": uniform tilt: 2:math:`\pi / N`
    * "inverted": inverted tilt :math:`\pi/N + \pi`
    * "golden": golden angle tilt :math:`\pi(\sqrt{5}-1)/2`. For 3D, refers to through plane axis (in-plane is uniform).
    * "tiny-golden": tiny golden angle tilt 2:math:`\pi(15 -\sqrt{5})`. For 3D, refers to through plane axis (in-plane is uniform).
    * "tgas": tiny golden angle tilt with shuffling along through-plane axis (3D only)`

    The returned ``head`` (:func:`deepmr.Header`) is a structure with the following fields:

    * shape (torch.Tensor):
        This is the expected image size of shape ``(nz, ny, nx)``.
    * resolution (torch.Tensor):
        This is the expected image resolution in mm of shape ``(dz, dy, dx)``.
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

