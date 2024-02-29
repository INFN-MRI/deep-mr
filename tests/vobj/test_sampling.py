"""Test sampling pattern generations."""

import numpy.testing as npt

import deepmr


def test_cartesian2d():
    # squared
    mask, head = deepmr.cartesian2D(128, accel=2)
    npt.assert_allclose(mask.shape, [128, 128])

    # rectangular
    mask, head = deepmr.cartesian2D((128, 96), accel=2)
    npt.assert_allclose(mask.shape, [96, 128])


def test_cartesian3d():
    # squared
    mask, head = deepmr.cartesian3D(128, accel=2)
    npt.assert_allclose(mask.shape, [1, 128, 128])

    # rectangular
    mask, head = deepmr.cartesian3D((128, 96), accel=2)
    npt.assert_allclose(mask.shape, [1, 96, 128])

    # squared multicontrast
    mask, head = deepmr.cartesian3D((128, 128, 32), accel_type="CS", accel=4)
    npt.assert_allclose(mask.shape, [32, 128, 128])  # (contrast, z, y)

    # rectangular multicontrast
    mask, head = deepmr.cartesian3D((128, 96, 32), accel_type="CS", accel=4)
    npt.assert_allclose(mask.shape, [32, 96, 128])  # (contrast, z, y)


def test_spiral():
    # single shot
    head = deepmr.spiral(128)
    npt.assert_allclose(head.traj.shape, [1, 1, 25856, 2])

    # multi shot
    head = deepmr.spiral(128, nintl=48)
    npt.assert_allclose(head.traj.shape, [1, 48, 538, 2])

    # multi shot accelerated
    head = deepmr.spiral(128, nintl=48, accel=4)
    npt.assert_allclose(head.traj.shape, [1, 12, 538, 2])

    # multi contrast
    head = deepmr.spiral((128, 420), nintl=48)
    npt.assert_allclose(head.traj.shape, [420, 1, 538, 2])

    # multi echo
    head = deepmr.spiral((128, 1, 8), nintl=48)
    npt.assert_allclose(head.traj.shape, [8, 48, 538, 2])


def test_spiral_stack():
    # single shot
    head = deepmr.spiral_stack((128, 120))
    npt.assert_allclose(head.traj.shape, [1, 120, 25856, 3])

    # multi shot
    head = deepmr.spiral_stack((128, 120), nintl=48)
    npt.assert_allclose(head.traj.shape, [1, 5760, 538, 3])

    # multi shot accelerated
    head = deepmr.spiral_stack((128, 120), nintl=48, accel=4)
    npt.assert_allclose(head.traj.shape, [1, 1440, 538, 3])

    head = deepmr.spiral_stack((128, 120), nintl=48, accel=(4, 2))
    npt.assert_allclose(head.traj.shape, [1, 720, 538, 3])

    # multi contrast
    head = deepmr.spiral_stack((128, 120, 420), nintl=48)
    npt.assert_allclose(head.traj.shape, [420, 120, 538, 3])

    # multi echo
    head = deepmr.spiral_stack((128, 120, 1, 8), nintl=48)
    npt.assert_allclose(head.traj.shape, [8, 5760, 538, 3])


def test_spiral_proj():
    # single shot
    head = deepmr.spiral_proj(128)
    npt.assert_allclose(head.traj.shape, [1, 402, 25856, 3])

    # multi shot
    head = deepmr.spiral_proj(128, nintl=48)
    npt.assert_allclose(head.traj.shape, [1, 19296, 538, 3])

    # multi shot accelerated
    head = deepmr.spiral_proj(128, nintl=48, accel=(1, 4))
    npt.assert_allclose(head.traj.shape, [1, 4800, 538, 3])

    head = deepmr.spiral_proj(128, nintl=48, accel=(4, 1))
    npt.assert_allclose(head.traj.shape, [1, 4824, 538, 3])

    # multi contrast
    head = deepmr.spiral_proj((128, 420), nintl=48)
    npt.assert_allclose(head.traj.shape, [420, 48, 538, 3])

    # multi echo
    head = deepmr.spiral_proj((128, 1, 8), nintl=48)
    npt.assert_allclose(head.traj.shape, [8, 19296, 538, 3])


def test_radial():
    # nyquist sampled
    head = deepmr.radial(128)
    npt.assert_allclose(head.traj.shape, [1, 402, 128, 2])

    # accelerated
    head = deepmr.radial(128, nviews=64)
    npt.assert_allclose(head.traj.shape, [1, 64, 128, 2])

    # multi contrast
    head = deepmr.radial((128, 420))
    npt.assert_allclose(head.traj.shape, [420, 1, 128, 2])

    # multi echo
    head = deepmr.radial((128, 1, 8))
    npt.assert_allclose(head.traj.shape, [8, 402, 128, 2])


def test_radial_stack():
    # nyquist sampled
    head = deepmr.radial_stack((128, 120))
    npt.assert_allclose(head.traj.shape, [1, 48240, 128, 3])

    # accelerated
    head = deepmr.radial_stack((128, 120), nviews=64)
    npt.assert_allclose(head.traj.shape, [1, 7680, 128, 3])

    head = deepmr.radial_stack((128, 120), accel=2)
    npt.assert_allclose(head.traj.shape, [1, 24120, 128, 3])

    # multi contrast
    head = deepmr.radial_stack((128, 120, 420))
    npt.assert_allclose(head.traj.shape, [420, 120, 128, 3])

    # multi echo
    head = deepmr.radial_stack((128, 120, 1, 8))
    npt.assert_allclose(head.traj.shape, [8, 48240, 128, 3])


def test_radial_proj():
    # nyquist sampled
    head = deepmr.radial_proj(128)
    npt.assert_allclose(head.traj.shape, [1, 161604, 128, 3])

    # accelerated
    head = deepmr.radial_proj(128, nviews=64)  # radial undersampling
    npt.assert_allclose(head.traj.shape, [1, 25728, 128, 3])

    head = deepmr.radial_proj(128, nviews=(64, 402))  # in-plane undersampling
    npt.assert_allclose(head.traj.shape, [1, 25728, 128, 3])

    # multi contrast
    head = deepmr.radial_proj((128, 420))
    npt.assert_allclose(head.traj.shape, [420, 402, 128, 3])

    # multi echo
    head = deepmr.radial_proj((128, 1, 8))
    npt.assert_allclose(head.traj.shape, [8, 161604, 128, 3])


def test_rosette():
    # nyquist sampled
    head = deepmr.rosette(128)
    npt.assert_allclose(head.traj.shape, [1, 402, 128, 2])

    # accelerated
    head = deepmr.rosette(128, nviews=64)
    npt.assert_allclose(head.traj.shape, [1, 64, 128, 2])

    # multi contrast
    head = deepmr.rosette((128, 420))
    npt.assert_allclose(head.traj.shape, [420, 1, 128, 2])

    # multi echo
    head = deepmr.rosette((128, 1, 8))
    npt.assert_allclose(head.traj.shape, [8, 402, 128, 2])


def test_rosette_stack():
    # nyquist sampled
    head = deepmr.rosette_stack((128, 120))
    npt.assert_allclose(head.traj.shape, [1, 48240, 128, 3])

    # accelerated
    head = deepmr.rosette_stack((128, 120), nviews=64)
    npt.assert_allclose(head.traj.shape, [1, 7680, 128, 3])

    head = deepmr.rosette_stack((128, 120), accel=2)
    npt.assert_allclose(head.traj.shape, [1, 24120, 128, 3])

    # multi contrast
    head = deepmr.rosette_stack((128, 120, 420))
    npt.assert_allclose(head.traj.shape, [420, 120, 128, 3])

    # multi echo
    head = deepmr.rosette_stack((128, 120, 1, 8))
    npt.assert_allclose(head.traj.shape, [8, 48240, 128, 3])


def test_rosette_proj():
    # nyquist sampled
    head = deepmr.rosette_proj(128)
    npt.assert_allclose(head.traj.shape, [1, 161604, 128, 3])

    # accelerated
    head = deepmr.rosette_proj(128, nviews=64)  # radial undersampling
    npt.assert_allclose(head.traj.shape, [1, 25728, 128, 3])

    head = deepmr.rosette_proj(128, nviews=(64, 402))  # in-plane undersampling
    npt.assert_allclose(head.traj.shape, [1, 25728, 128, 3])

    # multi contrast
    head = deepmr.rosette_proj((128, 420))
    npt.assert_allclose(head.traj.shape, [420, 402, 128, 3])

    # multi echo
    head = deepmr.rosette_proj((128, 1, 8))
    npt.assert_allclose(head.traj.shape, [8, 161604, 128, 3])
