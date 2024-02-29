"""Test sampling pattern generations."""

import numpy.testing as npt

import deepmr

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
    