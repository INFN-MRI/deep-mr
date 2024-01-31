"""Test filtering functions."""

import numpy.testing as npt

import torch
import deepmr


def test_filter():
    filt1d = deepmr.fermi(1, 128)
    filt2d = deepmr.fermi(2, 128)
    filt3d = deepmr.fermi(3, 128)

    npt.assert_allclose(filt1d.shape, [128])
    assert (filt1d >= 0.5).sum() == 128
    npt.assert_allclose(filt2d.shape, [128, 128])
    npt.assert_allclose(filt3d.shape, [128, 128, 128])

    # reduced width
    filt1d = deepmr.fermi(1, 128, 32)
    npt.assert_allclose(filt1d.shape, [128])
    assert (filt1d >= 0.5).sum() == 47
