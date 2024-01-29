"""Test phantom generations."""

import numpy.testing as npt

import torch
import deepmr

def test_shepp_logan():
    # single slice
    phantom = deepmr.shepp_logan(128)
    npt.assert_allclose(phantom.shape, [128, 128])
    
    # multislice
    phantom = deepmr.shepp_logan(128, 32)
    npt.assert_allclose(phantom.shape, [32, 128, 128])
    
    # isotropic 3D
    phantom = deepmr.shepp_logan(128, -1)
    npt.assert_allclose(phantom.shape, [128, 128, 128])

def test_qmr_shepp_logan():
    phantom = deepmr.shepp_logan(128, qmr=True)
    assert list(phantom.keys()) == ["M0", "T1", "T2", "T2star", "chi"]
    npt.assert_allclose(phantom["M0"].shape, [128, 128])
    npt.assert_allclose(phantom["T1"].shape, [128, 128])
    npt.assert_allclose(phantom["T2"].shape, [128, 128])
    npt.assert_allclose(phantom["T2star"].shape, [128, 128])
    npt.assert_allclose(phantom["chi"].shape, [128, 128])

def test_custom_phantom():
    segmentation = torch.tensor([0, 0, 0, 1, 1, 1], dtype=int)
    properties = {"M0": [0.7, 0.8], "T1": [500.0, 1000.0], "T2": [50.0, 100.0]}
    phantom = deepmr.custom_phantom(segmentation, properties)
    
    npt.assert_allclose(phantom["M0"], [ 0.7, 0.7, 0.7, 0.8, 0.8, 0.8])    
    npt.assert_allclose(phantom["T1"], [ 500., 500., 500., 1000., 1000., 1000.])    
    npt.assert_allclose(phantom["T2"], [ 50., 50., 50., 100., 100., 100.])
    
