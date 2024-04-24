"""Test sense linear operator."""

import itertools
import pytest

import numpy.testing as npt

import torch
import deepmr

ndim = [2, 3]
nslices = [1, 32]
dtype = [torch.float32, torch.float64, torch.complex64, torch.complex128]
device = ["cpu"]
if torch.cuda.is_available():
    device += ["cuda"]

# tolerance
tol = 1e-4


@pytest.mark.parametrize("ndim, nslices, dtype, device", list(itertools.product(*[ndim, nslices, dtype, device])))
def test_single_contrast(ndim, nslices, dtype, device):
    
    # generate data
    combined_truth, img_truth, smap = _generate_data(ndim, nslices, False)
    
    # build operator
    S = deepmr.linops.SenseOp(smap)
    
    # actual calculation
    img = S * combined_truth
    combined = S.H * img_truth
    
    # check
    npt.assert_allclose(img, img_truth, rtol=tol, atol=tol)
    npt.assert_allclose(combined, combined_truth, rtol=tol, atol=tol)
    

@pytest.mark.parametrize("ndim, nslices, dtype, device", list(itertools.product(*[ndim, nslices, dtype, device])))
def test_multiple_contrast(ndim, nslices, dtype, device):
    
    # generate data
    combined_truth, img_truth, smap = _generate_data(ndim, nslices, True)
    
    # build operator
    S = deepmr.linops.SenseOp(smap.unsqueeze(-ndim-1))
    
    # actual calculation
    img = S * combined_truth
    combined = S.H * img_truth
    
    # check
    npt.assert_allclose(img, img_truth, rtol=tol, atol=tol)
    npt.assert_allclose(combined, combined_truth, rtol=tol, atol=tol)
    
    
# %% utils
def _generate_data(ndim, nslices, multicontrast):
    
    # generate data
    combined = deepmr.shepp_logan(32, nslices=nslices)
    
    # generate coils
    smap = deepmr.sensmap([4, nslices, 32, 32])

    # apply maps
    img = smap * combined
    
    # for 2D multislice, permute channel and nz axes
    if ndim == 2:
        combined = combined.swapaxes(0, 1)
        img = img.swapaxes(0, 1)
        smap = smap.swapaxes(0, 1)
        
    if multicontrast:
        combined = combined.unsqueeze(-ndim-1)
        img = img.unsqueeze(-ndim-1)
        
    return combined, img, smap
    
    
    
