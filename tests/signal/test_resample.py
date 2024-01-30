"""Test resize/resampling functions."""

import numpy.testing as npt

import torch
import deepmr

def test_downsample():
    # 1D
    x = torch.ones(5, dtype=torch.float32)
    oshape = 3
    y = deepmr.resize(x, oshape)
    npt.assert_allclose(y.shape, [3]) 
    
    # 2D
    x = torch.ones([5, 5], dtype=torch.float32)
    oshape = [3, 3]
    y = deepmr.resize(x, oshape)
    npt.assert_allclose(y.shape, [3, 3]) 
    
    # 3D
    x = torch.ones([5, 5, 5], dtype=torch.float32)
    oshape = [3, 3, 3]
    y = deepmr.resize(x, oshape)
    npt.assert_allclose(y.shape, [3, 3, 3]) 
    

def test_upsample():
    # 1D
    x = torch.ones(3, dtype=torch.float32)
    oshape = 5
    y = deepmr.resize(x, oshape)
    npt.assert_allclose(y.shape, [5]) 
    
    # 2D
    x = torch.ones([3, 3], dtype=torch.float32)
    oshape = [5, 5]
    y = deepmr.resize(x, oshape)
    npt.assert_allclose(y.shape, [5, 5]) 
    
    # 3D
    x = torch.ones([3, 3, 3], dtype=torch.float32)
    oshape = [5, 5, 5]
    y = deepmr.resize(x, oshape)
    npt.assert_allclose(y.shape, [5, 5, 5]) 

    

