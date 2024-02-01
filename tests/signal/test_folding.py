"""Test folding/unfolding functions."""

import numpy.testing as npt

import torch
import deepmr

def test_nonoverlapping():
    image = torch.arange(16).reshape(4, 4)
    
    # folding
    patches = deepmr.tensor2patches(image, [2, 2])
    npt.assert_allclose(patches.shape, [2, 2, 2, 2])
    
    # unfolding
    output = deepmr.patches2tensor(patches, [4, 4], [2, 2])
    npt.assert_allclose(output, image)
    
    
def test_batched():
    image = torch.arange(128).reshape(8, 4, 4)
    
    # folding
    patches = deepmr.tensor2patches(image, [2, 2])
    npt.assert_allclose(patches.shape, [8, 2, 2, 2, 2])
    
    # unfolding
    output = deepmr.patches2tensor(patches, [4, 4], [2, 2])
    npt.assert_allclose(output, image)


def test_rectangular():
    image = torch.arange(24).reshape(6, 4)
    
    # folding
    patches = deepmr.tensor2patches(image, [3, 2])
    npt.assert_allclose(patches.shape, [2, 2, 3, 2])
    
    # unfolding
    output = deepmr.patches2tensor(patches, [6, 4], [3, 2])
    npt.assert_allclose(output, image)


def test_overlapping():
    image = torch.arange(16).reshape(4, 4)
    
    # folding
    patches = deepmr.tensor2patches(image, [2, 2], [1, 1])
    npt.assert_allclose(patches.shape, [3, 3, 2, 2])
    
    # unfolding
    output = deepmr.patches2tensor(patches, [4, 4], [2, 2], [1, 1])
    npt.assert_allclose(output, image)

def test_pad():
    image = torch.arange(25).reshape(5, 5)
    
    # folding
    patches = deepmr.tensor2patches(image, [2, 2])
    npt.assert_allclose(patches.shape, [3, 3, 2, 2])
    
    # unfolding
    output = deepmr.patches2tensor(patches, [5, 5], [2, 2])
    npt.assert_allclose(output, image)

def test_3D():
    image = torch.arange(64).reshape(4, 4, 4)
    
    # folding
    patches = deepmr.tensor2patches(image, [2, 2, 2])
    npt.assert_allclose(patches.shape, [2, 2, 2, 2, 2, 2])
    
    # unfolding
    output = deepmr.patches2tensor(patches, [4, 4, 4], [2, 2, 2])
    npt.assert_allclose(output, image)
