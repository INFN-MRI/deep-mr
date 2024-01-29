"""Test resize/resampling functions."""

import numpy.testing as npt

import torch
import deepmr

def test_zeropad():
        # Zero-pad
        x = torch.tensor([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])
        oshape = [5, 5]
        y = deepmr.resize(x, oshape)
        npt.assert_allclose(y, [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])

        x = torch.tensor([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])
        oshape = [4, 4]
        y = deepmr.resize(x, oshape)
        npt.assert_allclose(y, [[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0]])
        
def test_zeropad_batch():
        # Zero-pad
        x = torch.tensor([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])
        oshape = [5, 5]
        y = deepmr.resize(x[None, ...], oshape)
        npt.assert_allclose(y[0], [[0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]])

        x = torch.tensor([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])
        oshape = [4, 4]
        y = deepmr.resize(x[None, ...], oshape)
        npt.assert_allclose(y[0], [[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 0]])

def test_crop():
        # Crop
        x = torch.tensor([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
        oshape = [3, 3]
        y = deepmr.resize(x, oshape)
        npt.assert_allclose(y, [[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

        x = torch.tensor([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0]])
        oshape = [3, 3]
        y = deepmr.resize(x, oshape)
        npt.assert_allclose(y, [[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
        
def test_crop_batch():
        # Crop
        x = torch.tensor([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
        oshape = [3, 3]
        y = deepmr.resize(x[None, ...], oshape)
        npt.assert_allclose(y[0], [[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])

        x = torch.tensor([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0]])
        oshape = [3, 3]
        y = deepmr.resize(x[None, ...], oshape)
        npt.assert_allclose(y[0], [[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]])
