# -*- coding: utf-8 -*-
"""Unit tests for procedural versions of NUFFT and NUFFT adjoint routines."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument


import itertools
import pytest


import torch


from nufftorch import functional


from conftest import _kt_space_data, _kt_space_trajectory, _image, _lowrank_subspace_projection


# test values
ndim = [2, 3]
device = ['cpu']
dtype = [torch.float32, torch.complex64]

nframes = [1, 2]
nechoes = [1, 2]
ncoils = [1, 2]
nslices = [1, 2]

if torch.cuda.is_available():
    device += ['cuda']


@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, device])))
def test_nufft(ndim, nframes, nechoes, ncoils, nslices, device, 
                npix=8, osf=2.0, width=4, test=True):
    
    tol = 1e-2  

    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)

    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
            
    # computation
    kdata_out = functional.nufft(img_ground_truth.clone(), coord=coord, 
                                  device=device, oversamp=osf, width=width)
    
    # check
    res = torch.norm(kdata_ground_truth - kdata_out) / torch.norm(kdata_ground_truth)

    if test:
        assert res < tol
    
    return kdata_out.squeeze().detach()


@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, device])))
def test_nufft_adjoint(ndim, nframes, nechoes, ncoils, nslices, device, 
                        npix=8, osf=2.0, width=4, test=True):
    
    tol = 1e-2  

    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)

    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(kdata_ground_truth.device)
    shape = wave.acquisition_matrix
            
    # computation
    img_out = functional.nufft_adjoint(dcf * kdata_ground_truth.clone(), coord=coord, shape=shape, 
                                        device=device, oversamp=osf, width=width)
    
    # check
    res = torch.norm(img_ground_truth - img_out) / torch.norm(img_ground_truth)

    if test:
        assert res < tol
    
    return img_out.squeeze().detach()
 

@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
def test_nufft_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, 
                        npix=8, osf=2.0, width=4, test=True):
    
    tol = 1e-2  

    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)

    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    
    # get basis
    basis = _lowrank_subspace_projection(dtype, nframes)
            
    # computation
    kdata_out = functional.nufft(img_ground_truth.clone(), coord=coord, 
                                  device=device, basis=basis, oversamp=osf, width=width)
    
    # check
    res = torch.norm(kdata_ground_truth - kdata_out) / torch.norm(kdata_ground_truth)

    if test:
        assert res < tol
    
    return kdata_out.squeeze().detach()
    

@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
def test_nufft_adjoint_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, 
                        npix=8, osf=2.0, width=4, test=True):
        
    tol = 1e-2  

    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)

    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(kdata_ground_truth.device)
    shape = wave.acquisition_matrix
    
    # get basis
    basis = _lowrank_subspace_projection(dtype, nframes)
            
    # computation
    img_out = functional.nufft_adjoint(dcf * kdata_ground_truth.clone(), coord=coord, shape=shape, 
                                        device=device, oversamp=osf, width=width, basis=basis)
    
    # check
    res = torch.norm(img_ground_truth - img_out) / torch.norm(img_ground_truth)

    if test:
        assert res < tol
    
    return img_out.squeeze().detach()
