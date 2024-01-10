# -*- coding: utf-8 -*-
"""Unit tests for autograd of NUFFT, NUFFT adjoint and NUFFT self-adjoint routines, including autograd step."""
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument


import itertools
import pytest


import torch


from nufftorch import nn


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
    
    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # set requires_grad flag
    img_ground_truth.requires_grad = True
    kdata_ground_truth.requires_grad = True
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    shape = wave.acquisition_matrix
      
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
     
    # simple case of ||A(x)||**2
    kdata_out = F(img_ground_truth)
    loss = (kdata_out.abs() ** 2 / 2).sum().backward()
    
    # get autograd from AD and manually
    autograd_nufft = img_ground_truth.grad.clone()
    autograd_nufft_ground_truth = FH(kdata_out.detach())
    
    if test:
        assert torch.allclose(autograd_nufft, autograd_nufft_ground_truth)
    
    return autograd_nufft.squeeze().detach()


@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, device])))
def test_nufft_adjoint(ndim, nframes, nechoes, ncoils, nslices, device, 
                       npix=8, osf=2.0, width=4, test=True):
    
    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # set requires_grad flag
    img_ground_truth.requires_grad = True
    kdata_ground_truth.requires_grad = True
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(kdata_ground_truth.device)
    shape = wave.acquisition_matrix
          
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, oversamp=osf, width=width)
     
    # simple case of ||A(x)||**2
    img_out = FH(dcf * kdata_ground_truth)
    loss = (img_out.abs() ** 2 / 2).sum().backward()
    
    # get autograd from AD and manually
    autograd_nufft_adjoint = kdata_ground_truth.grad.clone()
    autograd_nufft_adjoint_ground_truth = F(img_out.detach())
    
    if test:
        assert torch.allclose(autograd_nufft_adjoint, autograd_nufft_adjoint_ground_truth)
    
    return autograd_nufft_adjoint.squeeze().detach()
 

@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
def test_nufft_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, 
                       npix=8, osf=2.0, width=4, test=True):
    
    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # set requires_grad flag
    img_ground_truth.requires_grad = True
    kdata_ground_truth.requires_grad = True
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    shape = wave.acquisition_matrix
    
    # get basis
    basis = _lowrank_subspace_projection(dtype, nframes)
      
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device, oversamp=osf, width=width, basis=basis)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, oversamp=osf, width=width, basis=basis)
     
    # simple case of ||A(x)||**2
    kdata_out = F(img_ground_truth)
    loss = (kdata_out.abs() ** 2 / 2).sum().backward()
    
    # get autograd from AD and manually
    autograd_nufft = img_ground_truth.grad.clone()
    autograd_nufft_ground_truth = FH(kdata_out.detach())
    
    if test:
        assert torch.allclose(autograd_nufft, autograd_nufft_ground_truth)
    
    return autograd_nufft.squeeze().detach()

    

@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
def test_nufft_adjoint_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, 
                               npix=8, osf=2.0, width=4, test=True):
        
    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    kdata_ground_truth = _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # set requires_grad flag
    img_ground_truth.requires_grad = True
    kdata_ground_truth.requires_grad = True
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(kdata_ground_truth.device)
    shape = wave.acquisition_matrix
    
    # get basis
    basis = _lowrank_subspace_projection(dtype, nframes)
      
    # prepare operators
    F = nn.NUFFT(coord=coord, shape=shape, device=device, oversamp=osf, width=width, basis=basis)
    FH = nn.NUFFTAdjoint(coord=coord, shape=shape, device=device, oversamp=osf, width=width, basis=basis)
     
    # simple case of ||A(x)||**2
    img_out = FH(dcf * kdata_ground_truth)
    loss = (img_out.abs() ** 2 / 2).sum().backward()
    
    # get autograd from AD and manually
    autograd_nufft_adjoint = kdata_ground_truth.grad.clone()
    autograd_nufft_adjoint_ground_truth = F(img_out.detach())
    
    if test:
        assert torch.allclose(autograd_nufft_adjoint, autograd_nufft_adjoint_ground_truth)
    
    return autograd_nufft_adjoint.squeeze().detach()

    
@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, device])))
def test_nufft_selfadjoint(ndim, nframes, nechoes, ncoils, nslices, device, 
                            npix=32, osf=2.0, width=4, test=True):
    
    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # set requires_grad flag
    img_ground_truth.requires_grad = True
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(img_ground_truth.device)
    shape = wave.acquisition_matrix
          
    # prepare operators
    FHF = nn.NUFFTSelfadjoint(coord=coord, shape=shape, device=device, dcf=dcf, prep_oversamp=osf, width=width)
     
    # simple case of ||A(x)||**2
    img_out = FHF(img_ground_truth)
    loss = (img_out.abs() ** 2 / 2).sum().backward()
    
    # get autograd from AD and manually
    autograd_nufft_selfadjoint = img_ground_truth.grad.clone()
    autograd_nufft_selfadjoint_ground_truth = FHF(img_out.detach())
    
    if test:
        assert torch.allclose(autograd_nufft_selfadjoint, autograd_nufft_selfadjoint_ground_truth)
    
    return autograd_nufft_selfadjoint.squeeze().detach(), autograd_nufft_selfadjoint_ground_truth.squeeze().detach()


@pytest.mark.parametrize("ndim, nframes, nechoes, ncoils, nslices, dtype, device",
    list(itertools.product(*[ndim, nframes, nechoes, ncoils, nslices, dtype, device])))
def test_nufft_selfadjoint_lowrank(ndim, nframes, nechoes, ncoils, nslices, dtype, device, 
                                    npix=32, osf=2.0, width=4, test=True):
    
    # get ground truth
    img_ground_truth = _image(ndim, nframes, nechoes, ncoils, nslices, npix, device)
    
    # set requires_grad flag
    img_ground_truth.requires_grad = True
    
    # k-space coordinates
    wave = _kt_space_trajectory(ndim, nframes, npix)
    coord = wave.coordinates
    dcf = wave.density_comp_factor.to(img_ground_truth.device)
    shape = wave.acquisition_matrix
    
    # get basis
    basis = _lowrank_subspace_projection(dtype, nframes)
          
    # prepare operators
    FHF = nn.NUFFTSelfadjoint(coord=coord, shape=shape, device=device, dcf=dcf, basis=basis, prep_oversamp=osf, width=width)
     
    # simple case of ||A(x)||**2
    img_out = FHF(img_ground_truth)
    loss = (img_out.abs() ** 2 / 2).sum().backward()
    
    # get autograd from AD and manually
    autograd_nufft_selfadjoint = img_ground_truth.grad.clone()
    autograd_nufft_selfadjoint_ground_truth = FHF(img_out.detach())
    
    if test:
        assert torch.allclose(autograd_nufft_selfadjoint, autograd_nufft_selfadjoint_ground_truth)
    
    return autograd_nufft_selfadjoint.squeeze().detach(), autograd_nufft_selfadjoint_ground_truth.squeeze().detach()
 