"""Test interpolation/gridding functions."""

import itertools
import pytest

import numpy.testing as npt

import torch
import deepmr

from conftest import _kt_space_trajectory
from conftest import _lowrank_subspace_projection

# test values
ncoils = [1, 2]
nslices = [1, 2]

device = ["cpu"]
if torch.cuda.is_available():
    device += ["cuda"]

# @pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[1, 2], ncoils, nslices, device])))
# def test_interp1(ncontrasts, ncoils, nslices, device, npix=4, width=12):

#     # get ground truth
#     if ncontrasts == 1:
#         kdata_ground_truth = torch.ones((nslices, ncoils, 1, npix), dtype=torch.complex64, device=device)
#     else:
#         kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

#     # k-space coordinates
#     wave = _kt_space_trajectory(1, ncontrasts, npix)
#     coord = wave.coordinates

#     # input
#     if ncontrasts == 1:
#         kdata_in = torch.ones((nslices, ncoils, npix), dtype=torch.complex64)
#     else:
#         kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

#     # computation
#     kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, device=device, width=width)

#     # check
#     npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=0.01, atol=0.01)

# @pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[2, 3], ncoils, nslices, device])))
# def test_interp_lowrank1(ncontrasts, ncoils, nslices, device, npix=4, width=12):

#     # get ground truth
#     kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

#     # k-space coordinates
#     wave = _kt_space_trajectory(1, ncontrasts, npix)
#     coord = wave.coordinates

#     # input
#     kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

#     # get basis
#     basis_adjoint = _lowrank_subspace_projection(torch.complex64, ncontrasts)

#     # computation
#     kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, basis_adjoint=basis_adjoint, device=device, width=width)

#     # check
#     npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=0.01, atol=0.01)
    
@pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[1, 2], ncoils, nslices, device])))
def test_interp2(ncontrasts, ncoils, nslices, device, npix=4, width=12):

    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((nslices, ncoils, 1, npix**2), dtype=torch.complex64, device=device)
    else:
        kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # k-space coordinates
    wave = _kt_space_trajectory(2, ncontrasts, npix)
    coord = wave.coordinates

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((nslices, ncoils, npix, npix), dtype=torch.complex64)
    else:
        kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=0.01, atol=0.01)

@pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[2, 3], ncoils, nslices, device])))
def test_interp_lowrank2(ncontrasts, ncoils, nslices, device, npix=4, width=12):

    # get ground truth
    kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # k-space coordinates
    wave = _kt_space_trajectory(2, ncontrasts, npix)
    coord = wave.coordinates

    # input
    kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # get basis
    basis_adjoint = _lowrank_subspace_projection(torch.complex64, ncontrasts)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, basis_adjoint=basis_adjoint, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=0.01, atol=0.01)

@pytest.mark.parametrize("ncontrasts, ncoils, device", list(itertools.product(*[[1, 2], ncoils, device])))
def test_interp3(ncontrasts, ncoils, device, npix=4, width=12):

    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((ncoils, 1, npix**3), dtype=torch.complex64, device=device)
    else:
        kdata_ground_truth = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # k-space coordinates
    wave = _kt_space_trajectory(3, ncontrasts, npix)
    coord = wave.coordinates

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((ncoils, npix, npix, npix), dtype=torch.complex64)
    else:
        kdata_in = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=0.01, atol=0.01)

@pytest.mark.parametrize("ncontrasts, ncoils, device", list(itertools.product(*[[2, 3], ncoils, device])))
def test_interp_lowrank3(ncontrasts, ncoils, device, npix=32, width=8):

    # get ground truth
    kdata_ground_truth = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # k-space coordinates
    wave = _kt_space_trajectory(3, ncontrasts, npix)
    coord = wave.coordinates

    # input
    kdata_in = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # get basis
    basis_adjoint = _lowrank_subspace_projection(torch.complex64, ncontrasts)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, basis_adjoint=basis_adjoint, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=0.01, atol=0.01)


# @pytest.mark.parametrize(
#     "ncontrasts, ncoils, nslices, device",
#     list(itertools.product(*[[1, 2], ncoils, nslices, device])),
# )
# def test_gridding1(ncontrasts, ncoils, nslices, device, npix=4, width=12):
#     # get ground truth
#     if ncontrasts == 1:
#         kdata_ground_truth = torch.ones((nslices, ncoils, npix), dtype=torch.complex64)
#     else:
#         kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

#     # k-space coordinates
#     wave = _kt_space_trajectory(1, ncontrasts, npix)
#     coord = wave.coordinates

#     # input
#     if ncontrasts == 1:
#         kdata_in = torch.ones((nslices, ncoils, 1, npix), dtype=torch.complex64, device=device)
#     else:
#         kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

#     # computation
#     kdata_out = deepmr.gridding(
#         kdata_in.clone(), shape=npix, coord=coord, device=device, width=width
#     )

#     # check
#     npt.assert_allclose(
#         kdata_out.detach().cpu(),
#         kdata_ground_truth.detach().cpu(),
#         rtol=0.01,
#         atol=0.01,
#     )
    

# @pytest.mark.parametrize(
#     "ncontrasts, ncoils, nslices, device",
#     list(itertools.product(*[[2, 3], ncoils, nslices, device])),
# )
# def test_gridding_lowrank1(ncontrasts, ncoils, nslices, device, npix=4, width=12):
#     # get ground truth
#     kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

#     # k-space coordinates
#     wave = _kt_space_trajectory(1, ncontrasts, npix)
#     coord = wave.coordinates

#     # input
#     kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

#     # get basis
#     basis = _lowrank_subspace_projection(torch.complex64, ncontrasts)

#     # computation
#     kdata_out = deepmr.gridding(
#         kdata_in.clone(),
#         shape=npix,
#         coord=coord,
#         basis=basis,
#         device=device,
#         width=width,
#     )

#     # check
#     npt.assert_allclose(
#         kdata_out.detach().cpu(),
#         kdata_ground_truth.detach().cpu(),
#         rtol=0.01,
#         atol=0.01,
#     )

    
@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_gridding2(ncontrasts, ncoils, nslices, device, npix=4, width=12):
    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((nslices, ncoils, npix, npix), dtype=torch.complex64)
    else:
        kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # k-space coordinates
    wave = _kt_space_trajectory(2, ncontrasts, npix)
    coord = wave.coordinates

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((nslices, ncoils, 1, npix**2), dtype=torch.complex64, device=device)
    else:
        kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # computation
    kdata_out = deepmr.gridding(
        kdata_in.clone(), shape=2 * [npix], coord=coord, device=device, width=width
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(),
        kdata_ground_truth.detach().cpu(),
        rtol=0.01,
        atol=0.01,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_gridding_lowrank2(ncontrasts, ncoils, nslices, device, npix=4, width=12):
    # get ground truth
    kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # k-space coordinates
    wave = _kt_space_trajectory(2, ncontrasts, npix)
    coord = wave.coordinates

    # input
    kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # get basis
    basis = _lowrank_subspace_projection(torch.complex64, ncontrasts)

    # computation
    kdata_out = deepmr.gridding(
        kdata_in.clone(),
        shape=2 * [npix],
        coord=coord,
        basis=basis,
        device=device,
        width=width,
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(),
        kdata_ground_truth.detach().cpu(),
        rtol=0.01,
        atol=0.01,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device",
    list(itertools.product(*[[1, 2], ncoils, device])),
)
def test_gridding3(ncontrasts, ncoils, device, npix=4, width=12):
    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((ncoils, npix, npix, npix), dtype=torch.complex64)
    else:
        kdata_ground_truth = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # k-space coordinates
    wave = _kt_space_trajectory(3, ncontrasts, npix)
    coord = wave.coordinates

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((ncoils, 1, npix**3), dtype=torch.complex64, device=device)
    else:
        kdata_in = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # computation
    kdata_out = deepmr.gridding(
        kdata_in.clone(), shape=3 * [npix], coord=coord, device=device, width=width
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(),
        kdata_ground_truth.detach().cpu(),
        rtol=0.01,
        atol=0.01,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device",
    list(itertools.product(*[[2, 3], ncoils, device])),
)
def test_gridding_lowrank3(ncontrasts, ncoils, device, npix=4, width=12):
    # get ground truth
    kdata_ground_truth = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # k-space coordinates
    wave = _kt_space_trajectory(3, ncontrasts, npix)
    coord = wave.coordinates

    # input
    kdata_in = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # get basis
    basis = _lowrank_subspace_projection(torch.complex64, ncontrasts)

    # computation
    kdata_out = deepmr.gridding(
        kdata_in.clone(),
        shape=3 * [npix],
        coord=coord,
        basis=basis,
        device=device,
        width=width,
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(),
        kdata_ground_truth.detach().cpu(),
        rtol=0.01,
        atol=0.01,
    )

