"""Test interpolation/gridding functions."""

import itertools
import pytest

import numpy.testing as npt

import torch
import deepmr

# test values
ncoils = [1, 2]
nslices = [1, 2]

device = ["cpu"]
if torch.cuda.is_available():
    device += ["cuda"]
    
# tolerance
tol = 1e-4

@pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[1, 2], ncoils, nslices, device])))
def test_interp1(ncontrasts, ncoils, nslices, device, npix=4, width=12):

    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((nslices, ncoils, 1, npix), dtype=torch.complex64, device=device)
    else:
        kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

    # k-space coordinates
    coord, _ = _generate_coordinates(1, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((nslices, ncoils, npix), dtype=torch.complex64)
    else:
        kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol)

@pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[2, 3], ncoils, nslices, device])))
def test_interp_lowrank1(ncontrasts, ncoils, nslices, device, npix=4, width=12):

    # get ground truth
    kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

    # k-space coordinates
    coord, _ = _generate_coordinates(1, ncontrasts, npix)

    # input
    kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, basis_adjoint=basis_adjoint, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol)
    
@pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[1, 2], ncoils, nslices, device])))
def test_interp2(ncontrasts, ncoils, nslices, device, npix=4, width=12):

    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((nslices, ncoils, 1, npix**2), dtype=torch.complex64, device=device)
    else:
        kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # k-space coordinates
    coord, _ = _generate_coordinates(2, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((nslices, ncoils, npix, npix), dtype=torch.complex64)
    else:
        kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol)

@pytest.mark.parametrize("ncontrasts, ncoils, nslices, device", list(itertools.product(*[[2, 3], ncoils, nslices, device])))
def test_interp_lowrank2(ncontrasts, ncoils, nslices, device, npix=4, width=12):

    # get ground truth
    kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # k-space coordinates
    coord, _ = _generate_coordinates(2, ncontrasts, npix)

    # input
    kdata_in = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, basis_adjoint=basis_adjoint, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol)

@pytest.mark.parametrize("ncontrasts, ncoils, device", list(itertools.product(*[[1, 2], ncoils, device])))
def test_interp3(ncontrasts, ncoils, device, npix=4, width=12):

    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((ncoils, 1, npix**3), dtype=torch.complex64, device=device)
    else:
        kdata_ground_truth = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # k-space coordinates
    coord, _ = _generate_coordinates(3, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((ncoils, npix, npix, npix), dtype=torch.complex64)
    else:
        kdata_in = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol)

@pytest.mark.parametrize("ncontrasts, ncoils, device", list(itertools.product(*[[2, 3], ncoils, device])))
def test_interp_lowrank3(ncontrasts, ncoils, device, npix=32, width=8):

    # get ground truth
    kdata_ground_truth = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # k-space coordinates
    coord, _ = _generate_coordinates(3, ncontrasts, npix)

    # input
    kdata_in = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.interpolate(kdata_in.clone(), coord=coord, basis_adjoint=basis_adjoint, device=device, width=width)

    # check
    npt.assert_allclose(kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_gridding1(ncontrasts, ncoils, nslices, device, npix=4, width=12):
    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones((nslices, ncoils, npix), dtype=torch.complex64)
    else:
        kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

    # k-space coordinates
    coord, _ = _generate_coordinates(1, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones((nslices, ncoils, 1, npix), dtype=torch.complex64, device=device)
    else:
        kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

    # computation
    kdata_out = deepmr.gridding(
        kdata_in.clone(), shape=npix, coord=coord, device=device, width=width
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(),
        kdata_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )
    

@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_gridding_lowrank1(ncontrasts, ncoils, nslices, device, npix=4, width=12):
    # get ground truth
    kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)

    # k-space coordinates
    coord, _ = _generate_coordinates(1, ncontrasts, npix)

    # input
    kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device)

    # get basis
    basis = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.gridding(
        kdata_in.clone(),
        shape=npix,
        coord=coord,
        basis=basis,
        device=device,
        width=width,
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(),
        kdata_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )

    
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
    coord, _ = _generate_coordinates(2, ncontrasts, npix)

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
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_gridding_lowrank2(ncontrasts, ncoils, nslices, device, npix=4, width=12):
    # get ground truth
    kdata_ground_truth = torch.ones((nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64)

    # k-space coordinates
    coord, _ = _generate_coordinates(2, ncontrasts, npix)

    # input
    kdata_in = torch.ones((nslices, ncoils, ncontrasts, 1, npix**2), dtype=torch.complex64, device=device)

    # get basis
    basis = torch.eye(ncontrasts, dtype=torch.complex64)

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
        rtol=tol,
        atol=tol,
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
    coord, _ = _generate_coordinates(3, ncontrasts, npix)

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
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device",
    list(itertools.product(*[[2, 3], ncoils, device])),
)
def test_gridding_lowrank3(ncontrasts, ncoils, device, npix=4, width=12):
    # get ground truth
    kdata_ground_truth = torch.ones((ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64)

    # k-space coordinates
    coord, _ = _generate_coordinates(3, ncontrasts, npix)

    # input
    kdata_in = torch.ones((ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device)

    # get basis
    basis = torch.eye(ncontrasts, dtype=torch.complex64)

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
        rtol=tol,
        atol=tol,
    )

# %% local subroutines
def _generate_coordinates(ndim, ncontrasts, npix):

    # data type
    dtype = torch.float32

    # build coordinates
    nodes = torch.arange(npix) - (npix // 2)

    if ndim == 1:
        coord = nodes[..., None]
    elif ndim == 2:
        x_i, y_i = torch.meshgrid(nodes, nodes, indexing="ij")
        x_i = x_i.flatten()
        y_i = y_i.flatten()
        coord = torch.stack((x_i, y_i), axis=-1).to(dtype)
    elif ndim == 3:
        x_i, y_i, z_i = torch.meshgrid(nodes, nodes, nodes, indexing="ij")
        x_i = x_i.flatten()
        y_i = y_i.flatten()
        z_i = z_i.flatten()
        coord = torch.stack((x_i, y_i, z_i), axis=-1).to(dtype)

    # assume single shot trajectory
    coord = coord[None, ...]  # (nview=1, nsamples=npix**ndim, ndim=ndim)
    if ncontrasts > 1:
        coord = torch.repeat_interleave(coord[None, ...], ncontrasts, axis=0)
        
    # normalize
    coord = coord / npix

    # build dcf
    dcf = torch.ones(coord.shape[:-1], dtype=dtype)
    
    return coord, dcf