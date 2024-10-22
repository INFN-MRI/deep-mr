"""Test sparse FFT / iFFT functions."""

import itertools
import pytest
import warnings


import numpy.testing as npt
from numba.core.errors import NumbaPerformanceWarning

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

# suppress performance warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_sparse_fft1(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones(
            (nslices, ncoils, 1, npix), dtype=torch.complex64, device=device
        )
    else:
        kdata_ground_truth = torch.ones(
            (nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device
        )

    # k-space indexes
    indexes = _generate_indexes(1, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        image_in = torch.zeros((nslices, ncoils, npix), dtype=torch.complex64)
    else:
        image_in = torch.zeros(
            (nslices, ncoils, ncontrasts, npix), dtype=torch.complex64
        )
    image_in[..., npix // 2] = 1.0

    # computation
    kdata_out = deepmr.fft.sparse_fft(image_in.clone(), indexes=indexes, device=device)

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_sparse_fft_lowrank1(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    kdata_ground_truth = torch.ones(
        (nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device
    )

    # k-space indexes
    indexes = _generate_indexes(1, ncontrasts, npix)

    # input
    image_in = torch.zeros((nslices, ncoils, ncontrasts, npix), dtype=torch.complex64)
    image_in[..., npix // 2] = 1.0

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.fft.sparse_fft(
        image_in.clone(), indexes=indexes, basis_adjoint=basis_adjoint, device=device
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_sparse_fft2(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones(
            (nslices, ncoils, 1, npix**2), dtype=torch.complex64, device=device
        )
    else:
        kdata_ground_truth = torch.ones(
            (nslices, ncoils, ncontrasts, 1, npix**2),
            dtype=torch.complex64,
            device=device,
        )

    # k-space indexes
    indexes = _generate_indexes(2, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        image_in = torch.zeros((nslices, ncoils, npix, npix), dtype=torch.complex64)
    else:
        image_in = torch.zeros(
            (nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64
        )
    image_in[..., npix // 2, npix // 2] = 1.0

    # computation
    kdata_out = deepmr.fft.sparse_fft(image_in.clone(), indexes=indexes, device=device)

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_sparse_fft_lowrank2(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    kdata_ground_truth = torch.ones(
        (nslices, ncoils, ncontrasts, 1, npix**2),
        dtype=torch.complex64,
        device=device,
    )

    # k-space indexes
    indexes = _generate_indexes(2, ncontrasts, npix)

    # input
    image_in = torch.zeros(
        (nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64
    )
    image_in[..., npix // 2, npix // 2] = 1.0

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.fft.sparse_fft(
        image_in.clone(), indexes=indexes, basis_adjoint=basis_adjoint, device=device
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device", list(itertools.product(*[[1, 2], ncoils, device]))
)
def test_sparse_fft3(ncontrasts, ncoils, device, npix=4):
    # get ground truth
    if ncontrasts == 1:
        kdata_ground_truth = torch.ones(
            (ncoils, 1, npix**3), dtype=torch.complex64, device=device
        )
    else:
        kdata_ground_truth = torch.ones(
            (ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device
        )

    # k-space indexes
    indexes = _generate_indexes(3, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        image_in = torch.zeros((ncoils, npix, npix, npix), dtype=torch.complex64)
    else:
        image_in = torch.zeros(
            (ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64
        )
    image_in[..., npix // 2, npix // 2, npix // 2] = 1.0

    # computation
    kdata_out = deepmr.fft.sparse_fft(image_in.clone(), indexes=indexes, device=device)

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device", list(itertools.product(*[[2, 3], ncoils, device]))
)
def test_sparse_fft_lowrank3(ncontrasts, ncoils, device, npix=32, width=8):
    # get ground truth
    kdata_ground_truth = torch.ones(
        (ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device
    )

    # k-space indexes
    indexes = _generate_indexes(3, ncontrasts, npix)

    # input
    image_in = torch.zeros(
        (ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64
    )
    image_in[..., npix // 2, npix // 2, npix // 2] = 1.0

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    kdata_out = deepmr.fft.sparse_fft(
        image_in.clone(), indexes=indexes, basis_adjoint=basis_adjoint, device=device
    )

    # check
    npt.assert_allclose(
        kdata_out.detach().cpu(), kdata_ground_truth.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_sparse_ifft1(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    if ncontrasts == 1:
        image_ground_truth = torch.zeros((nslices, ncoils, npix), dtype=torch.complex64)
    else:
        image_ground_truth = torch.zeros(
            (nslices, ncoils, ncontrasts, npix), dtype=torch.complex64
        )
    image_ground_truth[..., npix // 2] = 1.0

    # k-space indexes
    indexes = _generate_indexes(1, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones(
            (nslices, ncoils, 1, npix), dtype=torch.complex64, device=device
        )
    else:
        kdata_in = torch.ones(
            (nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device
        )

    # computation
    image_out = deepmr.fft.sparse_ifft(
        kdata_in.clone(),
        shape=npix,
        indexes=indexes,
        device=device,
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(),
        image_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_sparse_ifft_lowrank1(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    image_ground_truth = torch.zeros(
        (nslices, ncoils, ncontrasts, npix), dtype=torch.complex64
    )
    image_ground_truth[..., npix // 2] = 1.0

    # k-space indexes
    indexes = _generate_indexes(1, ncontrasts, npix)

    # input
    kdata_in = torch.ones(
        (nslices, ncoils, ncontrasts, 1, npix), dtype=torch.complex64, device=device
    )

    # get basis
    basis = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    image_out = deepmr.fft.sparse_ifft(
        kdata_in.clone(),
        shape=npix,
        indexes=indexes,
        basis=basis,
        device=device,
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(),
        image_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_sparse_ifft2(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    if ncontrasts == 1:
        image_ground_truth = torch.zeros(
            (nslices, ncoils, npix, npix), dtype=torch.complex64
        )
    else:
        image_ground_truth = torch.zeros(
            (nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64
        )
    image_ground_truth[..., npix // 2, npix // 2] = 1.0

    # k-space indexes
    indexes = _generate_indexes(2, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones(
            (nslices, ncoils, 1, npix**2), dtype=torch.complex64, device=device
        )
    else:
        kdata_in = torch.ones(
            (nslices, ncoils, ncontrasts, 1, npix**2),
            dtype=torch.complex64,
            device=device,
        )

    # computation
    image_out = deepmr.fft.sparse_ifft(
        kdata_in.clone(),
        shape=2 * [npix],
        indexes=indexes,
        device=device,
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(),
        image_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_sparse_ifft_lowrank2(ncontrasts, ncoils, nslices, device, npix=4):
    # get ground truth
    image_ground_truth = torch.zeros(
        (nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64
    )
    image_ground_truth[..., npix // 2, npix // 2] = 1.0

    # k-space indexes
    indexes = _generate_indexes(2, ncontrasts, npix)

    # input
    kdata_in = torch.ones(
        (nslices, ncoils, ncontrasts, 1, npix**2),
        dtype=torch.complex64,
        device=device,
    )

    # get basis
    basis = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    image_out = deepmr.fft.sparse_ifft(
        kdata_in.clone(),
        shape=2 * [npix],
        indexes=indexes,
        basis=basis,
        device=device,
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(),
        image_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device",
    list(itertools.product(*[[1, 2], ncoils, device])),
)
def test_sparse_ifft3(ncontrasts, ncoils, device, npix=4):
    # get ground truth
    if ncontrasts == 1:
        image_ground_truth = torch.zeros(
            (ncoils, npix, npix, npix), dtype=torch.complex64
        )
    else:
        image_ground_truth = torch.zeros(
            (ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64
        )
    image_ground_truth[..., npix // 2, npix // 2, npix // 2] = 1.0

    # k-space indexes
    indexes = _generate_indexes(3, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        kdata_in = torch.ones(
            (ncoils, 1, npix**3), dtype=torch.complex64, device=device
        )
    else:
        kdata_in = torch.ones(
            (ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device
        )

    # computation
    image_out = deepmr.fft.sparse_ifft(
        kdata_in.clone(),
        shape=3 * [npix],
        indexes=indexes,
        device=device,
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(),
        image_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device",
    list(itertools.product(*[[2, 3], ncoils, device])),
)
def test_sparse_ifft_lowrank3(ncontrasts, ncoils, device, npix=4):
    # get ground truth
    image_ground_truth = torch.zeros(
        (ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64
    )
    image_ground_truth[..., npix // 2, npix // 2, npix // 2] = 1.0

    # k-space indexes
    indexes = _generate_indexes(3, ncontrasts, npix)

    # input
    kdata_in = torch.ones(
        (ncoils, ncontrasts, 1, npix**3), dtype=torch.complex64, device=device
    )

    # get basis
    basis = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    image_out = deepmr.fft.sparse_ifft(
        kdata_in.clone(),
        shape=3 * [npix],
        indexes=indexes,
        basis=basis,
        device=device,
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(),
        image_ground_truth.detach().cpu(),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[1, 2], ncoils, nslices, device])),
)
def test_sparse_fft_selfadjoint2(ncontrasts, ncoils, nslices, device, npix=4):
    # k-space indexes
    indexes = _generate_indexes(2, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        image_in = torch.zeros((nslices, ncoils, npix, npix), dtype=torch.complex64)
    else:
        image_in = torch.zeros(
            (nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64
        )
    image_in[..., npix // 2, npix // 2] = 1.0

    # computation
    toeplitz_kern = deepmr.fft.plan_toeplitz_fft(indexes, npix, device=device)
    image_out = deepmr.fft.apply_sparse_fft_selfadj(
        image_in.clone(), toeplitz_kern, device=device
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(), image_in.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, nslices, device",
    list(itertools.product(*[[2, 3], ncoils, nslices, device])),
)
def test_sparse_fft_selfadjoint_lowrank2(ncontrasts, ncoils, nslices, device, npix=4):
    # k-space indexes
    indexes = _generate_indexes(2, ncontrasts, npix)

    # input
    image_in = torch.zeros(
        (nslices, ncoils, ncontrasts, npix, npix), dtype=torch.complex64
    )
    image_in[..., npix // 2, npix // 2] = 1.0

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    toeplitz_kern = deepmr.fft.plan_toeplitz_fft(
        indexes, npix, device=device, basis=basis_adjoint
    )
    image_out = deepmr.fft.apply_sparse_fft_selfadj(
        image_in.clone(), toeplitz_kern, device=device
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(), image_in.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device", list(itertools.product(*[[1, 2], ncoils, device]))
)
def test_sparse_fft_selfadjoint3(ncontrasts, ncoils, device, npix=4):
    # k-space indexes
    indexes = _generate_indexes(3, ncontrasts, npix)

    # input
    if ncontrasts == 1:
        image_in = torch.zeros((ncoils, npix, npix, npix), dtype=torch.complex64)
    else:
        image_in = torch.zeros(
            (ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64
        )
    image_in[..., npix // 2, npix // 2, npix // 2] = 1.0

    # computation
    toeplitz_kern = deepmr.fft.plan_toeplitz_fft(indexes, npix, device=device)
    image_out = deepmr.fft.apply_sparse_fft_selfadj(
        image_in.clone(), toeplitz_kern, device=device
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(), image_in.detach().cpu(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "ncontrasts, ncoils, device", list(itertools.product(*[[2, 3], ncoils, device]))
)
def test_sparse_fft_selfadjoint_lowrank3(ncontrasts, ncoils, device, npix=4):
    # k-space indexes
    indexes = _generate_indexes(3, ncontrasts, npix)

    # input
    image_in = torch.zeros(
        (ncoils, ncontrasts, npix, npix, npix), dtype=torch.complex64
    )
    image_in[..., npix // 2, npix // 2, npix // 2] = 1.0

    # get basis
    basis_adjoint = torch.eye(ncontrasts, dtype=torch.complex64)

    # computation
    toeplitz_kern = deepmr.fft.plan_toeplitz_fft(
        indexes, npix, device=device, basis=basis_adjoint
    )
    image_out = deepmr.fft.apply_sparse_fft_selfadj(
        image_in.clone(), toeplitz_kern, device=device
    )

    # check
    npt.assert_allclose(
        image_out.detach().cpu(), image_in.detach().cpu(), rtol=tol, atol=tol
    )


# %% local subroutines
def _generate_indexes(ndim, ncontrasts, npix):
    # data type
    dtype = torch.float32

    # build indexes
    nodes = torch.arange(npix, dtype=torch.int16)

    if ndim == 1:
        indexes = nodes[..., None]
    elif ndim == 2:
        x_i, y_i = torch.meshgrid(nodes, nodes, indexing="ij")
        x_i = x_i.flatten()
        y_i = y_i.flatten()
        indexes = torch.stack((x_i, y_i), axis=-1).to(dtype)
    elif ndim == 3:
        x_i, y_i, z_i = torch.meshgrid(nodes, nodes, nodes, indexing="ij")
        x_i = x_i.flatten()
        y_i = y_i.flatten()
        z_i = z_i.flatten()
        indexes = torch.stack((x_i, y_i, z_i), axis=-1).to(dtype)

    # assume single shot trajectory
    indexes = indexes[None, ...]  # (nview=1, nsamples=npix**ndim, ndim=ndim)
    if ncontrasts > 1:
        indexes = torch.repeat_interleave(indexes[None, ...], ncontrasts, axis=0)

    return indexes
