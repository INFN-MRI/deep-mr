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


@pytest.mark.parametrize(
    "nslices, dtype, device",
    list(itertools.product(*[nslices, dtype, device])),
)
def test_2D_sense_single_contrast(helpers, nslices, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_sense_data(device, 2, nslices, False)

    # build operator
    S = deepmr.linops.SenseOp(2, smap)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check correctness
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    helpers.check_linop_unitary(S, combined_truth)
    helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "dtype, device",
    list(itertools.product(*[dtype, device])),
)
def test_3D_sense_single_contrast(helpers, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_sense_data(device, 3, 32, False)

    # build operator
    S = deepmr.linops.SenseOp(3, smap)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    helpers.check_linop_unitary(S, combined_truth)
    helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "nslices, dtype, device",
    list(itertools.product(*[nslices, dtype, device])),
)
def test_2D_sense__multiple_contrast(helpers, nslices, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_sense_data(device, 2, nslices, True)

    # build operator
    S = deepmr.linops.SenseOp(2, smap, batchmode=True)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    helpers.check_linop_unitary(S, combined_truth)
    helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "dtype, device",
    list(itertools.product(*[dtype, device])),
)
def test_3D_sense__multiple_contrast(helpers, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_sense_data(device, 3, 32, True)

    # build operator
    S = deepmr.linops.SenseOp(3, smap, batchmode=True)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    helpers.check_linop_unitary(S, combined_truth)
    helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "nslices, dtype, device",
    list(itertools.product(*[nslices, dtype, device])),
)
def test_2D_soft_sense_single_contrast(helpers, nslices, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_soft_sense_data(
        device, 2, nslices, False
    )

    # build operator
    S = deepmr.linops.SoftSenseOp(2, smap)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    # helpers.check_linop_unitary(S, combined_truth)
    # helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "dtype, device",
    list(itertools.product(*[dtype, device])),
)
def test_3D_soft_sense_single_contrast(helpers, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_soft_sense_data(device, 3, 32, False)

    # build operator
    S = deepmr.linops.SoftSenseOp(3, smap)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    # helpers.check_linop_unitary(S, combined_truth)
    # helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "nslices, dtype, device",
    list(itertools.product(*[nslices, dtype, device])),
)
def test_2D_soft_sense__multiple_contrast(helpers, nslices, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_soft_sense_data(
        device, 2, nslices, True
    )

    # build operator
    S = deepmr.linops.SoftSenseOp(2, smap, batchmode=True)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    # helpers.check_linop_unitary(S, combined_truth)
    # helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


@pytest.mark.parametrize(
    "dtype, device",
    list(itertools.product(*[dtype, device])),
)
def test_3D_soft_sense__multiple_contrast(helpers, dtype, device):
    # generate data
    combined_truth, img_truth, smap = _generate_soft_sense_data(device, 3, 32, True)

    # build operator
    S = deepmr.linops.SoftSenseOp(3, smap, batchmode=True)

    # actual calculation
    img = S(combined_truth)
    combined = S.H(img_truth)

    # check
    npt.assert_allclose(
        img.numpy(force=True), img_truth.numpy(force=True), rtol=tol, atol=tol
    )
    npt.assert_allclose(
        combined.numpy(force=True), combined_truth.numpy(force=True), rtol=tol, atol=tol
    )

    # check properties
    # helpers.check_linop_unitary(S, combined_truth)
    # helpers.check_linop_linear(S, combined_truth)
    helpers.check_linop_adjoint(S, combined_truth, img_truth)


# %% utils
def _generate_sense_data(device, ndim, nslices, multicontrast):
    # generate data
    combined = deepmr.shepp_logan(32, nslices=nslices)  # (nslices, 32, 32)
    if nslices == 1:
        combined = combined[None, ...]

    # generat e coils
    smap = deepmr.sensmap([4, nslices, 32, 32])  # (4, nslices, 32, 32)
    # smap = smap / (smap * smap.conj()).sum(axis=0)**0.5

    # apply maps
    img = smap * combined  # (4, nslices, 32, 32)

    if multicontrast:
        combined = combined.unsqueeze(
            -ndim - 1
        )  # (nslices, 1, 32, 32) or (1, nslices, 32, 32)
        img = img.unsqueeze(
            -ndim - 1
        )  # (4, nslices, 1, 32, 32) or (4, 1, nslices, 32, 32)

    # for 2D multislice, permute channel and slices axes
    if ndim == 2:
        img = img.swapaxes(0, 1)  # (nslices, 4, 32, 32) or (nslices, 4, 1, 32, 32)
        smap = smap.swapaxes(0, 1)  # (nslices, 4, 32, 32)

    # remove slice axis for 2D singleslice
    if ndim == 2 and nslices == 1:
        combined = combined[0]  # (32, 32) or (1, 32 ,32)
        img = img[0]  # (4, 32, 32) or (4, 1, 32, 32)
        smap = smap[0]  # (4, 32, 32)

    return combined.to(device=device), img.to(device=device), smap.to(device=device)


def _generate_soft_sense_data(device, ndim, nslices, multicontrast):
    # generate data
    combined = deepmr.shepp_logan(32, nslices=nslices)  # (nslices, 32, 32)
    if nslices == 1:
        combined = combined[None, ...]
    combined = torch.stack((combined, 0 * combined), dim=0)  # (2, nslices, 32, 32)

    # generate coils
    smap = deepmr.sensmap([4, nslices, 32, 32])  # (4, nslices, 32, 32)
    smap = torch.stack((smap, 0 * smap), dim=0)  # (2, 4, nslices, 32, 32)

    # apply maps
    img = smap * combined[:, None, ...]  # (2, 4, nslices, 32, 32)
    img = img.sum(axis=0)  # (4, nslices, 32, 32)

    if multicontrast:
        combined = combined.unsqueeze(
            -ndim - 1
        )  # (2, nslices, 1, 32, 32) or (2, 1, nslices, 32, 32)
        img = img.unsqueeze(
            -ndim - 1
        )  # (4, nslices, 1, 32, 32) or (4, 1, nslices, 32, 32)

    # for 2D multislice, permute channel and slices axes
    if ndim == 2:
        combined = combined.swapaxes(
            0, 1
        )  # (nslices, 2, 32, 32) or (nslices, 2, 1, 32, 32)
        img = img.swapaxes(0, 1)  # (nslices, 4, 32, 32) or (nslices, 4, 1, 32, 32)
        smap = smap.permute(2, 0, 1, 3, 4)  # (nslices, 2, 4, 32, 32)

    # remove slice axis for 2D singleslice
    if ndim == 2 and nslices == 1:
        combined = combined[0]  # (2, 32, 32) or (2, 1, 32 ,32)
        img = img[0]  # (2, 4, 32, 32) or (2, 4, 1, 32, 32)
        smap = smap[0]  # (2, 4, 32, 32)

    return combined.to(device=device), img.to(device=device), smap.to(device=device)
