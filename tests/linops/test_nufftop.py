"""Test NUFFT linear operator."""

import itertools
import pytest

import torch
import deepmr

ndim = [2, 3]
nslices = [1, 32]
dtype = [torch.float32, torch.float64, torch.complex64, torch.complex128]
device = ["cpu"]
if torch.cuda.is_available():
    device += ["cuda"]


@pytest.mark.parametrize(
    "nslices, dtype, device",
    list(itertools.product(*[nslices, dtype, device])),
)
def test_2D_nufft_single_contrast(helpers, nslices, dtype, device):
    # generate data
    img, head = _generate_radial_data(device, 2, nslices, False)

    # build operator
    F = deepmr.linops.NUFFTOp(
        head.traj,
        head.shape[-2:],
        weight=head.dcf,
        device=device,
        oversamp=2.0,
        width=8,
    )

    # actual calculation
    data = F(img)

    # check properties
    helpers.check_linop_linear(F, img)
    helpers.check_linop_adjoint(F, img, data)


# %% utils
def _generate_radial_data(device, ndim, nslices, multicontrast):
    # generate data
    if ndim == 2:
        img = deepmr.shepp_logan(32, nslices=nslices)  # (nslices, 32, 32)
        if nslices == 1:
            img = img[None, ...]  # (1, 32, 32)
    elif ndim == 3:
        img = deepmr.shepp_logan(32, nslices=32)  # (nslices=32, 32, 32)

    # generate trajectory
    if ndim == 2:
        if multicontrast:
            head = deepmr.radial((32, 1, 4))  # (4, 100, 32, 2)
        else:
            head = deepmr.radial(32)  # (1, 100, 32, 2)
    elif ndim == 3:
        if multicontrast:
            head = deepmr.radial_proj((32, 1, 4))
        else:
            head = deepmr.radial_proj(32)

    if multicontrast:
        img = img.unsqueeze(-ndim - 1)  # (nslices, 1, 32, 32) or (1, nslices, 32, 32)
        img = torch.repeat_interleave(img, 4, dim=-ndim - 1)  # (4, nslices, 32, 32)

    # for 2D multislice, permute contrast and slices axes
    if ndim == 2 and multicontrast:
        img = img.swapaxes(0, 1)  # (nslices, 4, 32, 32)

    # remove slice axis for 2D singleslice
    if ndim == 2 and nslices == 1:
        img = img[0]  # (32, 32) or (4, 32, 32)

    return img.to(device=device), head
