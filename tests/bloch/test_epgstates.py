"""
Test basic EPG states initialization.
"""
import itertools

import pytest
import torch

from deepmr.bloch import EPGstates

# test values
natoms = [1, 2]
nstates = [1, 2]
nlocations = [1, 2]
npulses = [1, 2]
npools = [1, 2]
device = ["cpu"]
if torch.cuda.is_available():
    device += ["cuda:0"]


@pytest.mark.parametrize(
    "device, natoms, nstates, nlocations, npulses",
    list(itertools.product(*[device, natoms, nstates, nlocations, npulses])),
)
def test_bloch_epg(device, natoms, nstates, nlocations, npulses):
    """
    Test initialization of EPG matrix for single species (non-moving).
    """
    # initialize
    buffer = EPGstates(device, natoms, nstates, nlocations, npulses)

    # expected
    F = torch.zeros(
        (natoms, nstates, nlocations, 1, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros(
        (natoms, nstates, nlocations, 1), dtype=torch.complex64, device=device
    )
    Z[:, 0, :, :] = 1

    assert torch.allclose(
        buffer["states"]["F"]["real"] + 1j * buffer["states"]["F"]["imag"], F
    )
    assert torch.allclose(
        buffer["states"]["Z"]["real"] + 1j * buffer["states"]["Z"]["imag"], Z
    )
    assert "Zbound" not in buffer["states"]
    assert "moving" not in buffer["states"]
    assert "moving" not in buffer["states"]


@pytest.mark.parametrize(
    "device, natoms, nstates, nlocations, npulses",
    list(itertools.product(*[device, natoms, nstates, nlocations, npulses])),
)
def test_moving_bloch_epg(device, natoms, nstates, nlocations, npulses):
    """
    Test initialization of EPG matrix for single species (non-moving).
    """
    # initialize
    buffer = EPGstates(device, natoms, nstates, nlocations, npulses, moving=True)

    # expected
    F = torch.zeros(
        (natoms, nstates, nlocations, 1, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros(
        (natoms, nstates, nlocations, 1), dtype=torch.complex64, device=device
    )
    Z[:, 0, :, :] = 1

    Fmoving = torch.zeros(
        (natoms, nstates, nlocations, 1, 2), dtype=torch.complex64, device=device
    )
    Zmoving = torch.zeros(
        (natoms, nstates, nlocations, 1), dtype=torch.complex64, device=device
    )
    Zmoving[:, 0, :, :] = 1

    assert torch.allclose(
        buffer["states"]["F"]["real"] + 1j * buffer["states"]["F"]["imag"], F
    )
    assert torch.allclose(
        buffer["states"]["Z"]["real"] + 1j * buffer["states"]["Z"]["imag"], Z
    )
    assert "Zbound" not in buffer["states"]
    assert torch.allclose(
        buffer["states"]["moving"]["F"]["real"]
        + 1j * buffer["states"]["moving"]["F"]["imag"],
        Fmoving,
    )
    assert torch.allclose(
        buffer["states"]["moving"]["Z"]["real"]
        + 1j * buffer["states"]["moving"]["Z"]["imag"],
        Zmoving,
    )


@pytest.mark.parametrize(
    "device, natoms, nstates, nlocations, npulses",
    list(itertools.product(*[device, natoms, nstates, nlocations, npulses])),
)
def test_blochmcconnell_epg(device, natoms, nstates, nlocations, npulses):
    """
    Test initialization of EPG matrix for multi-pool exchanging model (non-moving).
    """
    # initialize
    weight = torch.as_tensor([0.7, 0.3], dtype=torch.float32, device=device)[None, ...]

    buffer = EPGstates(
        device,
        natoms,
        nstates,
        nlocations,
        npulses,
        npools=2,
        weight=weight,
        model="bm",
    )

    # expected
    F = torch.zeros(
        (natoms, nstates, nlocations, 2, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros(
        (natoms, nstates, nlocations, 2), dtype=torch.complex64, device=device
    )
    Z[:, 0, :, 0] = 0.7
    Z[:, 0, :, 1] = 0.3

    assert torch.allclose(
        buffer["states"]["F"]["real"] + 1j * buffer["states"]["F"]["imag"], F
    )
    assert torch.allclose(
        buffer["states"]["Z"]["real"] + 1j * buffer["states"]["Z"]["imag"], Z
    )
    assert "Zbound" not in buffer["states"]
    assert "moving" not in buffer["states"]
    assert "moving" not in buffer["states"]


@pytest.mark.parametrize(
    "device, natoms, nstates, nlocations, npulses",
    list(itertools.product(*[device, natoms, nstates, nlocations, npulses])),
)
def test_mt_epg(device, natoms, nstates, nlocations, npulses):
    """
    Test initialization of EPG matrix for multi-pool exchanging model (non-moving).
    """
    # initialize
    weight = torch.as_tensor(
        [0.7, 0.3], dtype=torch.float32, device=device
    ) * torch.ones((natoms, 2), dtype=torch.float32, device=device)
    buffer = EPGstates(
        device,
        natoms,
        nstates,
        nlocations,
        npulses,
        npools=1,
        weight=weight,
        model="mt",
    )

    # expected
    F = torch.zeros(
        (natoms, nstates, nlocations, 1, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros(
        (natoms, nstates, nlocations, 1), dtype=torch.complex64, device=device
    )
    Z[:, 0, :, 0] = 0.7
    Zbound = torch.zeros(
        (natoms, nstates, nlocations, 1), dtype=torch.complex64, device=device
    )
    Zbound[:, 0, :, 0] = 0.3

    assert torch.allclose(
        buffer["states"]["F"]["real"] + 1j * buffer["states"]["F"]["imag"], F
    )
    assert torch.allclose(
        buffer["states"]["Z"]["real"] + 1j * buffer["states"]["Z"]["imag"], Z
    )
    assert torch.allclose(
        buffer["states"]["Zbound"]["real"] + 1j * buffer["states"]["Zbound"]["imag"],
        Zbound,
    )
    assert "moving" not in buffer["states"]
    assert "moving" not in buffer["states"]


@pytest.mark.parametrize(
    "device, natoms, nstates, nlocations, npulses",
    list(itertools.product(*[device, natoms, nstates, nlocations, npulses])),
)
def test_blochmcconnell_mt_epg(device, natoms, nstates, nlocations, npulses):
    """
    Test initialization of EPG matrix for multi-pool exchanging model (non-moving).
    """
    # initialize
    weight = torch.as_tensor(
        [0.65, 0.25, 0.1], dtype=torch.float32, device=device
    ) * torch.ones((natoms, 3), dtype=torch.float32, device=device)
    buffer = EPGstates(
        device,
        natoms,
        nstates,
        nlocations,
        npulses,
        npools=2,
        weight=weight,
        model="bm-mt",
    )

    # expected
    F = torch.zeros(
        (natoms, nstates, nlocations, 2, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros(
        (natoms, nstates, nlocations, 2), dtype=torch.complex64, device=device
    )
    Z[:, 0, :, 0] = 0.65
    Z[:, 0, :, 1] = 0.25
    Zbound = torch.zeros(
        (natoms, nstates, nlocations, 1), dtype=torch.complex64, device=device
    )
    Zbound[:, 0, :, 0] = 0.1

    assert torch.allclose(
        buffer["states"]["F"]["real"] + 1j * buffer["states"]["F"]["imag"], F
    )
    assert torch.allclose(
        buffer["states"]["Z"]["real"] + 1j * buffer["states"]["Z"]["imag"], Z
    )
    assert torch.allclose(
        buffer["states"]["Zbound"]["real"] + 1j * buffer["states"]["Zbound"]["imag"],
        Zbound,
    )
    assert "moving" not in buffer["states"]
    assert "moving" not in buffer["states"]
