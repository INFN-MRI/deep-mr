"""
Test basic EPG operators.

Tested operators:
    - Shifting (reversible dephasing due to unbalanced gradients)
    - Spoiling (perfect gradient crusher or perfect RF spoil)

"""
import itertools

import pytest
import torch

from deepmr.bloch import ops
from deepmr.bloch import EPGstates

# test values
device = ["cpu"]
nlocations = [1, 2]
npools = [1, 2]

if torch.cuda.is_available():
    device += ["cuda:0"]


@pytest.mark.parametrize(
    "device, nlocations, npools",
    list(itertools.product(*[device, nlocations, npools])),
)
def test_epg_shift(device, nlocations, npools):
    """
    Test configuration shift due to unbalanced gradients.
    """
    # define
    nstates = 2

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    grad = ops.Shift()

    # prepare
    states = pulse(states)
    states = grad(states)

    # expected
    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    F[1, ..., 0] = -1j * 0.5000

    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.8660

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, nlocations, npools",
    list(itertools.product(*[device, nlocations, npools])),
)
def test_spoil(device, nlocations, npools):
    """
    Test perfect spoiling (destroy transverse magnetization).
    """
    # define
    nstates = 2

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    crusher = ops.Spoil()

    # prepare
    states = pulse(states)
    states = crusher(states)

    # expected
    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.8660

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)
