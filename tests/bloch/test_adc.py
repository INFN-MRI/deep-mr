"""
Test EPG adc operators.

Tested operators:
    - Signal recording

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
def test_observe(device, nlocations, npools):
    """
    Test ADC recording.
    """
    # define
    w = [
        torch.as_tensor([1.0], dtype=torch.float32, device=device),
        torch.as_tensor([0.5, 0.5], dtype=torch.float32, device=device),
    ]
    nstates = 2

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools, weight=w[npools - 1])[
        "states"
    ]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=90.0)

    # prepare
    states = pulse(states)
    signal = ops.observe(states)

    # expected
    expected = torch.as_tensor(-1j, dtype=torch.complex64, device=device)

    # assertions
    assert torch.allclose(signal, expected, atol=1e-4)


@pytest.mark.parametrize(
    "device, nlocations, npools",
    list(itertools.product(*[device, nlocations, npools])),
)
def test_observe_phase_demodulateion(device, nlocations, npools):
    """
    Test ADC recording with phase demodulation.
    """
    # define
    w = [
        torch.as_tensor([1.0], dtype=torch.float32, device=device),
        torch.as_tensor([0.5, 0.5], dtype=torch.float32, device=device),
    ]
    nstates = 2

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools, weight=w[npools - 1])[
        "states"
    ]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=90.0, phi=90.0)

    # prepare
    states = pulse(states)
    signal = ops.observe(states, pulse.phi)

    # expected
    expected = torch.as_tensor(-1j, dtype=torch.complex64, device=device)

    # assertions
    assert torch.allclose(signal, expected, atol=1e-4)
