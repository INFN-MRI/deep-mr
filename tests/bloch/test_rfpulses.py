"""
Test EPG RF Pulses operators.

Tested operators:
    - Simple RF pulse (no slice profile, nominal flip angle)
    - More realistic RF pulse (slice profile, single or multiple non-ideal transmit coils)

"""
import itertools

import pytest
import torch

from deepmr.bloch import ops
from deepmr.bloch import EPGstates

# test values
nstates = [1, 2]
npools = [1, 2]
nmodes = [1, 2]
device = ["cpu"]

if torch.cuda.is_available():
    device += ["cuda:0"]

sqrt2 = 2**0.5


# %% no MT
@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_ideal_pulse(device, nstates, npools, nmodes):
    """
    Test RF pulse with ideal pulse.
    """
    alpha = [90.0, 0.0]
    phi = [0.0, 0.0]

    # initialize
    states = EPGstates(device, 1, nstates, 1, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha[:nmodes], phi=phi[:nmodes])

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j
    F[0, ..., 1] = 1j
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_phased_pulse(device, nstates, npools, nmodes):
    """
    Test RF pulse with ideal (phased) pulse.
    """
    alpha = [90.0, 0.0]
    phi = [90.0, 0.0]

    # initialize
    states = EPGstates(device, 1, nstates, 1, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha[:nmodes], phi=phi[:nmodes])

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = 1.0
    F[0, ..., 1] = 1.0
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_B1_pulse(device, nstates, npools, nmodes):
    """
    Test RF pulse with imperfect B1+.
    """
    alpha = [90.0, 0.0]
    B1 = [0.5, 1.0]

    # initialize
    states = EPGstates(device, 1, nstates, 1, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha[:nmodes], B1=B1[:nmodes])

    # apply
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j * 0.5 * sqrt2
    F[0, ..., 1] = 1j * 0.5 * sqrt2
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.5 * sqrt2

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_sliceprof_pulse(device, nstates, npools, nmodes):
    """
    Test RF pulse with imperfect slice profile.
    """
    nlocations = 5
    alpha = [90.0, 0.0]

    # initialize
    sliceprof = 0.5 * torch.ones(nlocations, dtype=torch.float32, device=device)
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha[:nmodes], slice_profile=sliceprof)

    # apply
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j * 0.5 * sqrt2
    F[0, ..., 1] = 1j * 0.5 * sqrt2
    Z = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.5 * sqrt2

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_sliceprof_B1_pulse(device, nstates, npools, nmodes):
    """
    Test RF pulse with imperfect B1+ and slice profile.
    """
    nlocations = 5
    alpha = [90.0, 0.0]
    B1 = [2.0, 1.0]

    # initialize
    sliceprof = 0.5 * torch.ones(nlocations, dtype=torch.float32, device=device)
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(
        device, alpha=alpha[:nmodes], B1=B1[:nmodes], slice_profile=sliceprof
    )

    # apply
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j
    F[0, ..., 1] = 1j
    Z = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)


# %% MT
@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_ideal_pulse_mt(device, nstates, npools, nmodes):
    """
    Test RF pulse with ideal pulse including MT effect.
    """
    # initialize weight
    w1 = torch.as_tensor([0.7, 0.3], dtype=torch.float32, device=device)
    w2 = torch.as_tensor([0.45, 0.45, 0.1], dtype=torch.float32, device=device)

    # init parameters
    w = [w1, w2]
    alpha = [90.0, 0.0]

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha[0]) # uT / deg
    duration = torch.pi / 180.0 * alpha[0] / (267.5221 * 1e-3 * 13)  # [ms]

    # initialize
    states = EPGstates(
        device, 1, nstates, 1, 1, npools, weight=w[npools - 1], model="mt"
    )["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha[:nmodes], b1rms=b1rms, duration=duration)

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j
    F[0, ..., 1] = 1j
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Zbound = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Zbound[0, ...] = 0.7697

    if npools == 1:
        F *= 0.7
        Zbound *= 0.3
    elif npools == 2:
        F *= 0.45
        Zbound *= 0.1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_B1_pulse_mt(device, nstates, npools, nmodes):
    """
    Test RF pulse with ideal pulse including MT effect.
    """
    # initialize weight
    w1 = torch.as_tensor([0.7, 0.3], dtype=torch.float32, device=device)
    w2 = torch.as_tensor([0.45, 0.45, 0.1], dtype=torch.float32, device=device)

    # init parameters
    w = [w1, w2]
    alpha = [90.0, 0.0]
    B1 = [0.5, 1.0]

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha[0])
    duration = torch.pi / 180.0 * alpha[0] / (267.5221 * 1e-3 * 13)  # [ms]

    # initialize
    states = EPGstates(
        device, 1, nstates, 1, 1, npools, weight=w[npools - 1], model="mt"
    )["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(
        device, alpha=alpha[:nmodes], B1=B1[:nmodes], b1rms=b1rms, duration=duration
    )

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j * 0.5 * sqrt2
    F[0, ..., 1] = 1j * 0.5 * sqrt2
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.5 * sqrt2
    Zbound = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Zbound[0, ...] = 0.9366

    if npools == 1:
        F *= 0.7
        Z *= 0.7
        Zbound *= 0.3
    elif npools == 2:
        F *= 0.45
        Z *= 0.45
        Zbound *= 0.1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_sliceprof_pulse_mt(device, nstates, npools, nmodes):
    """
    Test RF pulse with ideal pulse including MT effect.
    """
    # initialize weight
    w1 = torch.as_tensor([0.7, 0.3], dtype=torch.float32, device=device)
    w2 = torch.as_tensor([0.45, 0.45, 0.1], dtype=torch.float32, device=device)

    # init parameters
    w = [w1, w2]
    nlocations = 5
    alpha = [90.0, 0.0]

    # initialize
    sliceprof = 0.5 * torch.ones(nlocations, dtype=torch.float32, device=device)

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha[0])
    duration = torch.pi / 180.0 * alpha[0] / (267.5221 * 1e-3 * 13)  # [ms]

    # initialize
    states = EPGstates(
        device, 1, nstates, nlocations, 1, npools, weight=w[npools - 1], model="mt"
    )["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(
        device,
        alpha=alpha[:nmodes],
        slice_profile=sliceprof,
        b1rms=b1rms,
        duration=duration,
    )

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j * 0.5 * sqrt2
    F[0, ..., 1] = 1j * 0.5 * sqrt2
    Z = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.5 * sqrt2
    Zbound = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    Zbound[0, ...] = 0.9366

    if npools == 1:
        F *= 0.7
        Z *= 0.7
        Zbound *= 0.3
    elif npools == 2:
        F *= 0.45
        Z *= 0.45
        Zbound *= 0.1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_sliceprof_B1_pulse_mt(device, nstates, npools, nmodes):
    """
    Test RF pulse with ideal pulse including MT effect.
    """
    # initialize weight
    w1 = torch.as_tensor([0.7, 0.3], dtype=torch.float32, device=device)
    w2 = torch.as_tensor([0.45, 0.45, 0.1], dtype=torch.float32, device=device)

    # init parameters
    w = [w1, w2]
    nlocations = 5
    alpha = [90.0, 0.0]
    B1 = [2.0, 1.0]

    # initialize
    sliceprof = 0.5 * torch.ones(nlocations, dtype=torch.float32, device=device)

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha[0])  # 8.3 [uT rad**-1]
    duration = torch.pi / 180.0 * alpha[0] / (267.5221 * 1e-3 * 13)  # [ms]

    # initialize
    states = EPGstates(
        device, 1, nstates, nlocations, 1, npools, weight=w[npools - 1], model="mt"
    )["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(
        device,
        alpha=alpha[:nmodes],
        B1=B1[:nmodes],
        slice_profile=sliceprof,
        b1rms=b1rms,
        duration=duration,
    )

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j
    F[0, ..., 1] = 1j
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Zbound = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    Zbound[0, ...] = 0.7697

    if npools == 1:
        F *= 0.7
        Zbound *= 0.3
    elif npools == 2:
        F *= 0.45
        Zbound *= 0.1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


# %% flow
@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_ideal_global_pulse(device, nstates, npools, nmodes):
    """
    Test nonselective RF pulse in presence of moving spins.
    """
    alpha = [90.0, 0.0]
    phi = [0.0, 0.0]

    # initialize
    states = EPGstates(device, 1, nstates, 1, 1, npools, moving=True)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["moving"]["F"] = (
        states["moving"]["F"]["real"][0] + 1j * states["moving"]["F"]["imag"][0]
    )
    states["moving"]["Z"] = (
        states["moving"]["Z"]["real"][0] + 1j * states["moving"]["Z"]["imag"][0]
    )
    pulse = ops.RFPulse(
        device, alpha=alpha[:nmodes], phi=phi[:nmodes], slice_selective=False
    )

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j
    F[0, ..., 1] = 1j
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)

    Fmoving = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    Fmoving[0, ..., 0] = -1j
    Fmoving[0, ..., 1] = 1j
    Zmoving = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)
    assert torch.allclose(states["moving"]["F"], Fmoving, atol=1e-7)
    assert torch.allclose(states["moving"]["Z"], Zmoving, atol=1e-7)


@pytest.mark.parametrize(
    "device, nstates, npools, nmodes",
    list(itertools.product(*[device, nstates, npools, nmodes])),
)
def test_ideal_local_pulse(device, nstates, npools, nmodes):
    """
    Test slice selective RF pulse in presence of moving spins.
    """
    alpha = [90.0, 0.0]
    phi = [0.0, 0.0]

    # initialize
    states = EPGstates(device, 1, nstates, 1, 1, npools, moving=True)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["moving"]["F"] = (
        states["moving"]["F"]["real"][0] + 1j * states["moving"]["F"]["imag"][0]
    )
    states["moving"]["Z"] = (
        states["moving"]["Z"]["real"][0] + 1j * states["moving"]["Z"]["imag"][0]
    )
    pulse = ops.RFPulse(
        device, alpha=alpha[:nmodes], phi=phi[:nmodes], slice_selective=True
    )

    # prepare
    states = pulse(states)

    # expected
    F = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    F[0, ..., 0] = -1j
    F[0, ..., 1] = 1j
    Z = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)

    Fmoving = torch.zeros((nstates, 1, 1, 2), dtype=torch.complex64, device=device)
    Zmoving = torch.zeros((nstates, 1, 1), dtype=torch.complex64, device=device)
    Zmoving[0, :, :] = 1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-7)
    assert torch.allclose(states["Z"], Z, atol=1e-7)
    assert torch.allclose(states["moving"]["F"], Fmoving, atol=1e-7)
    assert torch.allclose(states["moving"]["Z"], Zmoving, atol=1e-7)
