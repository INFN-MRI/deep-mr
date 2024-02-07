"""
Test EPG relaxation operators.

Tested operators:
    - Relaxation (Transverse and Longitudinal) without Exchange
    - Relaxation (Transverse and Longitudinal) with Exchange

"""
import itertools

import pytest
import torch

from deepmr.bloch import ops
from deepmr.bloch import EPGstates

# test values
time = [0.0, 10.0, 1000000.0]
nstates = [1, 2]
nlocations = [1, 2]
device = ["cpu"]

if torch.cuda.is_available():
    device += ["cuda:0"]

@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession(device, nstates, nlocations, time):
    """
    Test free precession in absence of exchange.
    """
    # define
    T1 = 1000.0
    T2 = 100.0

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 1)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    eps = ops.Relaxation(device, time, T1, T2)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0] = -1j * 0.4524
        F[0, ..., 1] = 1j * 0.4524
    elif time == 0.0:
        F[0, ..., 0] = -1j * 0.5000
        F[0, ..., 1] = 1j * 0.5000

    Z = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ...] = 0.8674
    elif time == 0.0:
        Z[0, ...] = 0.8660
    else:
        Z[0, ...] = 1.0

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession_mt(device, nstates, nlocations, time):
    """
    Test free precession in presence of magnetization transfer.
    """
    # define
    T1 = 779
    T2 = 45
    weight = torch.as_tensor([0.883, 0.117], dtype=torch.float32, device=device)
    # weight = weight[None, :]

    # rf
    alpha = 30.0

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha)
    duration = torch.pi / 180.0 * alpha / (267.5221 * 1e-3 * 13)  # [ms]

    # kex = 36.75 ms**-1  
    # k = torch.zeros((2, 2), dtype=torch.float32, device=device)
    # k[0, 0] = -4.3
    # k[1, 1] = -32.45

    # k[1, 0] = 4.3
    # k[0, 1] = 32.45
    k = torch.tensor([36.75], dtype=torch.float32, device=device)
    # k = k[None, :]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 1, weight, "mt")["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha, b1rms=b1rms, duration=duration)
    eps = ops.Relaxation(device, time, T1, T2, weight, k)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0, 0] = -1j * 0.3535
        F[0, ..., 0, 1] = 1j * 0.3535
    elif time == 0.0:
        F[0, ..., 0] = -1j * 0.4415
        F[0, ..., 1] = 1j * 0.4415

    Z = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ..., 0] = 0.7678
    elif time == 0.0:
        Z[0, ..., 0] = 0.7647
    else:
        Z[0, ..., 0] = 0.883

    Zbound = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Zbound[0, ..., 0] = 0.1058
    elif time == 0.0:
        Zbound[0, ..., 0] = 0.1072
    else:
        Zbound[0, ..., 0] = 0.117

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession_exchange(device, nstates, nlocations, time):
    """
    Test free precession in presence of exchange.
    """
    # define
    T1 = torch.as_tensor([1000.0, 500.0], dtype=torch.float32, device=device)
    T2 = torch.as_tensor([100.0, 20.0], dtype=torch.float32, device=device)
    weight = torch.as_tensor([0.8, 0.2], dtype=torch.float32, device=device)
    # weight = weight[None, :]

    # kex = 10.0 ms**-1  
    # k = torch.zeros((2, 2), dtype=torch.float32, device=device)
    # k[0, 0] = -2.0
    # k[1, 1] = -8.0

    # k[1, 0] = 2.0
    # k[0, 1] = 8.0
    k = torch.tensor([10.0], dtype=torch.float32, device=device)
    # k = k[None, :]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 2, weight)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    eps = ops.Relaxation(device, time, T1, T2, weight, k)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 2, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0, 0] = -1j * 0.3607
        F[0, ..., 0, 1] = 1j * 0.3607
        F[0, ..., 1, 0] = -1j * 0.0617
        F[0, ..., 1, 1] = 1j * 0.0617
    elif time == 0.0:
        F[0, ..., 0, 0] = -1j * 0.4
        F[0, ..., 0, 1] = 1j * 0.4
        F[0, ..., 1, 0] = -1j * 0.1
        F[0, ..., 1, 1] = 1j * 0.1

    Z = torch.zeros((nstates, nlocations, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ..., 0] = 0.6939
        Z[0, ..., 1] = 0.1737
    elif time == 0.0:
        Z[0, ..., 0] = 0.6928
        Z[0, ..., 1] = 0.1732
    else:
        Z[0, ..., 0] = 0.8
        Z[0, ..., 1] = 0.2

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession_mt_exchange(device, nstates, nlocations, time):
    """
    Test free precession in presence of magnetization transfer and exchange.
    """
    # define
    T1 = torch.as_tensor([832.0, 400.0], dtype=torch.float32, device=device)
    T2 = torch.as_tensor([80.0, 20.0], dtype=torch.float32, device=device)
    weight = torch.as_tensor([0.75, 0.15, 0.1], dtype=torch.float32, device=device)
    # weight = weight[None, :]

    # rf
    alpha = 30.0

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha)
    duration = torch.pi / 180.0 * alpha / (267.5221 * 1e-3 * 13)  # [ms]


    # nondirectional (WM) are:
    # kex = 13.333333 [s**-1]
    # kmt = 200.0 [s**-1]
    # k = torch.zeros((3, 3), dtype=torch.float32, device=device)
    # k[0, 0] = -2.0
    # k[1, 1] = -30.0
    # k[2, 2] = -30.0

    # k[1, 0] = 2.0
    # k[0, 1] = 10.0
    # k[2, 1] = 20.0
    # k[1, 2] = 30.0
    
    k = torch.tensor([13.333333, 200.0], dtype=torch.float32, device=device)
    # k = k[None, :]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 2, weight, "bm-mt")["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha, b1rms=b1rms, duration=duration)
    eps = ops.Relaxation(device, time, T1, T2, weight, k)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 2, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0, 0] = -1j * 0.3294
        F[0, ..., 0, 1] = 1j * 0.3294
        F[0, ..., 1, 0] = -1j * 0.0385
        F[0, ..., 1, 1] = 1j * 0.0385
    elif time == 0.0:
        F[0, ..., 0, 0] = -1j * 0.3750
        F[0, ..., 0, 1] = 1j * 0.3750
        F[0, ..., 1, 0] = -1j * 0.0750
        F[0, ..., 1, 1] = 1j * 0.0750

    Z = torch.zeros((nstates, nlocations, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ..., 0] = 0.6508
        Z[0, ..., 1] = 0.1315
    elif time == 0.0:
        Z[0, ..., 0] = 0.6495
        Z[0, ..., 1] = 0.1299
    else:
        Z[0, ..., 0] = 0.75
        Z[0, ..., 1] = 0.15

    Zbound = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Zbound[0, ..., 0] = 0.0906
    elif time == 0.0:
        Zbound[0, ..., 0] = 0.0916
    else:
        Zbound[0, ..., 0] = 0.1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession_exchange_chemshift(device, nstates, nlocations, time):
    """
    Test free precession in presence of chemical shift and exchange.
    """
    # define
    T1 = torch.as_tensor([1000.0, 500.0], dtype=torch.float32, device=device)
    T2 = torch.as_tensor([100.0, 20.0], dtype=torch.float32, device=device)
    weight = torch.as_tensor([0.8, 0.2], dtype=torch.float32, device=device)
    # weight = weight[None, :]
    chemshift = torch.as_tensor([0.0, 15.0], dtype=torch.float32, device=device)  # [Hz]

    # k = torch.zeros((2, 2), dtype=torch.float32, device=device)
    # k[0, 0] = -2.0
    # k[1, 1] = -8.0

    # k[1, 0] = 2.0
    # k[0, 1] = 8.0
    k = torch.tensor([10.0], dtype=torch.float32, device=device)
    # k = k[None, :]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 2, weight)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    eps = ops.Relaxation(device, time, T1, T2, weight, k, chemshift)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 2, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0, 0] = -0.0024 - 1j * 0.3600
        F[0, ..., 0, 1] = -0.0024 + 1j * 0.3600
        F[0, ..., 1, 0] = -0.0476 - 1j * 0.0379
        F[0, ..., 1, 1] = -0.0476 + 1j * 0.0379
    elif time == 0.0:
        F[0, ..., 0, 0] = -1j * 0.4
        F[0, ..., 0, 1] = 1j * 0.4
        F[0, ..., 1, 0] = -1j * 0.1
        F[0, ..., 1, 1] = 1j * 0.1

    Z = torch.zeros((nstates, nlocations, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ..., 0] = 0.6939
        Z[0, ..., 1] = 0.1737
    elif time == 0.0:
        Z[0, ..., 0] = 0.6928
        Z[0, ..., 1] = 0.1732
    else:
        Z[0, ..., 0] = 0.8
        Z[0, ..., 1] = 0.2

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession_mt_exchange_chemshift(device, nstates, nlocations, time):
    """
    Test free precession in presence of chemical shift, magnetization transfer and exchange.
    """
    # define
    T1 = torch.as_tensor([832.0, 400.0], dtype=torch.float32, device=device)
    T2 = torch.as_tensor([80.0, 20.0], dtype=torch.float32, device=device)
    weight = torch.as_tensor([0.75, 0.15, 0.1], dtype=torch.float32, device=device)
    # weight = weight[None, :]

    chemshift = torch.as_tensor([0.0, 15.0], dtype=torch.float32, device=device)  # [Hz]

    # rf
    alpha = 30.0

    # pulse stats
    b1rms = 13 / (torch.pi / 180.0 * alpha)
    duration = torch.pi / 180.0 * alpha / (267.5221 * 1e-3 * 13)  # [ms]


    # nondirectional (WM) are:
    # kex = 13.333333 [s**-1]
    # kmt = 200.0 [s**-1]
    # k = torch.zeros((3, 3), dtype=torch.float32, device=device)
    # k[0, 0] = -2.0
    # k[1, 1] = -30.0
    # k[2, 2] = -30.0

    # k[1, 0] = 2.0
    # k[0, 1] = 10.0
    # k[2, 1] = 20.0
    # k[1, 2] = 30.0    
    k = torch.tensor([13.333333, 200.0], dtype=torch.float32, device=device)
    # k = k[None, :]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 2, weight, "bm-mt")["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["Zbound"] = states["Zbound"]["real"][0] + 1j * states["Zbound"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=alpha, b1rms=b1rms, duration=duration)
    eps = ops.Relaxation(device, time, T1, T2, weight, k, chemshift)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 2, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0, 0] = -0.0019 - 1j * 0.3288
        F[0, ..., 0, 1] = -0.0019 + 1j * 0.3288
        F[0, ..., 1, 0] = -0.0292 - 1j * 0.0240
        F[0, ..., 1, 1] = -0.0292 + 1j * 0.0240
    elif time == 0.0:
        F[0, ..., 0, 0] = -1j * 0.3750
        F[0, ..., 0, 1] = 1j * 0.3750
        F[0, ..., 1, 0] = -1j * 0.0750
        F[0, ..., 1, 1] = 1j * 0.0750

    Z = torch.zeros((nstates, nlocations, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ..., 0] = 0.6508
        Z[0, ..., 1] = 0.1315
    elif time == 0.0:
        Z[0, ..., 0] = 0.6495
        Z[0, ..., 1] = 0.1299
    else:
        Z[0, ..., 0] = 0.75
        Z[0, ..., 1] = 0.15

    Zbound = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Zbound[0, ..., 0] = 0.0906
    elif time == 0.0:
        Zbound[0, ..., 0] = 0.0916
    else:
        Zbound[0, ..., 0] = 0.1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)
    assert torch.allclose(states["Zbound"], Zbound, atol=1e-4)


@pytest.mark.parametrize(
    "device, nstates, nlocations, time",
    list(itertools.product(*[device, nstates, nlocations, time])),
)
def test_free_precession_moving(device, nstates, nlocations, time):
    """
    Test free precession in absence of moving spins.
    """
    # define
    T1 = 1000.0
    T2 = 100.0

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, 1, moving=True)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["moving"]["F"] = (
        states["moving"]["F"]["real"][0] + 1j * states["moving"]["F"]["imag"][0]
    )
    states["moving"]["Z"] = (
        states["moving"]["Z"]["real"][0] + 1j * states["moving"]["Z"]["imag"][0]
    )
    pulse = ops.RFPulse(device, alpha=30.0, slice_selective=False)
    eps = ops.Relaxation(device, time, T1, T2)

    # prepare
    states = pulse(states)
    states = eps(states)

    # expected
    F = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        F[0, ..., 0] = -1j * 0.4524
        F[0, ..., 1] = 1j * 0.4524
    elif time == 0.0:
        F[0, ..., 0] = -1j * 0.5000
        F[0, ..., 1] = 1j * 0.5000

    Z = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Z[0, ...] = 0.8674
    elif time == 0.0:
        Z[0, ...] = 0.8660
    else:
        Z[0, ...] = 1.0

    Fmoving = torch.zeros((nstates, nlocations, 1, 2), dtype=torch.complex64, device=device)
    if time == 10.0:
        Fmoving[0, ..., 0] = -1j * 0.4524
        Fmoving[0, ..., 1] = 1j * 0.4524
    elif time == 0.0:
        Fmoving[0, ..., 0] = -1j * 0.5000
        Fmoving[0, ..., 1] = 1j * 0.5000

    Zmoving = torch.zeros((nstates, nlocations, 1), dtype=torch.complex64, device=device)
    if time == 10.0:
        Zmoving[0, ...] = 0.8674
    elif time == 0.0:
        Zmoving[0, ...] = 0.8660
    else:
        Zmoving[0, ...] = 1.0

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)
    assert torch.allclose(states["moving"]["F"], Fmoving, atol=1e-4)
    assert torch.allclose(states["moving"]["Z"], Zmoving, atol=1e-4)
