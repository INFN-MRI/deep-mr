"""
Test EPG motion operators.

Tested operators:
    - Diffusion Damping (only for isotropic diffusion)
    - Bulk Flow (both in-flow effect and dephasing)

"""
import itertools

import numpy as np
import pytest
import torch

from deepmr.bloch import ops
from deepmr.bloch import EPGstates

# test values
device = ["cpu"]
time = [0.0, 1.0]
nlocations = [1, 2]
npools = [1, 2]
direction = ["x", "y", "z", (0.0, 0.0, 1.0), None]

if torch.cuda.is_available():
    device += ["cuda:0"]


@pytest.mark.parametrize(
    "device, time, nlocations, npools, direction",
    list(itertools.product(*[device, time, nlocations, npools, direction])),
)
def test_diffusion_damping_totdephasing_and_voxelsize(
    device, time, nlocations, npools, direction
):
    """
    Test diffusion damping
    """
    # define
    nstates = 2
    D = 1.0  # [um**2 ms**-1]
    total_dephasing = 16 * torch.pi  # [rad]
    voxelsize = [1.0, 1.0, 2.5]  # [mm]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    grad = ops.Shift()
    diff = ops.DiffusionDamping(
        device,
        time,
        D,
        nstates,
        total_dephasing=total_dephasing,
        voxelsize=voxelsize,
        grad_direction=direction,
    )

    # prepare
    states = pulse(states)
    states = grad(states)
    states = diff(states)

    # expected
    if direction == "x" or direction == "y":
        value = 0.4971
    else:
        value = 0.4995

    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    if time == 1.0:
        F[1, ..., 0] = -1j * value
    elif time == 0.0:
        F[1, ..., 0] = -1j * 0.5000

    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.8660

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, time, nlocations, npools",
    list(itertools.product(*[device, time, nlocations, npools])),
)
def test_diffusion_damping_grad_specs(device, time, nlocations, npools):
    """
    Test diffusion damping
    """
    # define
    nstates = 2
    D = 1.0  # [um**2 ms**-1]
    grad_amp = 75  # [mT / m]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    grad = ops.Shift()
    diff = ops.DiffusionDamping(device, time, D, nstates, grad_amplitude=grad_amp)

    # prepare
    states = pulse(states)
    states = grad(states)
    states = diff(states)

    # expected
    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    if time == 1.0:
        F[1, ..., 0] = -1j * 0.4995
    elif time == 0.0:
        F[1, ..., 0] = -1j * 0.5000

    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.8660

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, time, nlocations, npools, direction",
    list(itertools.product(*[device, time, nlocations, npools, direction])),
)
def test_flow_dephasing_totdephasing_and_voxelsize(
    device, time, nlocations, npools, direction
):
    """
    Test flow dephasing
    """
    # define
    nstates = 1
    v = 100.0  # [cm s**-1]
    total_dephasing = 2 * torch.pi  # [rad]
    voxelsize = [2.5, 2.5, 5.0]  # [mm]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    dphase = ops.FlowDephasing(
        device,
        time,
        v,
        nstates,
        total_dephasing=total_dephasing,
        voxelsize=voxelsize,
        grad_direction=direction,
    )

    # prepare
    states = pulse(states)
    states = dphase(states)

    # expected
    if direction == "x" or direction == "y":
        value = -0.4755 - 0.1545j
    else:
        value = -0.2939 - 0.4045j

    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    if time == 1.0:
        F[0, ..., 0] = value
        F[0, ..., 1] = np.conj(value)
    elif time == 0.0:
        F[0, ..., 0] = -1j * 0.5000
        F[0, ..., 1] = 1j * 0.5000

    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.8660

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, time, nlocations, npools",
    list(itertools.product(*[device, time, nlocations, npools])),
)
def test_flow_dephasing_grad_specs(device, time, nlocations, npools):
    """
    Test flow dephasing
    """
    # define
    nstates = 1
    v = 100.0  # [cm s**-1]
    grad_amp = 10.0  # [mT / m]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    pulse = ops.RFPulse(device, alpha=30.0)
    dphase = ops.FlowDephasing(device, time, v, nstates, grad_amplitude=grad_amp)

    # prepare
    states = pulse(states)
    states = dphase(states)

    # expected
    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    if time == 1.0:
        F[0, ..., 0] = -0.4865 - 0.1155j
        F[0, ..., 1] = -0.4865 + 0.1155j
    elif time == 0.0:
        F[0, ..., 0] = -1j * 0.5000
        F[0, ..., 1] = 1j * 0.5000

    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    Z[0, ...] = 0.8660

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)


@pytest.mark.parametrize(
    "device, time, nlocations, npools, direction",
    list(
        itertools.product(
            *[device, [0.0, 1.0, 1000000.0], nlocations, npools, direction]
        )
    ),
)
def test_magnetization_replacement(device, time, nlocations, npools, direction):
    """
    Test magnetization replacement due to spin wash-out / inflow.
    """
    nstates = 2
    v = 100.0  # [cm s**-1]
    voxelsize = [2.5, 2.5, 5.0]  # [mm]

    # initialize
    states = EPGstates(device, 1, nstates, nlocations, 1, npools, moving=True)["states"]
    states["F"] = states["F"]["real"][0] + 1j * states["F"]["imag"][0]
    states["Z"] = states["Z"]["real"][0] + 1j * states["Z"]["imag"][0]
    states["moving"]["F"] = (
        states["moving"]["F"]["real"][0] + 1j * states["moving"]["F"]["imag"][0]
    )
    states["moving"]["Z"] = (
        states["moving"]["Z"]["real"][0] + 1j * states["moving"]["Z"]["imag"][0]
    )
    pulse = ops.RFPulse(device, alpha=30.0, slice_selective=True)
    wash = ops.FlowWash(device, time, v, voxelsize, direction)

    # prepare
    states = pulse(states)
    states = wash(states)

    # expected
    if direction == "x" or direction == "y":
        fvalue = -0.3000j
        zvalue = 0.9196
    else:
        fvalue = -0.4000j
        zvalue = 0.8928

    F = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    Z = torch.zeros((nstates, nlocations, npools), dtype=torch.complex64, device=device)
    if time == 1.0:
        F[0, ..., 0] = fvalue
        F[0, ..., 1] = np.conj(fvalue)
        Z[0, ...] = zvalue
    elif time == 0.0:
        F[0, ..., 0] = -1j * 0.5000
        F[0, ..., 1] = 1j * 0.5000
        Z[0, ...] = 0.8660
    else:
        Z[0, ...] = 1.0

    Fmoving = torch.zeros(
        (nstates, nlocations, npools, 2), dtype=torch.complex64, device=device
    )
    Zmoving = torch.zeros(
        (nstates, nlocations, npools), dtype=torch.complex64, device=device
    )
    Zmoving[0, :, :] = 1

    # assertions
    assert torch.allclose(states["F"], F, atol=1e-4)
    assert torch.allclose(states["Z"], Z, atol=1e-4)
    assert torch.allclose(states["moving"]["F"], Fmoving, atol=1e-4)
    assert torch.allclose(states["moving"]["Z"], Zmoving, atol=1e-4)
