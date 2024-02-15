"""Design RF pulse for magnetization preparation."""

__all__ = ["adiabatic_inversion", "adiabatic_t2prep"]

import numpy as np

from ._subroutines import _adiabatic as adb
from ..grad import utils

gamma_bar = 42.575 * 1e6  # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar  # rad / T / us -> rad / T / s


def adiabatic_inversion(
    dur, bw, gmax, smax, gdt=4.0, b1peak=14.0, ncycles=16, voxelsize=1
):
    """
    Design an adiabatic inversion pulse

    Args:
        dur (float): pulse duration.
        bw (float): pulse bandwidth in kHz.
        gdt (optional, float): rf and g raster time in [us]. Defaults to 4.0 us.
        gmax (optional, float): maximum slice/slab selection gradient strength in [mT/m].
        smax (float): RF waveform raster time in us.
        b1peak (float, optional):  Peak B1 in uT. Defaults to 15.0 uT.
        ncycles (float, optional): number of phase cycles per voxel for crusher. Defaults to 16.
        voxelsize (float, optional): voxel size along crusher axis in mm. Defaults to 1.

    Returns:
        rf (array): complex rf waveform.
    """
    # casting
    dur *= 1e-3  # ms -> s
    gdt *= 1e-6  # us -> s
    bw *= 1e3  # kHz -> Hz

    # calculate actual design parameters
    n = int(dur // gdt)

    # round to nearest multiple of 4
    n = int(np.ceil(n / 4) * 4)
    dur = n * gdt

    # Perform our WURST pulse design
    am, fm = adb.wurst(n=n, bw=bw, dur=dur)

    # now integral of frequency modulation waveform
    pm = np.cumsum(fm) * gdt

    # minimum of fm waveform and corresponding index
    ifm = np.argmin(np.abs(fm))
    dfm = abs(fm[ifm])

    # find rate of change of frequency at the center of the pulse
    if dfm == 0:
        pm0 = pm[ifm]
    else:  # we need to bracket the zero-crossing
        if fm[ifm] * fm[ifm + 1] < 0:
            b = 1
        else:
            b = -1
        pm0 = (pm[ifm] * fm[ifm + b] - pm[ifm + b] * fm[ifm]) / (fm[ifm + b] - fm[ifm])

    # scaling factor to achieve desired peak b1
    am0 = np.abs(am).max()
    a = b1peak / am0

    # compose complex rf
    pm -= pm0
    rf = a * am * np.exp(1j * pm)

    # get b1sqrd
    b1sqrdTau = (np.sum(np.abs(rf) ** 2, axis=-1)) * gdt

    # build crusher
    g = utils.make_crusher(ncycles, voxelsize, gmax, smax, gdt * 1e6)

    # pad both waveform and crusher
    # rflen = rf.shape[-1]
    # glen = g.shape[-1]

    # rf = np.pad(rf, (0, glen))
    # g = np.pad(g, (rflen, 0))

    # compute time axis
    t = np.arange(rf.shape[-1]) * gdt * 1e3  # ms

    return {"flip": 180.0, "b1sqrdTau": b1sqrdTau}, {
        "rf": rf,
        "grad": {"crush": g},
        "t": t,
    }


def adiabatic_t2prep(te, bw, gmax, smax, gdt=4.0, b1peak=14.0, ncycles=16, voxelsize=1):
    """
    Design an adiabatic inversion pulse

    Args:
        te (array or float): preparation (Echo Times).
        bw (float): pulse bandwidth in kHz.
        gdt (optional, float): rf and g raster time in [us]. Defaults to 4.0 us.
        gmax (optional, float): maximum slice/slab selection gradient strength in [mT/m].
        smax (float): RF waveform raster time in us.
        b1peak (float, optional):  Peak B1 in uT. Defaults to 15.0 uT.
        ncycles (float, optional): number of phase cycles per voxel for crusher. Defaults to 16.
        voxelsize (float, optional): voxel size along crusher axis in mm. Defaults to 1.

    Returns:
        None.

    """
    # get duration as minimum te
    if np.isscalar(te):
        te = [te]
    te = np.asarray(te).astype(float)

    # casting
    te *= 1e-3  # ms -> s
    gdt *= 1e-6  # us -> s
    bw *= 1e3  # kHz -> Hz

    # get duration
    dur = te.min()

    # each subpulse must be dur / 3 ms long
    dur = dur / 3

    # calculate actual design parameters
    n = int(dur // gdt)

    # round to nearest multiple of 8 (for cut)
    n = int(np.ceil(n / 8) * 8)
    dur = n * gdt

    # Perform our WURST pulse design
    am, fm = adb.wurst(n=n, bw=bw, dur=dur)

    # now integral of frequency modulation waveform
    pm = np.cumsum(fm) * gdt

    # minimum of fm waveform and corresponding index
    ifm = np.argmin(np.abs(fm))
    dfm = abs(fm[ifm])

    # find rate of change of frequency at the center of the pulse
    if dfm == 0:
        pm0 = pm[ifm]
    else:  # we need to bracket the zero-crossing
        if fm[ifm] * fm[ifm + 1] < 0:
            b = 1
        else:
            b = -1
        pm0 = (pm[ifm] * fm[ifm + b] - pm[ifm + b] * fm[ifm]) / (fm[ifm + b] - fm[ifm])

    # scaling factor to achieve desired peak b1
    am0 = np.abs(am).max()
    a = b1peak / am0

    # compose complex rf
    pm -= pm0
    rf = a * am * np.exp(1j * pm)

    # get half passage
    center = int(rf.shape[0] // 2)
    rfstart = rf[center:]
    rfend = rf[:center]

    # calculate wait times
    wait = (te - dur) / 2

    # wait times must be multiple of gdt
    wait = np.ceil(wait / gdt) * gdt  # s

    # calculate zeros size
    nwait = (wait // gdt).astype(int)

    # generate pulse
    rfref = [
        np.concatenate((rfstart, np.zeros(n), rf, np.zeros(n), rfend)) for n in nwait
    ]

    # pad
    if len(te) > 1:
        trec = []
        length = max([len(p) for p in rfref])
        for n in range(len(rfref)):
            p = rfref[n]
            padsize = length - p.shape[0]
            p = np.pad(p, (padsize, 0))
            rfref[n] = p
            trec.append(padsize * gdt * 1e3)
        trec = np.asarray(trec)
    else:
        trec = 0.0

    # build
    rfref = np.stack(rfref, axis=0)

    # get b1sqrd
    b1sqrdTau = (np.sum(np.abs(rfref) ** 2, axis=-1)) * gdt

    # build crusher
    g = utils.make_crusher(ncycles, voxelsize, gmax, smax, gdt * 1e6)

    # pad both waveform and crusher
    # rflen = rf.shape[-1]
    # glen = g.shape[-1]

    # rf = np.pad(rf, ((0, 0), (0, glen))).squeeze()
    # g = np.pad(g, (rflen, 0))

    # compute time axis
    t = np.arange(rfref.shape[-1]) * gdt * 1e3  # ms

    # put output together and return
    return {"flip": 0.0, "b1sqrdTau": b1sqrdTau, "rectime": trec}, {
        "rf": rfref,
        "grad": {"crush": g},
        "t": t,
        "rffiles": [rfstart, rf, rfend],
        "wait": wait * 1e3,
    }
