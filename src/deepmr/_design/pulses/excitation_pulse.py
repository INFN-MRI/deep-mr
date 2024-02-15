"""Design RF pulse for excitation."""

__all__ = ["excitation_pulse"]

import numpy as np

from ._subroutines import dzrf
from ._subroutines import ss_design
from ._subroutines import ss_globals
from ._stats import calc_efficiency_profile
from ._stats import calc_isodelay
from ..grad import utils

gambar = 4258  # Hz / G
gam = 2 * np.pi * gambar  # rad / G / s


def excitation_pulse(
    flip, dur, tb=4, dz=None, B0=None, gmax=None, smax=None, gdt=4.0
):  # , nbands=1):
    """
    Design an excitation pulse.

    The pulse can either be nonselective, spatially (slice or slab) selective,
    frequency (i.e., water) selective, or spectral-spatial (i.e., slice/slab + water) selective.

    Multiple bands can be specified for SMS acquisitions (spatially selective)
    or controlled saturation (nonselective / frequency selective).
    Currently, we do not support multiband for spsp pulses, neither 2D/3D spatially selective pulses.

    Args:
        flip (float or array): list of flip angles in [deg].
        dur (float): pulse duration in [ms]. For spsp pulse, this is ignored.
        tb (optional, int): time-bandiwdth product. Defaults to 4.
        dz (optional, float): slice/slab thickness in [mm]. If not provided, pulse is not
            spatially selective (i.e., is either nonselective or frequency selective). If both
            dz and B0 are provided, pulse is spsp.
        B0 (optional, float): field strength in [T]. If not provided, pulse is not
            frequency selective (i.e., is either nonselective or spatially selective). If both
            dz and B0 are provided, pulse is spsp.
        gdt (optional, float): rf and g raster time in [us]. Defaults to 4.0 us.
        gmax (optional, float): maximum slice/slab selection gradient strength in [mT/m].
            Required for spatially-selective and spsp pulses.
        smax (optional, float): maximum slice/slab selection gradient slew rate in [T/m/s].
            Required for spatially-selective and spsp pulses.

    Returns:
        (dict): structure containing the list of flip angles (in deg), of power (uT**2 * ms) and
            slice profile (normalized so that central point has nominal flip angle).
        (dict): structure containing rf/gradient waveforms and time for each rf/gradient point.

    """
    # nbands (optional, int): number of bands. Ignored for spsp pulses. For spatially-selective
    #     pulses, this is the number of simultaneous slices. For frequency-selective, this is the number
    #     of bands.
    # convert
    if np.isscalar(flip):
        flip = np.asarray([flip])

    # unit casting
    dur *= 1e-3  # ms -> s
    if dz is not None:
        dz *= 1e-3  # mm -> m

    # # get maximum flip angle (and then scale)
    alpha = flip.max()

    # determine pulse type
    if dz is not None and B0 is not None:
        ptype = "spectral-spatial"
    elif dz is not None:
        ptype = "spatially-selective"
    elif B0 is not None:
        ptype = "frequency-selective"
    else:
        ptype = "nonselective"

    # check for optional arguments
    # if ptype == "spectral-spatial":
    #     assert nbands == 1, "multiband not supported for spectral-spatial pulses"

    if ptype == "spatially-selective" or ptype == "spectral-spatial":
        assert (
            gmax is not None
        ), "Please provide hardware limitations for spatially selective pulses"
        assert (
            smax is not None
        ), "Please provide hardware limitations for spatially selective pulses"
        assert (
            gdt is not None
        ), "Please provide hardware limitations for spatially selective pulses"

    if "spat" in ptype:
        assert (
            dz is not None and dz > 0
        ), "Please provide slice thickness for spatially selective pulses"

    if "freq" in ptype or "spectral" in ptype:
        assert (
            B0 is not None and B0 > 0
        ), "Please provide field strength for frequency selective pulses"

    # now design
    if ptype == "nonselective":
        rf, g, zprof = _design_nonselective(alpha, dur, tb, gdt)  # , nbands)
    elif ptype == "frequency-selective":
        rf, g, zprof = _design_freqselective(
            alpha, dur, tb, B0, gmax, smax, gdt
        )  # , nbands)
    elif ptype == "spatially-selective":
        rf, g, zprof = _design_spatselective(
            alpha, dur, tb, dz, gmax, smax, gdt
        )  # , nbands)
    elif ptype == "spectral-spatial":
        rf, g, zprof = _design_spsp(alpha, dur, tb, dz, B0, gmax, smax, gdt)

    # get b1sqrd
    b1sqrdTau = (np.sum(np.abs(rf) ** 2)) * gdt * 1e-6

    # compute list of flip angles ([deg]) and power deposition (b1sqrdTau, uT**2 * s)
    scales = flip / alpha
    scales = scales**2
    b1sqrdTau = b1sqrdTau * scales

    # compute time axis
    t = np.arange(rf.shape[0]) * gdt * 1e-3  # ms

    # put output together and return
    return {"flip": flip, "b1sqrdTau": b1sqrdTau, "zprof": zprof}, {
        "rf": rf,
        "grad": {"slice": g},
        "t": t,
    }


# %% local utils
def _design_nonselective(flip, dur, tb, gdt, nbands=1, bandsep=None):
    # design base pulse
    N = int(round(dur / gdt / 1e-6))
    rf0 = dzrf(
        n=N, tb=tb, ptype="st", ftype="min", d1=0.01, d2=0.01, cancel_alpha_phs=True
    )

    # multiband
    if nbands > 1:
        pass
        # # for now support controlled saturation only -> set bands to 3
        # nbands = 3

        # # calculate pulse bandwidth
        # bw = tb / dur

        # # default value for band separation
        # if bandsep is None:
        #     bandsep = 4 * bw

        # # define powers

        # rf = mb_rf(rf0, nbands, bandsep)
    else:
        rf = rf0

    # scale to uT
    rf = rf / rf.sum()  # normalize

    # gamma * rf.sum() * dt must be the flip angle in radians
    rf = rf * np.deg2rad(flip) / ((gam * 1e4) * (gdt * 1e-6))  # T
    rf = rf * 1e6  # uT

    # first and last point to zero
    rf[0] = 0.0
    rf[-1] = 0.0

    # clean phase
    if np.allclose(rf.imag, 0):
        rf = rf.real

    return rf, None, None


def _design_spatselective(flip, dur, tb, dz, gmax, smax, gdt, nslices=1, mbfactor=2):
    # multiband
    if nslices > 1:
        # number of simultaneous slice is nslices / mbfactor
        nbands = int(nslices // mbfactor)

        # slice separation is mbfactor
        bandsep = mbfactor

        # calculate pulse
        rf0, rf, g = dzpins(nbands, dz, mbfactor, gmax, smax, gdt)
    else:
        # design base pulse
        rf0, _, _ = _design_nonselective(flip, dur, tb, gdt)
        rf = rf0

        # get gradient parameters
        bw = tb / dur  # 1 / s
        gamp = bw / dz  # 1 / s / m
        area = gamp * dur  # 1 / m
        npts = rf.shape[0]

        # design gradient
        g, _ = utils.make_trapezoid(2 * np.pi * area, gmax, smax, gdt, npts=npts)
        garea = gam * 1e4 * g.sum() * 1e-3 * gdt * 1e-6

        # pad
        padsize = int((g.shape[0] - rf.shape[0]) // 2)
        rf = np.pad(rf, (padsize, padsize))

        # refocusing
        isodelay = calc_isodelay(rf, gdt * 1e-6)[-1]
        area_frac = 1 - isodelay / rf.shape[0]
        area_rew = 2 * np.pi * (area * area_frac) + 0.5 * (garea - 2 * np.pi * area)
        grew, _ = utils.make_trapezoid(area_rew, gmax, smax, gdt, rampsamp=True)
        g = np.concatenate((g, -grew))
        garea = gam * 1e4 * g.sum() * 1e-3 * gdt * 1e-6

        # pad
        padsize = int(g.shape[0] - rf.shape[0])
        rf = np.pad(rf, (0, padsize))

    # calculate slice profile
    zprof = calc_efficiency_profile(rf0)

    return rf, g, zprof


def _design_freqselective(flip, dur, tb, B0, gmax, smax, gdt):
    rf, _, _ = _design_spsp(flip, dur, tb, 0.5, B0, gmax, smax, gdt)  # , nbands)

    return rf, None, None


def _design_spsp(flip, dur, tb, dz, B0, gmax, smax, gdt):
    # hardcoded constants for water selection
    df = 0.5e-6  # conservative shim requirement
    water = 4.7e-6
    fat2 = 1.3e-6
    fat1 = 0.9e-6

    # get frequency bandwidth
    B0 *= 1e4  # T -> G
    fspec = (
        B0
        * (np.asarray([(fat1 - df), (fat2 + df), (water - df), (water + df)]) - water)
        * gambar
    )
    f_off = (fspec[2] + fspec[3]) / 2
    flip = np.deg2rad(flip)

    # hardcoded options
    sg = ss_globals()
    sg.SS_MAX_DURATION = dur
    sg.SS_NUM_LOBE_ITERS = 5
    sg.SS_VERSE_FRAC = 0.9
    sg.SS_MXG = gmax / 10.0  # G / cm
    sg.SS_MXS = smax / 10.0  # G / cm / ms
    sg.SS_TS = gdt * 1e-6  # s

    # actual design
    g, rf, _ = ss_design(
        dz * 1e2,
        tb,
        np.asarray([0.01, 0.01]),
        fspec,
        np.asarray([0.0, 1.0]) * flip,
        np.asarray([0.02, 0.005]),
        "ex",
        "ls",
        "min",
        "Flyback Half",
        f_off,
        sg=sg,
    )

    # recast
    rf *= 1e2  # G -> uT
    g *= 10  # G/cm -> mT/m

    # clean phase
    if np.allclose(rf.imag, 0):
        rf = rf.real

    # calculate slice profile
    if dz < 0.1:
        zprof = calc_efficiency_profile(rf)
    else:
        zprof = None

    return rf, g, zprof


# %% multibanding tools
def mb_rf(rf0, nbands, bandsep, phs_0_pt="None", beta=None):
    """
    Multiband an input RF pulse.

     Args:
         pulse_in (array): samples of single-band RF pulse.
         n_bands (int): number of bands.
             If negative, additional bands are on the left of main one,
             otherwise on the right (ignored for odd bands).
         band_sep (float): normalized slice separation.
         phs_0_pt (str): set of phases to use. Can be:
                     - 'phs_mod' (Wong);
                     - 'amp_mod' (Malik);
                     - 'quad_mod' (Grissom);
                     - 'None'.
        beta (float): ratio of off-resonant to on-resonant power.
              Defaults to equal power for each band.

     Returns:
         multibanded pulse out

     References:
         Wong, E. (2012). 'Optimized Phase Schedules for Minimizing Peak RF
         Power in Simultaneous Multi-Slice RF Excitation Pulses'. Proc. Intl.
         Soc. Mag. Reson. Med., 20 p. 2209.
         Malik, S. J., Price, A. N., and Hajnal, J. V. (2015). 'Optimized
         Amplitude Modulated Multi-Band RF pulses'. Proc. Intl. Soc. Mag.
         Reson. Med., 23 p. 2398.
    """
    # get sign
    sign = np.sign(nbands)
    nbands = np.abs(nbands)

    # calculate set of phases for multiband modulation
    if phs_0_pt != "None":
        phs = mb_phs_tab(nbands, phs_0_pt)
    else:
        phs = np.zeros(nbands)

    # fix n_bands for even case
    if nbands % 2 == 0:
        is_even = True
    else:
        is_even = False

    # calculate default scaling
    if beta is not None:
        beta = (beta / (nbands - 1)) ** 0.5 * np.ones(nbands)
        beta[nbands // 2] = 1
    else:
        beta = np.ones(nbands)

    # flip for even number of bands
    if sign > 0:
        beta = np.flip(beta)

    # build multiband modulation function
    n = np.size(rf0)
    b = np.zeros(n, dtype=np.complex64)

    for ii in range(nbands):
        t = np.arange(-n / 2, n / 2, 1) / n
        delta = (ii - (nbands - 1) / 2) * bandsep
        b += beta[ii] * np.exp(1j * 2 * np.pi * delta * t) * np.exp(1j * phs[ii])

    # apply modulation
    rf = b * rf0

    # in even case center the main lobe
    if is_even:
        delta = sign * 0.5 * bandsep
        b = np.exp(1j * 2 * np.pi * delta * t)
        rf = b * rf0

    return rf


def mb_phs_tab(n_bands: int, phs_type: str = "phs_mod") -> np.ndarray:
    # Return phases to minimize peak b1 amplitude of an MB pulse
    if phs_type == "phs_mod":
        if n_bands < 3 or n_bands > 16:
            raise Exception("Wongs phases valid for 2 < nBands < 17.")

        # Eric Wong's phases: From E C Wong, ISMRM 2012, p. 2209
        p = np.zeros((14, 16))
        p[0, 1:3] = np.array([0.73, 4.602])
        p[1, 1:4] = np.array([3.875, 5.94, 6.197])
        p[2, 1:5] = np.array([3.778, 5.335, 0.872, 0.471])
        p[3, 1:6] = np.array([2.005, 1.674, 5.012, 5.736, 4.123])
        p[4, 1:7] = np.array([3.002, 5.998, 5.909, 2.624, 2.528, 2.440])
        p[5, 1:8] = np.array([1.036, 3.414, 3.778, 3.215, 1.756, 4.555, 2.467])
        p[6, 1:9] = np.array([1.250, 1.783, 3.558, 0.739, 3.319, 1.296, 0.521, 5.332])
        p[7, 1:10] = np.array(
            [4.418, 2.360, 0.677, 2.253, 3.472, 3.040, 3.974, 1.192, 2.510]
        )
        p[8, 1:11] = np.array(
            [5.041, 4.285, 3.001, 5.765, 4.295, 0.056, 4.213, 6.040, 1.078, 2.759]
        )
        p[9, 1:12] = np.array(
            [
                2.755,
                5.491,
                4.447,
                0.231,
                2.499,
                3.539,
                2.931,
                2.759,
                5.376,
                4.554,
                3.479,
            ]
        )
        p[10, 1:13] = np.array(
            [
                0.603,
                0.009,
                4.179,
                4.361,
                4.837,
                0.816,
                5.995,
                4.150,
                0.417,
                1.520,
                4.517,
                1.729,
            ]
        )
        p[11, 1:14] = np.array(
            [
                3.997,
                0.830,
                5.712,
                3.838,
                0.084,
                1.685,
                5.328,
                0.237,
                0.506,
                1.356,
                4.025,
                4.483,
                4.084,
            ]
        )
        p[12, 1:15] = np.array(
            [
                4.126,
                2.266,
                0.957,
                4.603,
                0.815,
                3.475,
                0.977,
                1.449,
                1.192,
                0.148,
                0.939,
                2.531,
                3.612,
                4.801,
            ]
        )
        p[13, 1:16] = np.array(
            [
                4.359,
                3.510,
                4.410,
                1.750,
                3.357,
                2.061,
                5.948,
                3.000,
                2.822,
                0.627,
                2.768,
                3.875,
                4.173,
                4.224,
                5.941,
            ]
        )

        out = p[n_bands - 3, 0:n_bands]

    elif phs_type == "amp_mod":
        # Malik's Hermitian phases: From S J Malik, ISMRM 2015, p. 2398
        if n_bands < 4 or n_bands > 12:
            raise Exception("Maliks phases valid for 3 < nBands < 13.")

        p = np.zeros((9, 12))
        p[0, 0:4] = np.array([0, np.pi, np.pi, 0])
        p[1, 0:5] = np.array([0, 0, np.pi, 0, 0])
        p[2, 0:6] = np.array([1.691, 2.812, 1.157, -1.157, -2.812, -1.691])
        p[3, 0:7] = np.array([2.582, -0.562, 0.102, 0, -0.102, 0.562, -2.582])
        p[4, 0:8] = np.array(
            [2.112, 0.220, 1.464, 1.992, -1.992, -1.464, -0.220, -2.112]
        )
        p[5, 0:9] = np.array(
            [0.479, -2.667, -0.646, -0.419, 0, 0.419, 0.646, 2.667, -0.479]
        )
        p[6, 0:10] = np.array(
            [1.683, -2.395, 2.913, 0.304, 0.737, -0.737, -0.304, -2.913, 2.395, -1.683]
        )
        p[7, 0:11] = np.array(
            [
                1.405,
                0.887,
                -1.854,
                0.070,
                -1.494,
                0,
                1.494,
                -0.070,
                1.854,
                -0.887,
                -1.405,
            ]
        )
        p[8, 0:12] = np.array(
            [
                1.729,
                0.444,
                0.722,
                2.190,
                -2.196,
                0.984,
                -0.984,
                2.196,
                -2.190,
                -0.722,
                -0.444,
                -1.729,
            ]
        )

        out = p[n_bands - 4, 0:n_bands]

    elif phs_type == "quad_mod":
        # Grissom's quadratic phases (unpublished)
        k = 3.4 / n_bands  # quadratic phase coefficient
        out = k * (np.arange(0, n_bands, 1) - (n_bands - 1) / 2) ** 2

    else:
        raise Exception('phase type ("{}") not recognized.'.format(phs_type))

    return out


def dzpins(nslices, dz, slicesep, gmax, smax, gdt, b1_max=0.18):
    """PINS multiband pulse design."""
    # hardcodec constant
    gambar = 4258

    # find zsep (slicesep * dz)
    zsep = slicesep * dz

    # get width in k-space
    dkz = nslices / zsep

    # width in k-space
    tb = dz * dkz

    # call SLR to get envelope
    rf0 = dzrf(nslices, tb, ptype="ex", ftype="ls", d1=0.01, d2=0.01)

    # design the blip trapezoid
    area = 1 / zsep / gambar
    gz_blip, _ = utils.make_trapezoid(area, gmax, smax, gdt, True)

    # Calculate the block/hard RF pulse width based on
    b1_scaled = 2 * np.pi * gambar * b1_max * gdt
    hpw = int(np.ceil(np.max(np.abs(rf0)) / b1_scaled))

    # interleave RF subpusles with gradient subpulses to form full pulses
    rf = np.kron(
        rf0[:-1],
        np.concatenate((np.ones(hpw), np.zeros((np.size(gz_blip))))),
    )
    rf = np.concatenate((rf, rf0[-1] * np.ones(hpw)))
    rf = rf / (np.sum(rf) * 2 * np.pi * gambar * gdt) * np.sum(rf0)

    g = np.concatenate([np.zeros(hpw), np.squeeze(gz_blip)])
    g = np.tile(g, nslices - 1)
    g = np.concatenate((g, np.zeros(hpw)))

    return rf0, rf, g
