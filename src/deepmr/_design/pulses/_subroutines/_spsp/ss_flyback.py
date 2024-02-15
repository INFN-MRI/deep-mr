"""Flyback design."""

__all__ = ["ss_flyback"]

import numpy as np

from .._slr import dzbeta
from .._slr import ab2ex
from .._slr import ab2se
from .._slr import abr
from .._slr import b2rf

from .ss_alias import ss_alias
from .ss_filters import fir_minphase_power
from .ss_filters import fir_qprog
from .ss_filters import fir_min_order_qprog
from .ss_fourier import fftf
from .ss_fourier import fftr
from .ss_globals import ss_globals
from .ss_grad import grad_ss
from .ss_grad import grad_mintrap
from .ss_spect_correct import ss_spect_correct
from .ss_verse import ss_verse
from .ss_verse import ss_b1verse
from .spec_interp import spec_interp


def ss_flyback(
    ang,
    z_thk,
    z_tb,
    z_d,
    f,
    a,
    d,
    fs,
    ptype,
    z_ftype,
    s_ftype,
    ss_type,
    f_off,
    dbg,
    sg=None,
):
    fs = float(fs)

    # global options
    if sg is None:
        sg = ss_globals()

    if "Half" in ss_type:
        sym_flag = 1
    else:
        sym_flag = 0
    f_a, a_a, d_a, f_off = ss_alias(f, a, d, f_off, fs, sym_flag)

    if len(f_a) == 0:
        raise ValueError("Strange: this frequency should be ok")

    kz_max = z_tb / z_thk
    kz_area = kz_max / sg.SS_GAMMA
    nsamp = round(1 / (fs * sg.SS_TS))

    # Generate gradients
    gpos, gneg, g1, g2, g3 = grad_ss(
        kz_area,
        nsamp,
        sg.SS_VERSE_FRAC,
        sg.SS_MXG,
        sg.SS_MXS,
        sg.SS_TS,
        sg.SS_EQUAL_LOBES,
    )
    ng1 = len(g1)
    ng2 = len(g2)
    ng3 = len(g3)

    t_poslobe = len(gpos) * sg.SS_TS
    t_lobe = (len(gpos) + len(gneg)) * sg.SS_TS
    max_lobe = int((sg.SS_MAX_DURATION - t_poslobe) / t_lobe) + 1

    a_dup = np.zeros(f_a.shape)
    a_dup[::2] = a_a
    a_dup[1::2] = a_a

    if sg.SS_MIN_ORDER:
        use_max = 0
    else:
        use_max = 1

    if s_ftype == "min":
        s_b, status = fir_minphase_power(max_lobe, f_a, a_dup, d_a, use_max)
    elif s_ftype == "max":
        s_b, status = fir_minphase_power(max_lobe, f_a, a_dup, d_a, use_max)
        s_b = np.conj(s_b[::-1])
    elif s_ftype == "lin":
        if use_max:
            if max_lobe % 2 == 0:
                max_lobe += 1
            s_b, status = fir_qprog(max_lobe, f_a, a_dup, d_a, 0)
        else:
            odd_or_even = 0
            s_b, status = fir_min_order_qprog(max_lobe, f_a, a_dup, d_a, odd_or_even)

    if status == "Solved":
        # Get Z pulse
        if dbg:
            print("Getting Z RF pulse")
        z_np = len(g2)
        z_b = dzbeta(z_np, z_tb, "st", z_ftype, z_d[0], z_d[1])

        # Correct for non-linear effects with SLR if desired
        #
        if sg.SS_SLR_FLAG == 1:
            # % Calculate excitation profile assuming in small-tip regime
            # %
            oversamp = 4
            nZ = oversamp * z_np
            nZ2 = 2 ** int(np.ceil(np.log2(nZ)))
            Z_b = fftf(z_b, nZ2)  # column transform, unit magnitude

            if sg.SS_SPECT_CORRECT_FLAG:
                # % Interpolate spectral filter on a grid that's equal to
                # % the number of time samples of the gradient --- do this
                # % partly before calling b2rf and partly afterwards
                # %
                print(
                    "Note: Spectral Correction for SLR pulses only corrects base bandwidth (not aliased bands)"
                )

                Nper = len(gpos) + len(gneg)
                oversamp_slr = 16
                Ntotal = Nper * len(s_b)

                off = -int(len(z_np) / 2) / Nper

                bsf = np.sin(ang / 2) * Z_b
                s_rfm = np.zeros((len(bsf), len(s_b) * oversamp_slr))

                s_bi = spec_interp(s_b, oversamp_slr, off, f_a, dbg)
                for idx in range(nZ2):
                    for bidx in range(oversamp_slr):
                        s_rfm[idx, bidx::oversamp_slr] = b2rf(
                            bsf[idx] * s_bi[bidx::oversamp_slr]
                        )

                # % Now calculate new z beta polynomial by inverse FFT, then use
                # % SLR to get RF
                # %
                z_bm = fftr(
                    np.sin(s_rfm / 2), z_np, 0
                )  #  Each row now scaled by tip-angle

                # % Now do SLR in Z direction
                # %
                if dbg:
                    print("Doing SLR in Z...  ")
                z_rf = np.zeros(z_bm.shape, dtype=np.complex128)
                for idx in range(z_bm.shape[1]):
                    z_rf[:, idx] = np.conj(b2rf(z_bm[:, idx]))

                # % Now raster scan for actual sampling positions
                # %
                z_rfm = []
                for idx in range(len(s_b)):
                    for zidx in range(z_np):
                        idx_intrp = int(
                            1 + ((idx - 1) * Nper + zidx - 1) / Ntotal * len(s_bi)
                        )
                        tmp_z_rf = np.interp(
                            range(z_rf.shape[1]),
                            range(z_rf.shape[1]),
                            z_rf[zidx, :].real,
                            idx_intrp,
                            "spline",
                        )
                        z_rfm.append(tmp_z_rf)

            else:  # no spectral correction
                if dbg:
                    print("Doing SLR in F...")

                s_rfm = []
                bsf = np.sin(ang / 2) * Z_b
                for idx in range(nZ2):
                    tmp_s_rf = b2rf(bsf[idx] * s_b)
                    s_rfm.append(tmp_s_rf)

                # % Now calculate new z beta polynomial by inverse FFT, then use
                # % SLR to get RF
                # %
                z_bm = fftr(
                    np.sin(s_rfm / 2), z_np, 1
                )  #  Each row now scaled by tip-angle
                z_rfm = []
                if dbg:
                    print("Doing SLR in Z...  ")
                for idx in range(z_bm.shape[0]):
                    tmp_z_rf = np.conj(b2rf(z_bm[idx, :]))
                    z_rfm.append(tmp_z_rf)

            # Modulate rf to passband frequency BEFORE versing, then
            # modulate back. This will make sure that the slice profile
            # shows no blurring at the passband.  In the case that
            # multiple passbands are defined, then the midpoint of
            # the first passband is used
            #
            pass_idx = np.where(a > 0)[0][0]
            fpass = [f[2 * pass_idx - 1], f[2 * pass_idx]]
            fpass_mid = np.mean(fpass) - f_off

            nlobe = len(s_b)
            rfmod = np.exp(1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid)

            if sg.SS_VERSE_B1:
                if dbg:
                    print("Versing RF with B1 minimization...  ")
                # Additional VERSE with B1 restriction, maintaining duration
                z_rfmax = np.max(np.abs(z_rfm))

                z_rfvmax1 = ss_verse(g2, z_rfmax)
                z_rfvmax, g2v = ss_b1verse(
                    g2,
                    z_rfvmax1,
                    sg.SS_MAX_B1,
                    sg.SS_MXG,
                    sg.SS_MXS,
                    sg.SS_TS,
                    sg.SS_GAMMA,
                    sg.SS_SLEW_PENALTY,
                    dbg,
                )

                if len(z_rfvmax) == 0:
                    rf = np.array([], dtype=np.complex128)
                    g = np.array([])
                    return rf, g

                z_rfmv = []
                for idx in range(nlobe):
                    z_rfmod = z_rfm[idx] * rfmod
                    z_rfvmod = ss_verse(g2v, z_rfmod)
                    z_rfv = z_rfvmod * np.conj(rfmod)
                    z_rfmv.append(z_rfv)
                z_rfmv = np.asarray(z_rfmv)

                # update gradient
                gpos = np.concatenate((g1, g2v, g3))
                if sg.SS_EQUAL_LOBES:
                    gneg = -gpos
            else:
                if dbg:
                    print("Versing RF...  ")

                z_rfmv = []
                for idx in range(nlobe):
                    z_rfmod = z_rfm[idx] * rfmod
                    if sg.SS_VERSE_FRAC == 0:
                        z_rfvmod = z_rfmod
                    else:
                        z_rfvmod = ss_verse(g2, z_rfmod)
                    z_rfv = z_rfvmod * np.conj(rfmod)
                    z_rfmv.append(z_rfv)
                z_rfmv = np.asarray(z_rfmv)

        else:  # no SLR
            if sg.SS_SPECT_CORRECT_FLAG:
                # % Spectral correction needs to be applied to unaliased bands,
                # % therefore the raw frequency spec needs to be passed through.
                # %
                # %		s_b = [0; s_b; 0];  % play with adding extra taps to make
                # % spectral correction easier
                Nper = len(gpos) + len(gneg)
                st_off = -int(ng2 / 2)
                Noff = range(st_off, st_off + ng2)
                bsf = np.sin(ang / 2) * np.ones(len(Noff))

                # updated
                if dbg:
                    print("No SLR.. spectral correction...")
                s_rfm = ss_spect_correct(
                    s_b,
                    bsf,
                    Nper,
                    Noff,
                    (f - f_off) / (fs / 2),
                    ptype,
                    "Flyback",
                    sg.SS_SLR_FLAG,
                    sg.SS_SPECT_CORRECT_REGULARIZATION,
                    dbg,
                )
            else:
                s_rfm = (
                    ang
                    * np.conj(s_b)[:, None]
                    * np.ones((len(s_b), ng2), dtype=np.complex128)
                )
                # % here -- needed because of
                # % possible asymmetric frequency
                # % response

            # Modulate rf to passband frequency BEFORE versing, then
            # modulate back. This will make sure that the slice profile
            # shows no blurring at the passband.  In the case that
            # multiple passbands are defined, then the midpoint of
            # the first passband is used
            #
            pass_idx = np.where(a > 0)[0][0]
            fpass = [f[2 * pass_idx - 1], f[2 * pass_idx]]
            fpass_mid = np.mean(fpass) - f_off

            z_bmod = z_b * np.exp(
                1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid
            )

            if sg.SS_VERSE_B1:
                if dbg:
                    print("Versing RF with B1 minimization...  ")
                # Additional VERSE with B1 restriction, maintaining duration
                # account for scaling by the largest possible spectral weightings at
                # each spatial sample
                z_rfmax = np.max(np.abs(z_rfm))

                z_rfvmax1 = ss_verse(g2, z_rfmax)
                z_rfvmax, g2v = ss_b1verse(
                    g2,
                    z_rfvmax1,
                    sg.SS_MAX_B1,
                    sg.SS_MXG,
                    sg.SS_MXS,
                    sg.SS_TS,
                    sg.SS_GAMMA,
                    sg.SS_SLEW_PENALTY,
                    dbg,
                )

                if len(z_rfvmax) == 0:
                    # B1 condition cannot be met
                    rf = np.array([], dtype=np.complex128)
                    g = np.array([])
                    return rf, g

                # update gradient
                gpos = np.concatenate((g1, g2v, g3))
                if sg.SS_EQUAL_LOBES:
                    gneg = -gpos

            else:
                if dbg:
                    print("Versing RF...  ")

                if sg.SS_VERSE_FRAC == 0:
                    z_bvmod = z_bmod
                else:
                    z_bvmod = ss_verse(g2, z_bmod)

            # % Modulate back
            # %
            z_bv = z_bvmod * np.exp(
                -1j * 2 * np.pi * np.arange(z_np)[:, None] * sg.SS_TS * fpass_mid
            )
            z_bv = z_bv.squeeze()

            # % Build RF matrix
            # %
            nlobe = len(s_b)
            z_rfmv = s_rfm * (np.ones(nlobe)[:, None] * z_bv[None, :])

        # % Compile g, RF
        # %
        rf = np.atleast_1d(np.asarray([]))
        g = np.atleast_1d(np.asarray([]))

        nneg = len(gneg)

        for idx in range(nlobe):
            rf_lobe = z_rfmv[idx]
            rf = np.concatenate((rf, np.zeros(ng1), rf_lobe, np.zeros(ng3)))
            if idx < nlobe:
                rf = np.concatenate((rf, np.zeros(nneg)))
                g = np.concatenate((g, gpos, gneg))
            else:
                g = np.concatenate((g, gpos))

        # Offset RF
        nrf = len(rf)
        rf = rf * np.exp(-1j * 2 * np.pi * np.arange(nrf) * sg.SS_TS * f_off)

        # Convert amplitude to Gauss
        rf = rf / (2 * np.pi * sg.SS_GAMMA * sg.SS_TS)

        # Step 1 - get passbands
        fmid = (f[::2] + f[1::2]) / 2
        idx_pass = np.where(a > 0)[0][0]
        fpass = fmid[idx_pass]
        if np.isscalar(fpass):
            npass = 1
        else:
            npass = len(fpass)

        # Step 2 - get spatial sample points
        nz = 101
        dz = z_thk / (nz - 1)
        z = np.linspace(-z_thk / 2, z_thk / 2, nz)

        # Step 3 - get spatial profile
        gzrot = 2 * np.pi * sg.SS_GAMMA * sg.SS_TS * g
        gfrot = 2 * np.pi * sg.SS_TS * np.ones_like(g)
        rrot = 2 * np.pi * sg.SS_GAMMA * sg.SS_TS * rf

        if ptype == "ex":
            mxy = ab2ex(abr(rrot, gzrot + 1j * gfrot, z, fpass))
        elif ptype == "se":
            mxy = ab2se(abr(rrot, gzrot + 1j * gfrot, z, fpass))
        else:
            mxy = np.zeros((nz, npass))

        # Step 4 - find best fit to phase ramp
        zpass = z[:, None] * np.ones((1, npass))
        zpass = zpass.flatten(order="F")
        mxy_phs = np.unwrap(np.angle(mxy))
        mxy_phs_mid = mxy_phs[(nz + 1) // 2]
        mxy_phs -= np.ones(nz) * mxy_phs_mid
        mxy_phs = mxy_phs.flatten(order="F")

        p = np.polyfit(zpass, mxy_phs, 1)
        slope = p[0]

        # Step 5 - get area of gradient required
        cyc_per_cm = slope / (2 * np.pi)
        m0 = cyc_per_cm / sg.SS_GAMMA
        gref = grad_mintrap(m0, sg.SS_MXG, sg.SS_MXS, sg.SS_TS)

        rf = np.concatenate((rf, np.zeros(len(gref))))
        g = np.concatenate((g, gref))
    else:
        rf = np.array([], dtype=np.complex128)
        g = np.array([])

    return rf, g
