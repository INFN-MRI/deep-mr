"""EPI design."""

__all__ = ["ss_ep"]

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
from .ss_verse import ss_verse
from .ss_verse import ss_b1verse


def ss_ep(
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
    # global options
    if sg is None:
        sg = ss_globals()

    # Check if ss_type contains 'Half'
    sym_flag = 1 if "Half" in ss_type else 0

    # Alias the input parameters
    f_a, a_a, d_a, f_off = ss_alias(f, a, d, f_off, fs, sym_flag)

    if f_a.size == 0:
        raise ValueError("Strange: this frequency should be ok")

    # Calculate cycles/cm required
    kz_max = z_tb / z_thk  # cycles/cm
    kz_area = kz_max / sg.SS_GAMMA  # G/cm * s

    # Calculate SS bipolars
    nsamp = int(round(2 / (fs * sg.SS_TS)))
    gpos, gneg, g1, g2, g3 = grad_ss(
        kz_area, nsamp, sg.SS_VERSE_FRAC, sg.SS_MXG, sg.SS_MXS, sg.SS_TS, 1
    )
    ng1 = len(g1)
    ng2 = len(g2)
    ng3 = len(g3)

    # Determine max order that can be supported
    t_lobe = len(gpos) * sg.SS_TS
    max_lobe = int(sg.SS_MAX_DURATION / t_lobe)

    # Prepare amplitude description
    a_dup = np.zeros(f_a.shape)
    a_dup[0::2] = a_a
    a_dup[1::2] = a_a

    # Call fir filter design based on spectral factorization and convex optimization
    if sg.SS_MIN_ORDER:
        use_max = 0
    else:
        use_max = 1

    if s_ftype == "min":
        s_b, status = fir_minphase_power(max_lobe, f_a, a_dup, d_a, use_max, dbg)
    elif s_ftype == "max":
        s_b, status = fir_minphase_power(max_lobe, f_a, a_dup, d_a, use_max, dbg)
        s_b = np.conj(s_b[::-1])
    elif s_ftype == "lin":
        if use_max:
            if max_lobe % 2 == 0:
                max_lobe += 1
            s_b, status = fir_qprog(max_lobe, f_a, a_dup, d_a, dbg)
        else:
            odd_or_even = 0
            s_b, status = fir_min_order_qprog(
                max_lobe, f_a, a_dup, d_a, odd_or_even, dbg
            )

    if status == "Solved":
        # Get Z RF pulse
        z_np = len(g2)
        z_b = dzbeta(z_np, z_tb, "st", z_ftype, z_d[0], z_d[1])

        if sg.SS_SLR_FLAG == 1:
            oversamp = 4
            nZ = oversamp * z_np
            nZ2 = 2 ** int(np.ceil(np.log2(nZ)))
            Z_b = fftf(z_b, nZ2)  # column transform, unit magnitude

            if sg.SS_SPECT_CORRECT_FLAG:
                print("Spectral Correction not supported for EP pulses with SLR")

            if dbg:
                print("Doing SLR in F...")

            s_rfm = np.zeros((nZ2, 0), dtype=complex)
            bsf = np.sin(ang / 2) * Z_b

            for idx in range(nZ2):
                tmp_s_rf = b2rf(bsf[idx] * s_b)
                s_rfm = np.hstack((s_rfm, tmp_s_rf))

            z_bm = fftr(np.sin(s_rfm / 2), z_np, 2)
            z_rfm = np.zeros((0, z_np), dtype=complex)

            if dbg:
                print("Doing SLR in Z...")

            for idx in range(z_bm.shape[0]):
                tmp_z_rf = np.conj(b2rf(z_bm[idx, :]))
                z_rfm = np.vstack((z_rfm, tmp_z_rf.flatten()))

            pass_idx = np.where(a > 0)[0][0]
            fpass = [f[2 * pass_idx - 1], f[2 * pass_idx]]
            fpass_mid = np.mean(fpass) - f_off
            nlobe = len(s_b)
            rfmod = np.exp(1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid)

            if sg.SS_VERSE_B1:
                if dbg:
                    print("Versing RF with B1 minimization...")

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

                if z_rfvmax.size == 0:
                    rf = np.array([], dtype=complex)
                    g = np.array([], dtype=float)
                    return rf, g

                z_rfmv = []
                for idx in range(nlobe):
                    if idx % 2 == 1:
                        z_rfmod = z_rfm[idx, :] * rfmod
                        z_rfvmod = ss_verse(g2v, z_rfmod)
                    else:
                        z_rfmod = z_rfm[idx, ::-1] * rfmod
                        z_rfvmod = ss_verse(g2v[::-1], z_rfmod)

                    z_rfv = z_rfvmod.flatten() * np.conj(rfmod)
                    z_rfmv.append(z_rfv)
                z_rfmv = np.stack(z_rfmv, axis=0)

                gpos = np.concatenate((g1, g2v, g3))
                gneg = np.concatenate((-g1, -g2v[::-1], -g3))
            else:
                if dbg:
                    print("Versing RF...")

                for idx in range(nlobe):
                    if idx % 2 == 1:
                        z_rfmod = z_rfm[idx, :] * rfmod
                    else:
                        z_rfmod = z_rfm[idx, ::-1] * rfmod

                    if sg.SS_VERSE_FRAC == 0:
                        z_rfvmod = z_rfmod
                    else:
                        z_rfvmod = ss_verse(g2, z_rfmod)

                    z_rfv = z_rfvmod.flatten() * np.conj(rfmod)
                    z_rfmv[idx, :] = z_rfv
        else:  # No SLR
            # Nper = len(gpos)
            Noff = np.arange(-(ng2 - 1) / 2, (ng2 - 1) / 2 + 1)
            bsf = np.sin(ang / 2) * np.ones(len(Noff))

            if sg.SS_SPECT_CORRECT_FLAG:
                print("Spectral Correction not supported for EP pulses with SLR")

            s_rfm = np.conj(s_b) * np.ones((len(Noff), ng2))
        fpass_mid = 0
        z_bmod_for = z_b * np.exp(
            1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid
        )
        z_b_rev = np.flip(z_b)
        z_bmod_rev = z_b_rev * np.exp(
            1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid
        )

        if sg.SS_VERSE_B1:
            if dbg:
                print("Versing RF with B1 minimization...")

            b1max_sc = np.max(np.abs(s_rfm), axis=0)
            z_bvmod1 = ss_verse(g2, z_bmod_for)
            z_bvmod_for, g2v_for = ss_b1verse(
                g2,
                z_bvmod1,
                sg.SS_MAX_B1 / b1max_sc,
                sg.SS_MXG,
                sg.SS_MXS,
                sg.SS_TS,
                sg.SS_GAMMA,
                sg.SS_SLEW_PENALTY,
                dbg,
            )

            z_bvmod1 = ss_verse(g2, z_bmod_rev)
            z_bvmod_rev, g2v_rev = ss_b1verse(
                g2,
                z_bvmod1,
                sg.SS_MAX_B1 / np.flip(b1max_sc),
                sg.SS_MXG,
                sg.SS_MXS,
                sg.SS_TS,
                sg.SS_GAMMA,
                sg.SS_SLEW_PENALTY,
                dbg,
            )

            if z_bvmod_for.size == 0 or z_bvmod_rev.size == 0:
                rf = np.array([], dtype=complex)
                g = np.array([], dtype=float)
                return rf, g

            g2v = np.minimum(g2v_for, np.flip(g2v_rev))

            z_bvmod_for = ss_verse(g2v, z_bmod_for)
            z_bv_for = z_bvmod_for * np.exp(
                -1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid
            )
            z_bv_rev = z_bv_for[::-1]

            gpos = np.concatenate((g1, g2v, g3))
            gneg = np.concatenate((-g1, -g2v[::-1], -g3))
        else:
            z_bvmod_for = ss_verse(g2, z_bmod_for)
            z_bv_for = z_bvmod_for * np.exp(
                -1j * 2 * np.pi * np.arange(z_np) * sg.SS_TS * fpass_mid
            )
            z_bv_rev = z_bv_for[::-1]

        nlobe = len(s_b)
        z_rfmv = np.zeros((nlobe, z_np), dtype=complex)

        for idx in range(nlobe):
            if idx % 2 == 1:
                z_rfmv[idx, :] = s_rfm[idx, :] * z_bv_for
            else:
                z_rfmv[idx, :] = s_rfm[idx, ::-1] * z_bv_rev
    else:
        rf = np.array([], dtype=complex)
        g = np.array([], dtype=float)
        return rf, g

    # Compile g, RF
    rf = np.array([], dtype=complex)
    g = np.array([], dtype=float)

    for idx in range(nlobe):
        rf_lobe = z_rfmv[idx, :]
        rf = np.hstack((rf, np.zeros(ng1), rf_lobe, np.zeros(ng3)))

        if idx % 2 == 1:
            g = np.hstack((g, gpos))
        else:
            g = np.hstack((g, gneg))

    nrf = len(rf)
    rf = rf * np.exp(-1j * 2 * np.pi * np.arange(nrf) * sg.SS_TS * f_off)
    rf = rf / (2 * np.pi * sg.SS_GAMMA * sg.SS_TS)

    # Calculate refocusing lobe
    if ptype == "ex" or ptype == "se":
        fmid = (f[0::2] + f[1::2]) / 2
        idx_pass = np.where(a > 0)[0]
        fpass = fmid[idx_pass]
        npass = len(fpass)
        nz = 101
        dz = z_thk / (nz - 1)
        z = np.arange(-z_thk / 2, z_thk / 2 + dz, dz)
        gzrot = 2 * np.pi * sg.SS_GAMMA * sg.SS_TS * g
        gfrot = 2 * np.pi * sg.SS_TS * np.ones(len(g))
        rrot = 2 * np.pi * sg.SS_GAMMA * sg.SS_TS * rf

        if ptype == "ex":
            mxy = ab2ex(abr(rrot, gzrot + 1j * gfrot, z, fpass))
        elif ptype == "se":
            mxy = ab2se(abr(rrot, gzrot + 1j * gfrot, z, fpass))
    else:
        return None, None

    zpass = np.kron(z, np.ones(npass))
    zpass = zpass.flatten()
    mxy_phs = np.unwrap(np.angle(mxy))
    mxy_phs_mid = mxy_phs[(nz + 1) // 2, :]
    mxy_phs = mxy_phs - np.outer(np.ones(nz), mxy_phs_mid)
    mxy_phs = mxy_phs.flatten()

    p = np.polyfit(zpass, mxy_phs, 1)
    slope = p[0]

    cyc_per_cm = slope / (2 * np.pi)
    m0 = cyc_per_cm / sg.SS_GAMMA
    gref = grad_mintrap(m0, sg.SS_MXG, sg.SS_MXS, sg.SS_TS)
    rf = np.hstack((rf, np.zeros(len(gref))))
    g = np.hstack((g, gref))

    return rf, g
