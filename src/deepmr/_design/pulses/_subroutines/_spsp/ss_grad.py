"""
"""
__all__ = ["grad_min_bridge", "grad_mintrap", "grad_ss"]

import numpy as np


def grad_ss(m0, n, f, mxg, mxs, ts, equal):
    m0 = abs(m0)  # Ensure m0 is positive

    # Check if n is even when equal lobes are specified
    if equal and n % 2 != 0:
        raise ValueError("Equal lobes specified, but n is not even")

    # Convert mxs to G/cm/s
    mxs_s = mxs * 1e3
    dg = mxs_s * ts  # Max delta in one sample

    # Use the grad_min_bridge function to calculate gradients for the bridge
    gp_tmp, g1_tmp, g2_tmp, g3_tmp = grad_min_bridge(m0, f, mxg, mxs, ts)

    if equal:
        gn_tmp = -gp_tmp
    else:
        m0_pos = np.sum(gp_tmp) * ts
        gn_tmp = grad_mintrap(-m0_pos, mxg, mxs, ts)

    # If the number of samples 'n' is not specified, use the computed gradients
    if n is None:
        gpos = gp_tmp
        g1 = g1_tmp
        g2 = g2_tmp
        g3 = g3_tmp
        gneg = gn_tmp
    else:
        if len(np.concatenate((gp_tmp, gn_tmp))) > n:
            raise ValueError("Solution not obtained in specified number of samples")

        # Save the known solution
        gp_save = gp_tmp
        # g1_save = g1_tmp
        # g2_save = g2_tmp
        # g3_save = g3_tmp
        gn_save = -gp_save
        nb_save = np.where(np.diff(gp_save) == 0)[0][0]

        # Decrease the number of ramp samples in the positive lobe until 'n' is exceeded
        spec_met = True
        while spec_met and nb_save > 1:
            nb = nb_save - 1
            nc = max(1, int(np.ceil(nb * f)))
            a_ramps = (2 * nb - nc + 1) * nc * dg * ts
            a_const = m0 - a_ramps
            na = max(0, int(np.ceil(a_const / (nb * dg * ts))))
            dg_test = m0 / (((2 * nb - nc + 1) * nc + nb * na) * ts)
            A = nb * dg_test
            if A > mxg or dg_test > dg:
                spec_met = False
                continue
            g1 = np.arange(1, nb - nc + 1) * dg_test
            g2 = (
                np.concatenate(
                    (
                        np.arange(nb - nc + 1, nb + 1),
                        np.tile(nb, na),
                        np.arange(nb, nb - nc, -1),
                    )
                )
                * dg_test
            )
            g3 = np.arange(nb - nc, -1, -1) * dg_test
            gp = np.concatenate((g1, g2, g3))
            if abs(np.sum(g2) * ts - m0) > 10 * np.finfo(float).eps:
                raise ValueError("Area not calculated correctly")
            if equal:
                gn = -gp
            else:
                gn = grad_mintrap(-np.sum(gp) * ts, mxg, mxs, ts)
            if len(np.concatenate((gp, gn))) < n:
                spec_met = True
                gp_save = gp
                # g1_save = g1
                # g2_save = g2
                # g3_save = g3
                gn_save = gn
                nb_save = nb
            else:
                spec_met = False

        # Adjust the result to have exactly 'n' samples
        if not equal:
            na = n - len(gn_save) - 2 * nb_save - 1
        else:
            na = (n - 2 * (2 * nb_save + 1)) // 2
        nb = nb_save
        nc = max(1, int(np.ceil(nb * f)))
        dg_test = m0 / (((2 * nb - nc + 1) * nc + nb * na) * ts)
        A = nb * dg_test
        if A >= 1.001 * mxg or dg_test > 1.001 * dg:
            raise ValueError("Amplitude/Slew rate is exceeded")
        g1 = np.arange(1, nb - nc + 1) * dg_test
        g2 = (
            np.concatenate(
                (
                    np.arange(nb - nc + 1, nb + 1),
                    np.tile(nb, na),
                    np.arange(nb, nb - nc, -1),
                )
            )
            * dg_test
        )
        g3 = np.arange(nb - nc, -1, -1) * dg_test
        gpos = np.concatenate((g1, g2, g3))
        if abs(np.sum(g2) * ts - m0) > 10 * np.finfo(float).eps:
            raise ValueError("Area not calculated correctly")
        if not equal:
            ratio = -np.sum(gn_save) / np.sum(gpos)
            if ratio < 1 - 10 * np.finfo(float).eps:
                pass  # Warning: Improbable ratio
            gneg = gn_save / ratio
        else:
            gneg = -gpos

    return gpos, gneg, g1, g2, g3


def grad_mintrap(m0, mxg, mxs, ts):
    if m0 < 0:
        s = -1
        m0 = -m0
    else:
        s = 1

    # Convert mxs to G/cm/s
    mxs = mxs * 1e3

    # Determine trapezoid parameters
    dg = mxs * ts  # Max delta in one sample
    nb = int(np.ceil(np.sqrt(m0 / dg / ts)))
    A = m0 / (nb * ts)

    if A <= mxg:
        na = 0
        dg_act = A / nb
    else:
        nb = int(np.ceil(mxg / dg))
        dg_act = mxg / nb
        na = int(np.ceil((m0 - (nb**2 * dg_act * ts)) / (mxg * ts)))
        dg_act = m0 / (nb**2 + na * nb) / ts
        A = nb * dg_act

    # Construct discrete trapezoid --- always end with a zero value
    g = s * np.concatenate(
        (
            np.arange(1, nb + 1) * dg_act,
            np.ones(na) * A,
            np.arange(nb - 1, -1, -1) * dg_act,
        )
    )

    return g


def grad_min_bridge(m0, f, mxg, mxs, ts):
    if m0 < 0:
        s = -1
        m0 = -m0
    else:
        s = 1

    # Convert mxs to G/cm/s
    mxs = mxs * 1e3

    # Determine trapezoid parameters
    dg = mxs * ts  # Max delta in one sample

    # Assume triangle at first and see if max amp requirements met
    if f != 0:
        aq = (2 - f) * f
        bq = f
        cq = -m0 / (dg * ts)
        nb = (-bq + np.sqrt(bq**2 - 4 * aq * cq)) / (2 * aq)
        nb = int(np.ceil(nb))
        nc = max(1, int(np.ceil(nb * f)))
    else:
        A = m0 / (2 * ts)
        nb = int(np.ceil(A / dg))
        nc = 1

    # Test result
    dg_test = m0 / ((2 * nb - nc + 1) * nc * ts)
    A = nb * dg_test

    if A <= mxg and dg_test < dg:  # This works!
        g1 = s * np.arange(1, nb - nc + 1) * dg_test
        g2 = (
            s
            * np.concatenate(
                (np.arange(nb - nc + 1, nb + 1), np.arange(nb, nb - nc, -1))
            )
            * dg_test
        )
        g3 = s * np.arange(nb - nc, -1, -1) * dg_test
        g = np.concatenate((g1, g2, g3))
        if abs((np.sum(g2) * ts) - s * m0) > 10 * np.finfo(float).eps:
            print("Area Spec:", m0, "Actual:", np.sum(g) * ts)
            raise ValueError("grad_min_bridge: Area not calculated correctly")
    else:  # Must be trapezoid
        # Subtract area of ramps
        nb = int(np.ceil(mxg / dg))
        nc = max(1, int(np.ceil(nb * f)))
        dg_test = mxg / nb
        a_ramps = (2 * nb - nc + 1) * nc * dg_test * ts

        # Get number of constant samples
        a_const = m0 - a_ramps
        na = int(np.ceil(a_const / ts / mxg))

        # Get correct amplitude now
        dg_test = m0 / (((2 * nb - nc + 1) * nc + nb * na) * ts)
        A = nb * dg_test
        if A > mxg or dg_test > dg:
            raise ValueError("Amp/Slew being exceeded")
        g1 = s * np.arange(1, nb - nc + 1) * dg_test
        g2 = (
            s
            * np.concatenate(
                (
                    np.arange(nb - nc + 1, nb + 1),
                    np.tile(nb, na),
                    np.arange(nb, nb - nc, -1),
                )
            )
            * dg_test
        )
        g3 = s * np.arange(nb - nc, -1, -1) * dg_test
        g = np.concatenate((g1, g2, g3))
        if abs((np.sum(g2) * ts) - s * m0) > 10 * np.finfo(float).eps:
            raise ValueError("grad_min_bridge: Area not calculated correctly")

    return g, g1, g2, g3
