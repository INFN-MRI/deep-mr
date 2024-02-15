"""Filter creation subroutines."""

__all__ = ["fir_minphase_power", "fir_qprog", "fir_min_order_qprog"]

from types import SimpleNamespace

import warnings
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import linprog
from scipy.signal import freqz


from .ss_fourier import fftf
from .ss_fourier import fftr


def fir_minphase_power(n, f, a, d, use_max=0, dbg=0):
    if use_max is None:
        use_max = 0

    if dbg is None:
        dbg = 1

    hn = []
    status = "Failed"

    d2 = np.stack([d, d], axis=1)
    d2 = d2.ravel()
    mxspec = (a + d2) ** 2
    mnspec = np.maximum(0, (a - d2)) ** 2

    ztol_perc = 2
    ztol = min(ztol_perc / 100 * (mxspec - mnspec))

    mnspec = np.maximum(ztol, mnspec)
    a_sqr = (mxspec + mnspec) / 2
    d2_sqr = (mxspec - mnspec) / 2
    d_sqr = d2_sqr[0::2]
    n_max = 2 * n - 1

    odd_only = 1
    if not use_max:
        r, status = fir_min_order(n_max, f, a_sqr, d_sqr, odd_only, ztol, dbg)
    else:
        r, status = fir_pm(n_max, f, a_sqr, d_sqr, ztol, dbg)

    if status == "Failed":
        if dbg:
            print("Failed to get filter")
        return hn, status

    Rok = False
    ncurrent = len(r)

    while not Rok:
        rn, status = fir_pm_minpow(ncurrent, f, a_sqr, d_sqr, ztol, dbg)

        oversamp = 15
        m = 2 * oversamp * len(rn)
        m2 = 2 ** (np.ceil(np.log2(m)))
        R = np.real(fftf(rn[:, None], int(m2))).squeeze()

        if np.any(R < 0):
            min_stop = np.min(a_sqr + d2_sqr)
            Rtol_perc = 0.1
            Rtol = Rtol_perc * min_stop

            if np.min(R) < -Rtol:
                if dbg:
                    print("Autocorrelation has negative value")
                    print(
                        f"Tol ({int(Rtol_perc * 100)}% stopband): {Rtol}  Actual: {-min(R)}"
                    )

                # test spectral factorization
                rn = rn + Rtol
                hn = spectral_fact(rn)
                hn = np.conj(hn[::-1])

                # Get squared frequency response and check
                # against specs
                H = np.abs(fftf(hn[:, None], int(m2))).squeeze()
                freq = np.arange(-m2 / 2, m2 / 2) / m2 * 2

                H2 = abs(H) ** 2
                nband = len(f) // 2
                atol = 0.05
                Rok = True
                for band in range(nband):
                    idx = np.where((freq >= f[2 * band]) & (freq <= f[2 * band + 1]))[0]
                    amax = (1 + atol) * (a_sqr[2 * band] + d_sqr[band] + Rtol)
                    amin = (1 - atol) * (a_sqr[2 * band] - d_sqr[band])
                    fail = np.where(np.logical_or(H2[idx] > amax, H2[idx] < amin))[0]
                    if fail.size > 0:
                        if dbg:
                            print("Spectral factorization doesn" "t meet specs")
                            print(f"Increase number of taps to: {len(r) + 2}")
                            print("\r          \r")
                        ncurrent = ncurrent + 2
                        Rok = 0
                        break
            else:
                if dbg:
                    print("Autocorrelation has negative value, but within tol")
                    print(
                        f"Tol ({int(Rtol_perc * 100)}% stopband): {Rtol}  Actual: {-min(R)}"
                    )

                rn = rn - min(R)
                Rok = 1
        elif status == "Failed":
            if dbg:
                print("\r          \r")
            ncurrent = ncurrent + 2
            Rok = False
        else:
            if dbg:
                print("Autocorrelation OK")
            Rok = True

    rn = np.asarray(rn)
    hn = spectral_fact(rn)
    hn = np.conj(hn[::-1])

    return hn, status


def fir_qprog(n, f, a, d, dbg=0):
    # Determine if real or complex coefficients
    f = np.array(f) * np.pi  # Scale to +/- pi
    if np.min(f) < 0:
        real_filter = False
    else:
        real_filter = True

    # Determine if filter has odd or even number of taps
    odd_filter = n % 2 == 1

    # If the frequency specification has a non-zero point at +/- pi,
    # then the order must be even. A warning is printed and a failure is returned if this is the case.
    if not odd_filter:
        idx = np.where(np.abs(f) == np.pi)
        if np.any(a[idx] == 1):
            if dbg:
                print("n odd and frequency spec 1 at fs/2")
            status = "Failed"
            h = np.array([])
            return h, status

    # Determine the number of optimization parameters
    nhalf = np.ceil(n / 2)
    nx = nhalf
    if not real_filter:
        if odd_filter:
            nx = 2 * nhalf - 1
        else:
            nx = 2 * nhalf

    # Create optimization arrays
    oversamp = 15
    undersamp_tran = 1  # Undersampling factor for transition regions

    # Get first pass on w
    if real_filter:
        m = oversamp * n
        w = np.linspace(0, np.pi, m)
    else:
        m = 2 * oversamp * n
        w = np.linspace(-np.pi, np.pi, m)

    # Add explicit samples to w at the edge of each specified band
    w = np.sort(np.concatenate((w, f)))

    # Find indices to passbands/stopbands, and fill in upper/lower bounds
    idx_band = []
    U_band = []
    L_band = []
    nband = len(f) // 2

    for band in range(nband):
        idx = np.where((w >= f[2 * band]) & (w <= f[2 * band + 1]))
        idx_band.extend(idx[0])

        amp = np.interp(
            w[idx], [f[2 * band], f[2 * band + 1]], [a[2 * band], a[2 * band + 1]]
        )
        U_band.extend(amp + d[band])
        L_band.extend(amp - d[band])

    # Get transition indices
    idx_tmp = np.ones(len(w), dtype=int)
    idx_tmp[idx_band] = 0
    idx_tran = np.where(idx_tmp == 1)[0]

    # Get average representation of response
    lb_resp = np.empty_like(w)
    lb_resp[idx_band] = (np.array(U_band) + np.array(L_band)) / 2
    lb_resp[idx_tran] = (np.max(U_band) + np.min(L_band)) / 2

    if real_filter:
        lb_resp = np.concatenate((lb_resp[::-1], lb_resp[1:-1]))

    # Decimate w in transition regions
    idx_tran = idx_tran[0::undersamp_tran]

    # Add transition band limits to be between the + max specification on each band and min of (0, min(L_band))
    if idx_tran.size > 0:
        U_amp_tran = np.max(U_band)
        U_tran = np.full(idx_tran.shape, U_amp_tran)
        L_amp_tran = min(0, np.min(L_band))
        L_tran = np.full(idx_tran.shape, L_amp_tran)
    else:
        U_tran = np.array([])
        L_tran = np.array([])

    # Update w, idx_band
    wband = w[idx_band]
    idx_band = np.arange(len(wband))
    wtran = w[idx_tran]
    idx_tran = np.arange(len(wtran)) + len(wband)
    w = np.concatenate((wband, wtran))
    m = w.size

    if real_filter:
        # Create optimization matrices
        # A is the matrix used to compute the power spectrum
        # A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(n*w)]
        Acos = np.zeros((m, nhalf))
        if odd_filter:
            Acos[:, 0] = 1
            Acos[:, 1:] = 2 * np.cos(np.outer(w, np.arange(1, nhalf)))
        else:
            Acos = 2 * np.cos(np.outer(w, np.arange(0.5, nhalf)))

        Asin = np.zeros((m, nhalf))
    else:
        if odd_filter:
            Acos = np.zeros((m, nhalf))
            Acos[:, 0] = 1
            Acos[:, 1:] = 2 * np.cos(np.outer(w, np.arange(1, nhalf)))
            Asin = 2 * np.sin(np.outer(w, np.arange(1, nhalf)))
        else:
            Acos = 2 * np.cos(np.outer(w, np.arange(0.5, nhalf)))
            Asin = 2 * np.sin(np.outer(w, np.arange(0.5, nhalf)))

    # Get subset of A matrix for the current order
    A = np.concatenate((Acos, Asin), axis=1)

    # Build matrix for upper bound constraints
    A_U = np.concatenate((A[idx_band], A[idx_tran]))
    U_b = np.concatenate((U_band, U_tran))

    # Build matrices for lower bound constraints
    A_L = np.concatenate((A[idx_band], A[idx_tran]))
    L_b = np.concatenate((L_band, L_tran))

    # Combine matrices
    A_b = np.concatenate((A_U, -A_L), axis=0)
    b = np.concatenate((U_b, -L_b))

    # Set H to minimize the total energy in the filter
    # H = np.eye(nx)
    fmin = np.zeros(nx)

    # Call the minimization routine
    x0 = None
    if real_filter:
        res = linprog(
            c=fmin,
            A_ub=A_b,
            b_ub=b,
            bounds=[(-np.inf, np.inf)] * nx,
            x0=x0,
            method="interior-point",
            options={"disp": False},
        )
    else:
        res = linprog(
            c=fmin,
            A_ub=A_b,
            b_ub=b,
            bounds=[(-np.inf, np.inf)] * nx,
            x0=x0,
            method="interior-point",
            options={"disp": False, "tol": 1e-8},
        )

    if res.status == 1:  # Feasible
        h = fill_h(res.x, nhalf, real_filter, odd_filter)
        status = "Solved"
    else:
        h = np.array([])
        status = "Failed"

    return h, status


def fir_min_order_qprog(n, f, a, d, even_odd=0, dbg=0):
    if even_odd not in [1, 2]:
        even_odd = 0

    # Initialize best odd/even filters
    hbest_odd = []
    hbest_even = []

    # Get max odd/even num taps
    n_odd_max = 2 * (n - 1) // 2 + 1
    n_even_max = 2 * (n // 2)

    if dbg >= 2:
        # filt_fig = figure()
        pass

    # Test odd filters first
    if even_odd != 2:
        n_bot = 1
        n_top = (n_odd_max + 1) // 2
        n_cur = n_top

        if dbg:
            print("Testing odd length filters...")

        while n_top - n_bot > 1:
            n_tap = n_cur * 2 - 1

            if dbg:
                print(f"{n_tap:4d} taps: ...", end="")

            h, status = fir_qprog(n_tap, f, a, d, dbg)

            if status == "Solved":
                # feasible
                hbest_odd = h

                if dbg:
                    print("Feasible")

                if dbg >= 2:
                    # figure(filt_fig)
                    pass

                n_top = n_cur

                if n_top == n_bot + 1:
                    n_cur = n_bot
                else:
                    n_cur = (n_top + n_bot) // 2
            else:
                if dbg:
                    print("Infeasible")

                n_bot = n_cur
                n_cur = (n_bot + n_top) // 2

    # Test even filters now
    if even_odd != 1:
        n_bot = 1

        if not hbest_odd:
            n_top = n_even_max // 2
            n_cur = n_top
        else:
            n_top = min(n_even_max // 2, (len(hbest_odd) + 1) // 2)
            n_cur = n_top

        if dbg:
            print("Testing even length filters...")

        while n_top - n_bot > 1:
            n_tap = n_cur * 2

            if dbg:
                print(f"{n_tap:4d} taps: ...", end="")

            h, status = fir_qprog(n_tap, f, a, d, dbg)

            if status == "Solved":
                # feasible
                hbest_even = h

                if dbg:
                    print("Feasible")

                if dbg >= 2:
                    # figure(filt_fig)
                    pass

                n_top = n_cur

                if n_top == n_bot + 1:
                    n_cur = n_bot
                else:
                    n_cur = (n_top + n_bot) // 2
            else:
                if dbg:
                    print("Infeasible")

                n_bot = n_cur
                n_cur = (n_bot + n_top) // 2

    if not hbest_odd and not hbest_even:
        status = "Failed"
        h = []

        if dbg:
            print("\nFailed to achieve specs")
    else:
        status = "Solved"

        if not hbest_odd:
            h = hbest_even
        elif not hbest_even:
            h = hbest_odd
        elif len(hbest_odd) < len(hbest_even):
            h = hbest_odd
        else:
            h = hbest_even

        if dbg:
            print(f"\nOptimum number of filter taps is: {len(h)}.")

    return h, status


# %% local utils
def spectral_fact(r):
    # Length of the impulse response sequence
    nr = len(r)
    n = (nr + 1) // 2

    # Over-sampling factor
    mult_factor = 30  # Should have mult_factor * n >> n
    m = mult_factor * n

    # Computation method:
    # H(exp(jTw)) = alpha(w) + j*phi(w)
    # where alpha(w) = 1/2 * ln(R(w)) and phi(w) = Hilbert_trans(alpha(w))

    # Compute 1/2 * ln(R(w))
    w = 2 * np.pi * np.arange(m) / m
    R = np.exp(-1j * np.kron(w[:, None], -np.arange(-(n - 1), n)[None, :])) @ r
    R = np.abs(np.real(R))  # Remove numerical noise from the imaginary part
    alpha = 1 / 2 * np.log(R)

    # Find the Hilbert transform
    alphatmp = np.fft.fft(alpha)
    alphatmp[m // 2 :] = -alphatmp[m // 2 :]
    alphatmp[0] = 0
    alphatmp[m // 2] = 0
    phi = np.real(np.fft.ifft(1j * alphatmp))

    # Retrieve the original sampling
    index = np.where(np.arange(m) % mult_factor == 0)
    alpha1 = alpha[index]
    phi1 = phi[index]

    # Compute the impulse response (inverse Fourier transform)
    h = np.fft.ifft(np.exp(alpha1 + 1j * phi1), n)

    return h


def fir_min_order(n, f, a, d, even_odd=0, a_min=None, dbg=0):
    if even_odd not in (1, 2):
        even_odd = 0

    if a_min is None:
        a_min = min(0, np.min(a - d))

    n_odd_max = 2 * (n // 2) + 1
    n_even_max = 2 * (n // 2)

    hbest_odd = []
    hbest_even = []

    if dbg >= 2:
        filt_fig = None

    n_bot = 1
    n_top = (n_odd_max + 1) // 2
    n_cur = n_top

    if dbg:
        print("Testing odd length filters...")

    while n_top - n_bot > 1:
        n_tap = n_cur * 2 - 1
        if dbg:
            print(f"{n_tap} taps: ...", end="")

        h, status = fir_pm(n_tap, f, a, d, a_min, dbg)

        if status == "Solved":
            hbest_odd = h
            if dbg:
                print("Feasible")

            n_top = n_cur

            if n_top == n_bot + 1:
                n_cur = n_bot
            else:
                n_cur = (n_top + n_bot) // 2
        else:
            if dbg:
                print("Infeasible")
            n_bot = n_cur
            n_cur = (n_bot + n_top) // 2

    if even_odd != 1:
        n_bot = 1
        n_top = n_even_max // 2

        if hbest_odd is not None:
            n_top = min(n_even_max // 2, (len(hbest_odd) + 1) // 2)

        n_cur = n_top

        if dbg:
            print("Testing even length filters...")

        while n_top - n_bot > 1:
            n_tap = n_cur * 2

            if dbg:
                print(f"{n_tap} taps: ...", end="")

            h, status = fir_pm(n_tap, f, a, d, a_min, dbg)

            if status == "Solved":
                hbest_even = h

                if dbg:
                    print("Feasible")
                n_top = n_cur

                if n_top == n_bot + 1:
                    n_cur = n_bot
                else:
                    n_cur = (n_top + n_bot) // 2
            else:
                if dbg:
                    print("Infeasible")
                n_bot = n_cur
                n_cur = (n_bot + n_top) // 2

    if len(hbest_odd) == 0 and len(hbest_even) == 0:
        status = "Failed"
        h = []

        if dbg:
            print("\nFailed to achieve specs")
    else:
        status = "Solved"
        h = hbest_odd if len(hbest_odd) > len(hbest_even) else hbest_even

        if dbg:
            print(f"\nOptimum number of filter taps is: {len(h)}.")

    return h, status


def fir_pm(n, f, a, d, a_min=None, dbg=0):
    d2 = np.stack([d, d], axis=1)
    d2 = d2.ravel()

    if a_min is None:
        a_min = min(0, min(a - d2))

    if dbg == 0:
        dbg = 0

    # determine if real or complex coefficients
    f = f * np.pi
    real_filter = 1 if np.min(f) >= 0 else 0

    # determine if filter has odd or even number of taps
    odd_filter = 1 if n & 1 else 0

    # if the frequency specification has a non-zero point ad +/- 1, then order mut be even.
    # A warning is printed and a failure returned if this is the case
    if not odd_filter:
        idx = np.where(np.abs(f) != 0)
        if np.any(a[idx] != 0):
            warnings.warn("n odd and frequency spec non-zero at fs/2")
            status = "Failed"
            return None, status

    # oversampling on frequency to determine transition bands
    oversamp = 8

    # get first pass on w
    if real_filter:
        m = oversamp * n
        w = np.linspace(0, np.pi, m)
    else:
        m = 2 * oversamp * n
        w = np.linspace(-np.pi, np.pi, m)

    # find bounds on transition regions and convert to amp/ripple
    ub_tran = np.max(a + d2)
    lb_tran = a_min
    amp_tran = (ub_tran + lb_tran) / 2
    ripple_tran = (ub_tran - lb_tran) / 2

    # find indices of transition bands, build up new frequency spec
    nband = len(f) // 2
    ntran = nband + 1
    fn = []
    an = []
    dn = []

    for tran in range(1, ntran + 1):
        if tran == 1:
            f_l = np.min(w)  # avoid sample at -pi
            rband = tran
            f_r = f[2 * rband - 2]
        elif tran == ntran:
            lband = tran - 1
            f_l = f[2 * lband - 1]
            f_r = np.pi  # avoid sample at pi
        else:
            lband = tran - 1
            f_l = f[2 * lband - 1]
            rband = tran
            f_r = f[2 * rband - 2]

        idx_tran = np.where((w > f_l) & (w < f_r))[0]

        # cfirpm seems to choke sometimes -- I hypothesize this is because the transition edges are too close to the actual passbands,
        # so don't take the immediately adjacent points
        nskip = 1
        if len(idx_tran) <= 1 + 2 * nskip:
            f_tran = []
            a_tran = []
            d_tran = []
        else:
            idx_tran = idx_tran[1 + nskip : -nskip]
            f_tran = [np.min(w[idx_tran]), np.max(w[idx_tran])]
            a_tran = [amp_tran, amp_tran]
            d_tran = [ripple_tran]

        fn.extend(f_tran)
        an.extend(a_tran)
        dn.extend(d_tran)

        if tran < ntran:
            fn.extend([f[2 * tran - 2], f[2 * tran - 1]])
            an.extend([a[2 * tran - 2], a[2 * tran - 1]])
            dn.extend([d[tran - 1]])

    # determine error weigths, then call firpm
    w = np.max(dn) / dn
    lgrid = 31  # oversample, default 25

    try:
        fn = np.asarray(fn)
        an = np.asarray(an)
        w = np.asarray(w)
        h, d_opt, opt = cfirpm(n - 1, fn / np.pi, an, w, lgrid)
    except:
        h = []

    # check frequency response at extremal frequencies
    # that are within specified bands
    try:
        resp_ok = check_response(f / np.pi, a, d, opt.fgrid, np.abs(opt.H))
    except:
        resp_ok = False

    if not resp_ok:
        status = "Failed"
        h = []
    else:
        h = h.ravel(order="F")
        status = "Solved"

    return h, status


def fir_pm_minpow(n, f, a, d, a_min=None, dbg=0):
    d2 = np.stack([d, d], axis=1)
    d2 = d2.ravel()

    if a_min is None or a_min == []:
        a_min = min(0, np.min(a - d2))

    if len(f.shape) > 0:
        f = f * np.pi
        if np.min(f) < 0:
            real_filter = 0
        else:
            real_filter = 1

        if n % 2 == 1:
            odd_filter = 1
        else:
            odd_filter = 0

        if not odd_filter:
            idx = np.where(np.abs(f) != 0)
            if np.any(a[idx] != 0):
                print("Warning: n odd and frequency spec non-zero at fs/2")
                status = "Failed"
                h = []
                return h, status

        oversamp = 16
        if real_filter:
            m = oversamp * n
            w = np.linspace(0, np.pi, m)
        else:
            m = 2 * oversamp * n
            w = np.linspace(-np.pi, np.pi, m)

        ub_tran = np.max(a + d2)
        lb_tran = a_min
        amp_tran = (ub_tran + lb_tran) / 2
        ripple_tran = (ub_tran - lb_tran) / 2

        nband = len(f) // 2
        ntran = nband + 1

        fn = []
        an = []
        dn = []

        for tran in range(1, ntran + 1):
            if tran == 1:
                f_l = np.min(w)  # avoid sample at -pi
                rband = tran
                f_r = f[2 * rband - 2]
            elif tran == ntran:
                lband = tran - 1
                f_l = f[2 * lband - 1]
                f_r = np.pi  # avoid sample at pi
            else:
                lband = tran - 1
                f_l = f[2 * lband - 1]
                rband = tran
                f_r = f[2 * rband - 2]

            idx_tran = np.where((w > f_l) & (w < f_r))[0]
            nskip = 1
            if len(idx_tran) <= 1 + 2 * nskip:
                f_tran = []
                a_tran = []
                d_tran = []
            else:
                idx_tran = idx_tran[1 + nskip : -nskip]
                f_tran = [np.min(w[idx_tran]), np.max(w[idx_tran])]
                a_tran = [amp_tran, amp_tran]
                d_tran = [ripple_tran]

            fn.extend(f_tran)
            an.extend(a_tran)
            dn.extend(d_tran)

            if tran < ntran:
                fn.extend([f[2 * tran - 2], f[2 * tran - 1]])
                an.extend([a[2 * tran - 2], a[2 * tran - 1]])
                dn.extend([d[tran - 1]])

        wt = np.max(dn) / dn
        lgrid = 31

        try:
            fn = np.asarray(fn)
            an = np.asarray(an)
            w = np.asarray(w)
            h, d_opt, opt = cfirpm(n - 1, fn / np.pi, an, wt, lgrid)
        except:
            h = []

        resp_ok = 0
        if len(h) != 0:
            resp_ok = check_response(fn / np.pi, an, dn, opt.fgrid, np.abs(opt.H))
        if not resp_ok:
            status = "Failed"
            h = []
            return h, status

        if dbg:
            print("Getting linear filter based on PM design")

        hlin, _ = fir_linprog(n, f / np.pi, a, d, h, dbg)
        _, Hlin = freqz(hlin, 1, w / np.pi, fs=2.0)

        fn = []
        an = []
        dn = []

        for tran in range(1, ntran + 1):
            if tran == 1:
                f_l = np.min(w)  # avoid sample at -pi
                rband = tran
                f_r = f[2 * rband - 2]
            elif tran == ntran:
                lband = tran - 1
                f_l = f[2 * lband - 1]
                f_r = np.pi  # avoid sample at pi
            else:
                lband = tran - 1
                f_l = f[2 * lband - 1]
                rband = tran
                f_r = f[2 * rband - 2]

            idx_tran = np.where((w > f_l) & (w < f_r))[0]
            nskip = 1
            ntran_pts = oversamp * 2
            ntran_region = np.max([1, round((len(idx_tran) - nskip) / ntran_pts)])
            ntran_pts = (len(idx_tran) - nskip) / ntran_region
            idx_region_st = nskip + 1 + np.round(np.arange(0, ntran_region) * ntran_pts)
            idx_region_end = (
                np.concatenate([idx_region_st[1:], [len(idx_tran) - nskip + 1]]) - 1
            )

            for reg in range(len(idx_region_st)):
                idx_tran_reg = idx_tran[
                    int(idx_region_st[reg]) : int(idx_region_end[reg])
                ]

                if len(idx_tran_reg) == 0:
                    f_tran = []
                    a_tran = []
                    d_tran = []
                else:
                    f_tran = [np.min(w[idx_tran_reg]), np.max(w[idx_tran_reg])]
                    max_H = np.max(np.abs(Hlin[idx_tran_reg]))
                    tol_H = 0.01
                    max_H = max_H + tol_H * np.max(a + d2)
                    max_H = max(np.min(a + d2), max_H)
                    min_H = a_min
                    amp_tran = (max_H + min_H) / 2
                    ripple_tran = (max_H - min_H) / 2
                    a_tran = [amp_tran, amp_tran]
                    d_tran = [ripple_tran]

                fn.extend(f_tran)
                an.extend(a_tran)
                dn.extend(d_tran)

            if tran < ntran:
                fn.extend([f[2 * tran - 2], f[2 * tran - 1]])
                an.extend([a[2 * tran - 2], a[2 * tran - 1]])
                dn.extend([d[tran - 1]])

        wt = np.max(dn) / dn
        lgrid = 31

        try:
            fn = np.asarray(fn)
            an = np.asarray(an)
            w = np.asarray(w)
            h, d_opt, opt = cfirpm(n - 1, fn / np.pi, an, wt, lgrid)
        except:
            h = []

        resp_ok = 0
        if len(h) != 0:
            resp_ok = check_response(fn / np.pi, an, dn, opt.fgrid, np.abs(opt.H))

        if not resp_ok:
            print("*** Failed to get min energy pulse ***")
            status = "Failed"
            h = []
        else:
            status = "Solved"

    return h, status


def fir_linprog(n, f, a, d, h0=None, dbg=0):
    f = f * np.pi
    real_filter = 1 if min(f) >= 0 else 0
    odd_filter = 1 if n % 2 == 1 else 0

    if not odd_filter:
        idx = np.where(np.abs(f) != 0)
        if np.any(a[idx] != 0):
            print("n odd and frequency spec non-zero at fs/2")
            status = "Failed"
            h = []
            return

    nhalf = np.ceil(n / 2)
    nx = nhalf
    if not real_filter:
        if odd_filter:
            nx = 2 * nhalf - 1
        else:
            nx = 2 * nhalf

    oversamp = 15
    undersamp_tran = 1

    if real_filter:
        m = oversamp * n
        w = np.linspace(0, np.pi, m)
    else:
        m = 2 * oversamp * n
        w = np.linspace(-np.pi, np.pi, m)

    w = np.sort(np.append(w, f))

    idx_band = []
    U_band = []
    L_band = []
    nband = len(f) // 2

    for band in range(nband):
        idx = np.where((w >= f[band * 2]) & (w <= f[band * 2 + 1]))
        idx_band.extend(idx[0])
        if f[band * 2] == f[band * 2 + 1]:
            amp = a[band * 2]
        else:
            amp = a[band * 2] + (a[band * 2 + 1] - a[band * 2]) * (
                (w[idx] - f[band * 2]) / (f[band * 2 + 1] - f[band * 2])
            )
        U_band.extend(amp + d[band])
        L_band.extend(amp - d[band])

    U_band = np.asarray(U_band)
    L_band = np.asarray(L_band)

    idx_tmp = np.ones(len(w))
    idx_tmp[idx_band] = 0
    idx_tran = np.where(idx_tmp == 1)[0]

    lb_resp = np.zeros(len(w))
    lb_resp[idx_band] = (U_band + L_band) / 2
    lb_resp[idx_tran] = (max(U_band) + min(L_band)) / 2
    if real_filter:
        lb_resp = np.concatenate((lb_resp[::-1], lb_resp[1:-1]))

    if h0 is None:
        h0 = []

    # x0 = fill_opt_param(h0, int(nx), real_filter, odd_filter, lb_resp)
    idx_tran = idx_tran[::undersamp_tran]

    U_amp_tran = max(U_band)
    U_tran = U_amp_tran * np.ones(len(idx_tran))
    L_amp_tran = min(0, min(L_band))
    L_tran = L_amp_tran * np.ones(len(idx_tran))

    wband = w[idx_band]
    idx_band = np.arange(wband.shape[0])
    wtran = w[idx_tran]
    idx_tran = np.arange(wtran.shape[0]) + len(wband)
    w = np.concatenate((wband, wtran))
    m = w.shape[0]

    if real_filter:
        if odd_filter:
            Acos = np.concatenate(
                (
                    np.ones((m, 1)),
                    2 * np.cos(np.kron(w[:, None], np.arange(1, nhalf)[None, :])),
                ),
                axis=-1,
            )
        else:
            Acos = (
                2
                * np.cos(np.kron(w[:, None], np.arange(nhalf)[None, :] + 0.5))[:, None]
            )
        Asin = None
    else:
        if odd_filter:
            Acos = np.concatenate(
                (
                    np.ones((m, 1)),
                    2 * np.cos(np.kron(w[:, None], np.arange(1, nhalf)[None, :])),
                ),
                axis=-1,
            )
            Asin = (
                2 * np.sin(np.kron(w[:, None], np.arange(1, nhalf)[None, :]))[:, None]
            )
        else:
            Acos = (
                2
                * np.cos(np.kron(w[:, None], np.arange(nhalf)[None, :] + 0.5))[:, None]
            )
            Asin = (
                2
                * np.sin(np.kron(w[:, None], np.arange(nhalf)[None, :] + 0.5))[:, None]
            )

    if Asin is not None:
        A = np.concatenate((Acos, Asin), axis=-1)
    else:
        A = Acos

    A_U = np.concatenate((A[idx_band], A[idx_tran]), axis=0)
    U_b = np.concatenate((U_band, U_tran), axis=0)

    A_L = np.concatenate((A[idx_band], A[idx_tran]), axis=0)
    L_b = np.concatenate((L_band, L_tran), axis=0)

    A_b = np.concatenate((A_U, -A_L), axis=0)
    b = np.concatenate((U_b, -L_b), axis=0)

    fmin = np.sum(A[idx_tran], axis=0)

    options = {"disp": False}
    # x, fval, exitflag, output = linprog(fmin, A_ub=A_b, b_ub=b, options=options) # x0 is deprecated for modern scipy
    res = linprog(
        fmin, A_b, b, bounds=(None, None), options=options
    )  # x0 is deprecated for modern scipy

    # get output
    x = res.x
    exitflag = res.success

    if exitflag:
        h = fill_h(x, int(nhalf), real_filter, odd_filter)
        status = "Solved"
    else:
        h = []
        status = "Failed"

    return h, status


def fill_h(x, nhalf, real_filter, odd_filter):
    x = x.flatten()
    if real_filter:
        if odd_filter:
            h = x
            h = np.concatenate((x[-1:0:-1], h))
        else:
            h = x
            h = np.concatenate((x[-1::-1], h))
    else:
        if odd_filter:
            h = x[:nhalf] + 1j * np.concatenate(([0], x[nhalf:]))
            h = np.concatenate((np.conj(h[-1:0:-1]), h))
        else:
            h = x[:nhalf] + 1j * x[nhalf:]
            h = np.concatenate((np.conj(h[-1::-1]), h))
    return h


def fill_opt_param(h0, nx, real_filter, odd_filter, lb_resp):
    x0 = np.zeros(nx)

    if real_filter:
        nx_half = nx
    else:
        if odd_filter:
            nx_half = nx // 2
        else:
            nx_half = nx - 1 // 2

    fft_init = 0

    if len(h0) == 0:
        fft_init = 1
    else:
        nh = len(h0)
        nh_half = nh // 2
        if odd_filter and not nh % 2 or not odd_filter and nh % 2:
            fft_init = 1

    if fft_init:
        nh_half = nx_half
        if odd_filter:
            nh = 2 * nh_half + 1
        else:
            nh = 2 * nh_half
        h0 = fftr(np.hamming(len(lb_resp)) * lb_resp, nh)

    if odd_filter:
        if real_filter:
            x0[: min(nx_half, nh_half) + 1] = np.real(
                h0[nh_half : nh_half + min(nx_half, nh_half) + 1]
            )
        else:
            x0[: min(nx_half, nh_half) + 1] = np.real(
                h0[nh_half : nh_half + min(nx_half, nh_half) + 1]
            )
            x0[nx_half + 1 : nx_half + min(nx_half, nh_half) + 2] = np.imag(
                h0[nh_half + 1 : nh_half + min(nx_half, nh_half) + 2]
            )
    else:
        if real_filter:
            x0[: min(nx_half, nh_half) + 1] = np.real(
                h0[nh_half : nh_half + min(nx_half, nh_half) + 1]
            )
        else:
            x0[: min(nx_half, nh_half) + 1] = np.real(
                h0[nh_half : nh_half + min(nx_half, nh_half) + 1]
            )
            x0[nx_half + 1 : nx_half + min(nx_half, nh_half) + 2] = np.imag(
                h0[nh_half + 1 : nh_half + min(nx_half, nh_half) + 2]
            )
    return x0


def check_response(f, a, d, ftest, htest):
    nband = len(f) // 2
    status = 1

    for band in range(1, nband + 1):
        idx = np.where((ftest >= f[2 * band - 2]) & (ftest <= f[2 * band - 1]))[0]
        if len(idx) == 0:
            break

        f_off = ftest[idx] - f[2 * band - 2]
        a_test = a[2 * band - 2] + (a[2 * band - 1] - a[2 * band - 2]) * f_off / (
            f[2 * band - 1] - f[2 * band - 2]
        )
        a_hi = a_test + d[band - 1]
        a_lo = a_test - d[band - 1]

        if np.any((htest[idx] > a_hi) | (htest[idx] < a_lo)):
            status = 0
            return status

    return status


# %% MATLAB utils
def cfirpm(n, f, a, w=None, grid_density=25):
    """
    Adaptation of MATLAB cfirpm.

    Args:
        n (float): filter oreder.
        f (array[float]): vector of frequency band edges which must appear monotonically
            between -1 and +1, where 1 is the Nyquist frequency. The frequency
            bands span F(k) to F(k+1) for k odd; the intervals F(k+1) to F(k+2)
            for k odd are "transition bands" or "don't care" regions during
            optimization.
        a (array[float]): real vector the same size as f which specifies the desired
            amplitude of the frequency response of the resultant filter B. The
            desired response is the line connecting the points (F(k),A(k)) and
            (F(k+1),A(k+1)) for odd k.
        w (array[float], optional): vector of real, positive weights, one per band, for use during
            optimization.  W is optional; if not specified, it is set to unity.
        grid_density (array[int], optional): Density of frequency grid.
            The frequency grid has roughly 2**nextpow2(lgrid*n) frequency points.
            Defaults to 25.

    Returns:
        (array[float, complex]): output FIR filter of length N+1 which has the best approximation to the desired
            frequency response specified as (f, a).
        (float): maximum ripple height.
        (SimpleNamespace): optional results with the following structure:
            RES.fgrid: vector containing the frequency grid used in
                the filter design optimization.
            RES.des: desired response on fgrid.
            RES.wt: weights on fgrid.
            RES.H: actual frequency response on the grid.
            RES.error: error at each point on the frequency grid.
            RES.iextr: vector of indices into fgrid of extremal frequencies.
            RES.fextr: vector of extremal frequencies.

    Example:
        Design a 31-tap, complex multiband filter.
            b, _, _ = cfirpm(30, [-1 -.5 -.4 .7 .8 1], [0 0 1 2 0 0])

    """
    # naming
    M = n
    edges = f
    filt_str = a
    wgts = w

    # hardcoded
    h_sym = "unspecified"

    # Declare globals
    globvars = SimpleNamespace()
    globvars.DES_CRMZ = None
    globvars.WT_CRMZ = None
    globvars.GRID_CRMZ = None
    globvars.TGRID_CRMZ = None
    globvars.IFGRD_CRMZ = None

    if np.isscalar(filt_str):
        filt_str = np.asarray([filt_str], dtype=float)

    # cast to enforce Precision rules
    M = float(M)
    edges = np.array(edges, dtype=float)

    L = M + 1
    edges /= 2
    num_bands = len(edges) // 2

    # some quick checks on band edge vector
    if num_bands != int(num_bands):
        raise ValueError("Number of bands must be even.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Invalid frequency band edges.")

    # assign default parameter values
    if wgts is None:
        wgts = np.ones(num_bands)

    # determine symmetry options
    h_sym = h_sym.lower()
    if any(edges < 0):
        h_sym = "none"
    else:
        try:
            h_sym = multiband("defaults", [M, 2 * edges, None, wgts, a])
            if not isinstance(h_sym, str):
                h_sym = "none"
        except:
            print(
                "Symmetry not specified, and default value could not be determined. Using 'none'."
            )
            h_sym = "none"

    if h_sym in ["real", "none"]:
        fdomain = "whole"
        sym = 0
    else:
        fdomain = "half"
        sym = 1 if "even" in h_sym else 2

    # check domain before generating frequency grid
    if fdomain == "whole":
        # domain is [-1, +1) for user input, [-0.5, 0.5) internally
        if np.any(edges < -0.5) or np.any(edges > 0.5):
            raise ValueError(
                "Frequency band edges must be specified over the entire interval [-1, +1) for designs with SYM='real'."
            )
    else:
        # domain is [0, +1] for user input, [0, .5] internally
        if np.any(edges < 0) or np.any(edges > 0.5):
            raise ValueError(
                "Frequency band edges must be specified over the entire interval [0, 0.5] for designs with SYM='real'."
            )

    Lfft, indx_edges, _, globvars = eval_grid(
        globvars, edges, num_bands, M, L, fdomain, wgts, grid_density, [a]
    )

    # check for odd order with zero at f = +-0.5
    if M % 2 == 1:
        sk1 = np.where(np.abs(globvars.GRID_CRMZ - 0.5) < np.finfo(float).eps)[0]
        sk2 = np.where(np.abs(globvars.GRID_CRMZ + 0.5) < np.finfo(float).eps)[0]
        sk = np.concatenate((sk1, sk2))
        if np.any(np.abs(globvars.DES_CRMZ[sk]) > np.sqrt(np.finfo(float).eps)):
            warnings.warn(
                "Odd order FIR filters must have a gain of zero at +- the Nyquist frequency. The order is being increased by one."
            )
            M = M + 1
            L = M + 1
            Lfft, indx_edges, _, globvars = eval_grid(
                globvars, edges, num_bands, M, L, fdomain, wgts, grid_density, [a]
            )

    # check for zero at DC for odd-symmetric filters
    if "odd" in h_sym:
        sk = np.where(np.abs(globvars.GRID_CRMZ) < 1e-15)
        if np.any(np.abs(globvars.DES_CRMZ[sk]) > np.sqrt(1e-15)):
            raise ValueError("Odd-symmetric filters must have a gain of zero at DC.")

    if "real" in h_sym and (edges >= 0).all():
        # make DES and WT conjugate symmetric

        # crmz_grid moved the band edge grid points, so do the same
        # when constructing symmetric spectrum
        len_grid = len(globvars.TGRID_CRMZ)

        # if the DC term is included in the band edges, remove id
        if len(indx_edges) > 0 and indx_edges[0] == len_grid // 2 + 1:
            indx_edges = np.delete(indx_edges, 0)  # throw away DC point

        q = len_grid + 2 - np.flip(indx_edges, axis=0)
        globvars.TGRID_CRMZ[np.flip(indx_edges, axis=0)] = -globvars.TGRID_CRMZ[
            indx_edges
        ]

        # adjust other grid vectors accordingly
        indx_edges = np.concatenate((q, indx_edges), axis=0)
        globvars.IFGRD_CRMZ = np.concatenate(
            (len_grid + 2 - np.flip(globvars.IFGRD_CRMZ, axis=1), globvars.IFGRD_CRMZ),
            axis=1,
        )
        globvars.GRID_CRMZ = globvars.TGRID_CRMZ[globvars.IFGRD_CRMZ]

        # now, impose conjugate simmetry
        globvars.DES_CRMZ = np.concatenate(
            (np.conj(np.flip(globvars.DES_CRMZ), axis=0), globvars.DES_CRMZ), axis=0
        )
        globvars.WT_CRMZ = np.concatenate(
            (np.conj(np.flip(globvars.WT_CRMZ), axis=0), globvars.WT_CRMZ), axis=0
        )

    # complex Remez Stage
    (
        h,
        a,
        delta,
        not_optimal,
        iext,
        HH,
        EE,
        M_str,
        HH_str,
        h_str,
        Lf,
        Lb,
        Ls,
        Lc,
        A,
        globvars,
    ) = crmz(globvars, L, sym, Lfft, indx_edges)

    # cast to enforce Precision rules
    h = h.astype(complex)
    a = a.astype(complex)
    delta = delta.astype(complex)

    if not_optimal:
        # ascent-descent Stage:
        h, a, delta, HH, EE, globvars = adesc(
            globvars,
            L,
            Lf,
            Lb,
            Ls,
            Lc,
            sym,
            Lfft,
            indx_edges,
            iext,
            HH,
            EE,
            a,
            M_str,
            HH_str,
            h_str,
            A,
            delta,
        )

    # Return a row-vector, and remove imag part if it's small:
    h = h.squeeze()
    if np.iscomplexobj(h) and np.linalg.norm(h.imag) < 1e-12 * np.linalg.norm(h.real):
        h = h.real

    # % A 'real' filter was "forced" by making DES and WT conjugate symmetric (above).
    # % The optimization is done in the complex domain, even if 'real' was specified.
    # % Remove the imaginary part that was caused by roundoff errors during optimization
    # % Similar argument for 'even' and 'odd'
    if h_sym != "none":
        h = h.real

    # prepare output
    result = SimpleNamespace()
    result.fgrid = 2 * globvars.GRID_CRMZ.astype(np.float32)
    result.des = globvars.DES_CRMZ.astype(np.complex64)
    result.wt = globvars.WT_CRMZ.astype(np.float32)
    result.H = HH[globvars.IFGRD_CRMZ].astype(np.complex64)
    result.error = EE.astype(np.complex64)
    result.iextr = iext.astype(np.float32)
    result.fextr = 2 * globvars.GRID_CRMZ[iext].astype(np.float32)

    return h, delta, result


def multiband(N, F, GF=None, W=None, mags=None, delay=0):
    # Support query by CFIRPM for the default symmetry option
    if isinstance(N, str) and N == "defaults":
        num_args = len(F)
        if num_args < 6:
            delay = 0
        else:
            delay = F[5]

        # Use delay arg to base symmetry decision
        if delay == 0:
            return "even"
        else:
            return "real"

    # Standard call
    assert GF is not None, "Invalid number of parameters"
    assert W is not None, "Invalid number of parameters"
    assert mags is not None, "Invalid number of parameters"

    delay += N / 2  # adjust for linear phase
    W = np.tile(W[None, :], (2, 1))
    DH = interp1d(F, mags, kind="linear")(GF) * np.exp(-1j * np.pi * GF * delay)
    DW = interp1d(F, W.flatten(order="F"), kind="linear")(GF)

    return DH, DW


def eval_grid(
    globvars, edges, num_bands, M, L, fdomain, wgts, grid_density, other_params
):
    # Generate frequency grid:
    edge_pairs = np.array(edges).reshape(num_bands, 2)

    tgrid, Lfft, vec_edges, indx_edges = crmz_grid(edge_pairs, L, fdomain, grid_density)

    # Generate IFGRD_CRMZ:
    globvars.IFGRD_CRMZ = np.concatenate(
        [
            np.arange(indx_edges[jj], indx_edges[jj + 1] + 1, dtype=int)
            for jj in range(0, len(indx_edges), 2)
        ]
    )

    # Get points corresponding to frequency band intervals:
    globvars.TGRID_CRMZ = tgrid.flatten()
    globvars.GRID_CRMZ = globvars.TGRID_CRMZ[globvars.IFGRD_CRMZ]

    if (max(globvars.GRID_CRMZ) > edges[-1]) or (min(globvars.GRID_CRMZ) < edges[0]):
        raise ValueError("Internal Error")

    # Get desired frequency characteristics at specified intervals:
    # Note: We use the [0, 0.5] range, so we adjust the frequency bands and grid accordingly.
    globvars.DES_CRMZ, globvars.WT_CRMZ = multiband(
        M, 2 * np.array(edges), 2 * globvars.GRID_CRMZ, wgts, *other_params
    )

    # Cleanup the results and check sizes:
    globvars.DES_CRMZ = globvars.DES_CRMZ.flatten()
    globvars.WT_CRMZ = globvars.WT_CRMZ.flatten()

    if not (
        globvars.DES_CRMZ.shape == globvars.GRID_CRMZ.shape
        and globvars.WT_CRMZ.shape == globvars.GRID_CRMZ.shape
    ):
        raise ValueError("Invalid Dimensions: multiband, GF")

    return Lfft, indx_edges, vec_edges, globvars


def crmz(globvars, L, sym=0, Lfft=None, indx_edges=None):
    N1 = 0

    if sym is None:
        sym = 0

    N2 = N1 + L - 1
    is_odd = L % 2 != 0
    if is_odd:
        Lf = int((L + 1) // 2)
        Ws = "W[:,1:Lf]"
        hr_str = "np.concatenate((a[Lc-1:0:-1] / 2, a[[0]], a[1:Lc] / 2), axis=0)"
        hi_str = "np.concatenate((-a[Lb-1:Lc-1:-1] / 2, np.asarray([0]), a[Lc:Lb] / 2), axis=0)"
        ph_str = "np.ones(len(globvars.TGRID_CRMZ))"
    else:
        Lf = int(L // 2)
        Ws = "W"
        hr_str = "np.concatenate((a[Lc-1::-1] / 2, a[0:Lc] / 2), axis=0)"
        hi_str = "np.concatenate((-a[Lb-1:Lc-1:-1] / 2, a[Lc:Lb] / 2), axis=0)"
        ph_str = "np.exp(-1j * np.pi * globvars.TGRID_CRMZ)"
    Lc = Lf
    Ls = L - Lf

    if sym == 0:
        Lb = int(L)
        M_str = "np.concatenate((np.cos(W), np.sin(Ws)), axis=1)"
        h_str = f"{hr_str} + 1j * {hi_str}"
        HH_str = "np.fft.fftshift(np.fft.fft(hc, Lfft)) * " + ph_str
    elif sym == 1:
        Lb = int(Lc)
        M_str = "np.cos(W)"
        h_str = hr_str
        mask_str = "(np.arange(hc.shape[0]) <= Lfft//2 )"
        HH_str = f"np.fft.fft(hc, Lfft)[{mask_str}] * {ph_str}"
    elif sym == 2:
        Lb = int(Ls)
        Lc = 0
        M_str = "np.sin(Ws)"
        h_str = "1j * " + hi_str
        mask_str = "(np.arange(hc.shape[0]) <= Lfft//2 )"
        HH_str = f"np.fft.fft(hc, Lfft)[{mask_str}] * {ph_str}"

    A = globvars.DES_CRMZ * np.exp(1j * 2 * np.pi * globvars.GRID_CRMZ * (N1 + N2) / 2)

    vec_edges = globvars.TGRID_CRMZ[indx_edges]
    edges = vec_edges.reshape(2, len(vec_edges) // 2, order="F").T
    fext, iext = crmz_guess(edges, globvars.GRID_CRMZ, Lb)

    it = 0
    delta = 0.0
    delta_old = -1
    no_stp = True

    exactTol = np.finfo(float).eps ** (2 / 3)
    last_ee = np.zeros(10)

    while no_stp:
        it += 1
        delta_old = abs(delta)
        fext = globvars.GRID_CRMZ[iext]
        W = 2 * np.pi * fext[:, None] * (np.arange(Lf) + (not is_odd) * 0.5)
        Mb = eval(M_str)
        M = np.column_stack((Mb, ((-1) ** np.arange(Lb + 1)) / globvars.WT_CRMZ[iext]))
        a = np.linalg.lstsq(M, A[iext], rcond=None)[0]
        delta = a[Lb]
        h = eval(h_str)
        hc = crmz_rotate(zeropad(h, Lfft - L), -Ls)
        HH = eval(HH_str)
        W = 2 * np.pi * vec_edges[:, None] * (np.arange(Lf) + (not is_odd) * 0.5)
        Mb = eval(M_str)

        HH[indx_edges] = Mb @ a[:Lb]
        EE = globvars.WT_CRMZ * (A - HH[globvars.IFGRD_CRMZ])
        EE[iext] = delta * ((-1) ** np.arange(2, len(iext) + 2))
        jext, EEj = crmz_find(EE, iext)

        e_max = max(np.abs(EE))
        last_ee = np.roll(last_ee, shift=1)
        last_ee[0] = e_max
        s = max(np.abs(last_ee - e_max))

        if (
            np.all(iext == jext)
            or (e_max / max(np.abs(A)) < exactTol)
            or ((s < exactTol) and (it > 10))
        ):
            no_stp = False
        iext = jext

    tlr = abs(delta) / 100
    if e_max <= (abs(delta) + tlr) or e_max / max(np.abs(A)) < exactTol:
        not_optimal = 0
    else:
        not_optimal = 1

    return (
        h,
        a,
        delta,
        not_optimal,
        iext,
        HH,
        EE,
        M_str,
        HH_str,
        h_str,
        Lf,
        Lb,
        Ls,
        Lc,
        A,
        globvars,
    )


def crmz_find(error, fold):
    fold = fold.flatten("F")
    Nx = len(fold)
    Ngrid = len(error)

    if error[fold[0]] != 0:
        sgn_error = error[fold[0]] / abs(error[fold[0]])
    else:
        sgn_error = 1

    error = np.real(np.conj(sgn_error) * error)
    delta = min(np.abs(error[fold]))
    up = np.sign(error[fold[0]]) > 0

    if up:
        tmp1 = np.concatenate((np.asarray([0]), fold[:Nx:2]))
        tmp2 = np.concatenate((fold[:Nx:2] + 1, np.asarray([Ngrid])))
        fence = np.stack((tmp1, tmp2), axis=1)
    else:
        tmp1 = np.concatenate((np.asarray([0]), fold[1:Nx:2]))
        tmp2 = np.concatenate((fold[1:Nx:2] + 1, np.asarray([Ngrid])))
        fence = np.stack((tmp1, tmp2), axis=1)

    Lf = fence.shape[0]
    emn = np.zeros(Lf, dtype=float)
    imn = np.zeros(Lf, dtype=int)
    for i in range(Lf):
        start, end = int(fence[i][0]), int(fence[i][1])
        emn[i] = np.min(error[start:end])
        imn[i] = np.argmin(error[start:end]) + start

    imn = imn[np.logical_not(emn > -delta)]
    tmp1 = np.concatenate((np.asarray([0]), imn))
    tmp2 = np.concatenate((imn + 1, np.asarray([Ngrid])))
    fence = np.stack((tmp1, tmp2), axis=1)

    Lf = fence.shape[0]
    emx = np.zeros(Lf, dtype=float)
    imx = np.zeros(Lf, dtype=int)
    for i in range(Lf):
        start, end = int(fence[i][0]), int(fence[i][1])
        emx[i] = np.max(error[start:end])
        imx[i] = np.argmax(error[start:end]) + start

    imx = imx[np.logical_not(emx < delta)]
    fnew = np.sort(np.concatenate((imx, imn)))
    Nf = len(fnew)

    if Nf > Nx:
        if abs(error[fnew[0]]) >= abs(error[fnew[-1]]):
            fnew = fnew[:Nx]
        else:
            fnew = fnew[-Nx:]

    return fnew, error[fnew]


def crmz_grid(edge_pairs, L, fdomain, grid_density):
    if edge_pairs[0][0] == -0.5 and edge_pairs[-1][-1] == 0.5:
        # -pi and +pi are the same point - move one a little:
        new_freq = 0.5 - 1 / (50 * L)
        if new_freq <= edge_pairs[-1][-2]:
            # Last two freq points are too close to move - try first two:
            new_freq = -new_freq
            if new_freq >= edge_pairs[0][1]:
                raise ValueError("InvalidParam")
            else:
                edge_pairs[0][0] = new_freq
        else:
            edge_pairs[-1][-1] = new_freq

    Ngrid = 2 ** int(np.ceil(np.log2(L * grid_density)))
    if (Ngrid // 2) > 20 * L:
        Ngrid //= 2

    edge_vec = edge_pairs.T  # M-by-2 to 2-by-M
    edge_vec = edge_vec.ravel(order="F")  # single column of adjacent edge-pairs

    if fdomain == "whole":
        grid = np.arange(Ngrid) / Ngrid - 0.5  # uniform grid points [-.5,.5)
        edge_idx = np.round((edge_vec + 0.5) * Ngrid).astype(
            int
        )  # closest indices in grid
    elif fdomain == "half":
        grid = np.arange(Ngrid // 2 + 1) / Ngrid  # uniform grid points [0,.5]
        edge_idx = np.round(edge_vec * Ngrid).astype(int)  # closest indices in grid
    else:
        raise ValueError("InternalError")

    edge_idx[-1] = min(len(grid), edge_idx[-1])  # Clip last index

    # Fix repeated edges
    m = np.where(edge_idx[:-1] == edge_idx[1:])[0]
    if len(m) > 0:
        # Replace REPEATED band edges with the uniform grid points
        # Could be a problem if [-1 -1] (if whole) or [0 0] (if half) specified
        edge_idx[m] -= 1  # move 1 index lower
        edge_vec[m] = grid[edge_idx[m]]  # change user's band edge accordingly
        m += 1
        edge_idx[m] += 1
        edge_vec[m] = grid[edge_idx[m]]

    # Replace closest grid points with exact band edges:
    grid[edge_idx] = edge_vec

    return grid, Ngrid, edge_vec, edge_idx


def crmz_guess(edges, grid, nfcns):
    TOL = 5 * np.finfo(float).eps
    next = int(nfcns + 1)
    Nbands = len(edges[:, 1])
    tt = edges.copy()
    merged = tt.copy()
    if Nbands > 1:
        jkl = np.where(np.abs(tt[0 : Nbands - 1, 1] - tt[1:Nbands, 0]) > TOL)[0]
        tmp1 = np.concatenate((np.atleast_1d(tt[0, 0]), tt[jkl + 1, 0]), axis=0)
        tmp2 = np.concatenate((tt[jkl, 1], np.atleast_1d(tt[Nbands - 1, 1])), axis=0)
        merged = np.stack((tmp1, tmp2), axis=1)

    Nbands = len(merged[:, 0])
    bw = merged[:, 1] - merged[:, 0]

    if np.any(bw < 0):
        edges
        raise ValueError("InternalErrorNegBW")

    percent_bw = bw / np.sum(bw)
    fext = np.zeros(next)
    n = 0
    i = 0

    while n < next and i < Nbands:
        nfreqs_i = min(next - n, int(np.ceil(percent_bw[i] * next)))

        if nfreqs_i == 0:
            n += 1
            fext[n] = merged[i, 0]
        else:
            fext[n : n + nfreqs_i] = np.linspace(merged[i, 0], merged[i, 1], nfreqs_i)
            n += nfreqs_i
        i += 1

    iext = np.zeros(next, dtype=int)

    for i in range(next):
        iext[i] = np.argmin(np.abs(fext[i] - grid))

    if np.any(np.diff(iext) == 0):
        raise ValueError("InternalErrorGridPoint")

    fext = grid[iext]
    return fext, iext


def crmz_rotate(x, num_places):
    if len(x.shape) > 1:
        M, N = x.shape
        num_places = int(num_places % M)  # Ensure num_places is in the range [0, M-1]
        rotated = np.vstack((x[M - num_places : M, :], x[0 : M - num_places, :]))
    else:
        N = x.shape[0]
        num_places = int(num_places % N)  # Ensure num_places is in the range [0, N-1]
        rotated = np.hstack((x[N - num_places : N], x[0 : N - num_places]))
    return rotated


def adesc(
    globvars,
    L,
    Lf,
    Lb,
    Ls,
    Lc,
    sym,
    Lfft,
    indx_edges,
    iext,
    HH,
    EE,
    a,
    M_str,
    HH_str,
    h_str,
    A,
    delta,
):
    ACCURACY = 0.01
    is_odd = L % 2 != 0
    vec_edges = globvars.TGRID_CRMZ[indx_edges]
    bands = np.zeros(len(vec_edges))
    for k in range(len(vec_edges)):
        if vec_edges[k] != 1:
            bands[k] = np.where(vec_edges[k] == globvars.GRID_CRMZ)[0]
        else:
            bands[k] = len(globvars.GRID_CRMZ)

    no_stp = 1
    a = a[:Lb]
    a = a.reshape(-1, 1)
    n2 = len(a)
    n = 2 * len(a)
    mxl = 2 * L + 1
    maj_it = 0
    alpha = 1
    nu = 0

    acc_scale = np.max(np.abs(globvars.WT_CRMZ * A))
    acc_min = ACCURACY
    acc = acc_min
    if acc < acc_min:
        acc = acc_min
    r_min = 0.5
    r_o = acc_scale * r_min
    if r_o < r_min:
        r_o = r_min
    r = r_o
    epsi_min = ACCURACY
    epsi_o = acc_scale * epsi_min
    epsil = epsi_o
    epsilon = 1 - 0.005

    HH_o = HH
    e_max = np.max(np.abs(EE))

    iext = adesc_findset(EE, iext, bands, delta)
    sub_EE = EE[iext]
    sub_grd = globvars.GRID_CRMZ[iext]
    sub_WT = globvars.WT_CRMZ[iext]
    sub_max = np.max(np.abs(sub_EE))
    fmax = iext
    jext = adesc_findextr(EE, bands, epsilon)

    gext = adesc_grad(EE, jext, globvars.GRID_CRMZ, globvars.WT_CRMZ, M_str, Lf, is_odd)
    rext = adesc_minpolytope(gext)
    no_stp = np.linalg.norm(rext) > acc

    while no_stp:
        jext = np.where((np.max(np.abs(sub_EE)) - np.abs(sub_EE)) <= epsil)[0]
        G = adesc_grad(sub_EE, jext, sub_grd, sub_WT, M_str, Lf, is_odd)
        NrG = adesc_minpolytope(G)
        norm_NrG = np.linalg.norm(NrG)

        if norm_NrG <= r:
            nu += 1
            epsil = epsi_o / (2**nu)
            r = r_o / (2**nu)
            f_extr = np.where(np.abs(sub_EE) / sub_max >= epsilon)[0]
            gext = adesc_grad(sub_EE, f_extr, sub_grd, sub_WT, M_str, Lf, is_odd)
            rext = adesc_minpolytope(gext)
            if np.linalg.norm(rext) < acc:
                maj_it += 1
                nu = 0
                epsil = epsi_o
                r = r_o
                iext = adesc_findset(EE, iext, bands, sub_max)
                sub_EE = EE[iext]
                sub_grd = globvars.GRID_CRMZ[iext]
                sub_WT = globvars.WT_CRMZ[iext]
                sub_max = np.max(np.abs(sub_EE))
                if len(iext) == len(fmax):
                    no_stp = not np.all(iext == fmax)
                fmax = iext
        else:
            d = -NrG / norm_NrG
            d2 = d[:n2] + 1j * d[n2:n]
            HH_d, globvars = adesc_reconst(
                globvars,
                d2,
                h_str,
                HH_str,
                M_str,
                globvars.TGRID_CRMZ,
                Lfft,
                L,
                Lf,
                Lc,
                Ls,
                Lb,
                is_odd,
                vec_edges,
                indx_edges,
            )
            HH_n, EE, e_max, alpha = adesc_linsearch(
                alpha,
                iext,
                HH_o,
                HH_d,
                sub_max,
                A,
                globvars.WT_CRMZ,
                globvars.IFGRD_CRMZ,
            )
            a += alpha * d2
            HH_o = HH_n
            sub_EE = EE[iext]
            sub_max = np.max(np.abs(sub_EE))

    HH = HH_o
    a = a.ravel()
    h = eval(h_str)
    epsilon = 0.9 * sub_max / e_max
    iext = adesc_findextr(EE, bands, epsilon)
    jext = iext.ravel()
    delta = np.max(np.abs(EE))

    return h, a, delta, HH, EE, globvars


def adesc_findextr(error, indx_edges, epsilon):
    error = error.ravel()
    indx_edges = indx_edges.ravel()
    Ngrid = len(error)
    abs_e = np.abs(error)
    abs_e = np.concatenate(([abs_e[1]], abs_e, [abs_e[Ngrid - 1]]))
    fmax = np.where(
        (abs_e[1 : Ngrid + 1] >= abs_e[0:Ngrid])
        & (abs_e[1 : Ngrid + 1] > abs_e[2 : Ngrid + 2])
    )[0]
    fmax = fmax.ravel()
    fmax = np.sort(np.concatenate((fmax, indx_edges)))
    abs_e = np.delete(abs_e, [0, Ngrid + 1])

    idx = fmax[0 : len(fmax) - 1] - fmax[1 : len(fmax)] != 0
    if idx.shape[0] < fmax.shape[0]:
        idx = np.pad(idx, (0, fmax.shape[0] - idx.shape[0]), constant_values=(1, 1))
    fmax = fmax[idx].astype(int)

    if len(fmax) > 1:
        fmax = fmax[abs_e[fmax] / np.max(abs_e) >= epsilon]

    return fmax


def adesc_findset(error, iext, indx_edges, delta):
    error = error.ravel(order="F")
    indx_edges = indx_edges.ravel(order="F")
    Ngrid = len(error)
    abs_e = np.abs(error)
    abs_e = np.concatenate(([abs_e[1]], abs_e, [abs_e[Ngrid - 1]]))
    fmax = np.where(
        (abs_e[1 : Ngrid + 1] >= abs_e[0:Ngrid])
        & (abs_e[1 : Ngrid + 1] >= abs_e[2 : Ngrid + 2])
    )[0]
    fmax = fmax.ravel(order="F")
    fmax = np.sort(np.concatenate((fmax, indx_edges, iext)))
    abs_e = np.delete(abs_e, [0, Ngrid + 1])
    idx = fmax[0 : len(fmax) - 1] - fmax[1 : len(fmax)] != 0
    if idx.shape[0] < fmax.shape[0]:
        idx = np.pad(idx, (0, fmax.shape[0] - idx.shape[0]), constant_values=(1, 1))
    fmax = fmax[idx].astype(int)
    fmax = fmax[np.logical_not(abs_e[fmax] < 0.9 * np.abs(delta))]

    return fmax


def adesc_grad(EE, iext, grd, WT, M_str, Lf, is_odd):
    J = 1j
    fext = grd[iext]
    W = 2 * np.pi * fext[:, None] * (np.arange(Lf) + (not is_odd) * 0.5)
    Mb = eval(M_str)
    Mb = -Mb.conj().T
    MWT = np.diag(2 * WT[iext] * EE[iext])
    M = Mb @ MWT
    G = np.concatenate((M.real, M.imag), axis=0)

    return G


def adesc_linsearch(t0, iext, HH_o, d, emx, DD, WT, ifgrid):
    t = t0
    HH_o = HH_o.ravel()
    d = d.ravel()
    no_stp = True
    r_flg = False
    t_flg = True
    h_flg = False
    i_flg = False
    grtr = True
    c = 2
    acc = 1 - 10 ** (-0.1 / 10)
    while grtr:
        HHt = HH_o + t * d
        EEt = WT * (DD - HHt[ifgrid])
        emxt = np.max(np.abs(EEt[iext]))
        if emxt <= emx:
            grtr = False
        else:
            t = t / 2
    tmin = t
    HHt_min = HHt
    EEt_min = EEt
    emin = emxt
    I = 2 * t * np.ones(2)
    while no_stp:
        if not i_flg:
            t = c * t
        HHt = HH_o + t * d
        EEt = WT * (DD - HHt[ifgrid])
        emxt = np.max(np.abs(EEt[iext]))
        R = emxt <= emin
        if R:
            tmin = t
            HHt_min = HHt
            EEt_min = EEt
            emin = emxt
            t_flg = False
            h_flg = False
            if i_flg:
                I = np.sort([t, I[1]])
                t = (t + I[1]) / 2
            r_flg = True
        elif not R and i_flg:
            I = np.sort([I[0], t])
            t = (t + I[0]) / 2
        elif t_flg:
            t1 = t
            t_flg = False
        elif r_flg:
            if r_flg:
                t1 = t
            I = np.sort([tmin, t1])
            t = (tmin + t1) / 2
            i_flg = True
        else:
            no_stp = False
        if i_flg and I[1] - t < acc * I[1]:
            no_stp = False
    HHt = HHt_min
    EEt = EEt_min
    emxt = np.max(np.abs(EEt))
    t = tmin

    return HHt, EEt, emxt, t


def adesc_reconst(
    globvars,
    at,
    h_str,
    HH_str,
    M_str,
    tgrid,
    Lfft,
    L,
    Lf,
    Lc,
    Ls,
    Lb,
    is_odd,
    v_edges,
    in_edges,
):
    J = 1j
    a = at.ravel()
    h = eval(h_str)
    hc = crmz_rotate(zeropad(h, Lfft - L), -Ls)
    eval(HH_str)
    W = 2 * np.pi * v_edges * ((np.arange(Lf) + (~is_odd) * 0.5))
    Mb = eval(M_str)
    HH = np.zeros(len(globvars.GRID_CRMZ))
    HH[in_edges] = Mb @ a

    return HH, globvars


def adesc_minpolytope(P):
    N, M = P.shape
    if M == 1:
        Ptmin = P
        return Ptmin

    Z1 = 1e-10
    Z2 = 1e-10
    Z3 = 1e-10
    P_norm = np.sum(np.abs(P) ** 2, axis=0)
    pmn, J = np.min(P_norm), np.argmin(P_norm)
    S = np.array([J])
    w = np.array([1])
    no_stp = 1

    while no_stp:
        X = P[:, S] @ w
        pmn, J = np.min(X @ P), np.argmin(X @ P)
        PS_norm = np.max(np.sum(np.abs(P[:, S]) ** 2))
        PJ_norm = P[:, J].conj().T @ P[:, J]
        no_stp = X.conj().T @ P[:, J] <= (X.conj().T @ X - Z1 * max(PJ_norm, PS_norm))
        if no_stp:
            I = np.where(np.logical_not(S - J))[0]
            if len(I) > 0:
                no_stp = 0
        if no_stp:
            S = np.concatenate((S, [J]))
            w = np.concatenate((w, [0]))
            flg = 1
            while flg:
                e = np.ones(len(S))
                A = np.ones((len(S), len(S))) + P[:, S].conj().T @ P[:, S]
                u = np.linalg.solve(A, e)
                v = u / (e @ u)
                I = np.where(v <= Z2)[0]
                if len(I) == 0:
                    w = v
                    flg = 0
                else:
                    I = np.where((w - v) > Z3)[0]
                    t = w[I] / (w[I] - v[I])
                    theta = min([1, 1 - min(t)])
                    w = theta * w + (1 - theta) * v
                    I = np.where(w <= Z2)[0]
                    if len(I) > 0:
                        w[I] = 0
                        w = np.delete(w, I[0])
                        S = np.delete(S, I[0])
        Ptmin = X

    return Ptmin


def zeropad(x, N):
    """Zero pads signal x to length N."""
    return np.pad(x, (0, int(N)))
