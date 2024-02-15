"""MRI RF excitation pulse design functions, i.e., Shinnar-LeRoux (SLR) design algorithm.

SLR algorithm simplifies the solution of the Bloch equations to the design of 2 polynomials.
"""

__all__ = [
    "dzrf",
    "dzbeta",
    "ab2rf",
    "b2rf",
    "ab2ex",
    "ab2inv",
    "ab2sat",
    "ab2se",
    "ab2st",
    "abr",
]

import numpy as np
import scipy.signal as signal

from ._utils import dinf

gamma_bar = 42.575 * 1e6  # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar  # rad / T / s


def dzrf(n=64, tb=4, ptype="st", ftype="ls", d1=0.01, d2=0.01, cancel_alpha_phs=False):
    """
    Primary function for design of pulses using the SLR algorithm.

    Args:
        n (int): number of time points.
        tb (int): pulse time bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        ftype (string): type of filter to use: 'ms' (sinc), 'pm'
            (Parks-McClellan equal-ripple), 'min' (minphase using factored pm),
            'max' (maxphase using factored pm), 'ls' (least squares), or 'cp'
            (custom excitation profile).
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        cancel_alpha_phs (bool): For 'ex' pulses, absorb the alpha phase
            profile from beta's profile, so they cancel for a flatter
            total phase
        custom_profile (array): if provided, pulse will be designed to excite
            an arbitrary profile rather than a rectangular one, following [2].

    Returns:
        rf (array): designed RF pulse.

    References:
        [1] Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.
        (1991). Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

        [2] Barral, J., Pauly, J., and Nishimura, D. (2008). SLR RF Pulse
        Design for Arbitrarily-Shaped Excitation Profiles.
        Proc. Intl. Soc. Mag. Reson. Med. 16, 1323.
    """
    [bsf, d1, d2] = calc_ripples(ptype, d1, d2)

    if ftype == "ms":  # sinc
        b = msinc(n, tb / 4)
    elif ftype == "pm":  # linphase
        b = dzlp(n, tb, d1, d2)
    elif ftype == "min":  # minphase
        b = dzmp(n, tb, d1, d2)
        b = b[::-1]
    elif ftype == "max":  # maxphase
        b = dzmp(n, tb, d1, d2)
    elif ftype == "ls":  # least squares
        b = dzls(n, tb, d1, d2)
    else:
        raise Exception('Filter type ("{}") is not recognized.'.format(ftype))

    if ptype == "st":
        rf = b
    elif ptype == "ex":
        b = bsf * b
        rf = b2rf(b, cancel_alpha_phs)
    else:
        b = bsf * b
        rf = b2rf(b)

    return rf


def dzbeta(n=64, tb=4, ptype="st", ftype="ls", d1=0.01, d2=0.01):
    """
    Return beta for rf design using the SLR algorithm.

    Args:
        n (int): number of time points.
        tb (int): pulse time bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        ftype (string): type of filter to use: 'ms' (sinc), 'pm'
            (Parks-McClellan equal-ripple), 'min' (minphase using factored pm),
            'max' (maxphase using factored pm), 'ls' (least squares), or 'cp'
            (custom excitation profile).
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        cancel_alpha_phs (bool): For 'ex' pulses, absorb the alpha phase
            profile from beta's profile, so they cancel for a flatter
            total phase
        custom_profile (array): if provided, pulse will be designed to excite
            an arbitrary profile rather than a rectangular one, following [2].

    Returns:
        rf (array): designed RF pulse.

    References:
        [1] Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.
        (1991). Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

        [2] Barral, J., Pauly, J., and Nishimura, D. (2008). SLR RF Pulse
        Design for Arbitrarily-Shaped Excitation Profiles.
        Proc. Intl. Soc. Mag. Reson. Med. 16, 1323.
    """
    [bsf, d1, d2] = calc_ripples(ptype, d1, d2)

    if ftype == "ms":  # sinc
        b = msinc(n, tb / 4)
    elif ftype == "pm":  # linphase
        b = dzlp(n, tb, d1, d2)
    elif ftype == "min":  # minphase
        b = dzmp(n, tb, d1, d2)
        b = b[::-1]
    elif ftype == "max":  # maxphase
        b = dzmp(n, tb, d1, d2)
    elif ftype == "ls":  # least squares
        b = dzls(n, tb, d1, d2)
    else:
        raise Exception('Filter type ("{}") is not recognized.'.format(ftype))

    if ptype != "ex":
        b = bsf * b

    return b


# %% local utils (i.e.m John Pauly rf_tools package)
def calc_ripples(ptype="st", d1=0.01, d2=0.01):
    if ptype == "st":
        bsf = 1
    elif ptype == "ex":
        bsf = np.sqrt(1 / 2)
        d1 = np.sqrt(d1 / 2)
        d2 = d2 / np.sqrt(2)
    elif ptype == "se":
        bsf = 1
        d1 = d1 / 4
        d2 = np.sqrt(d2)
    elif ptype == "inv":
        bsf = 1
        d1 = d1 / 8
        d2 = np.sqrt(d2 / 2)
    elif ptype == "sat":
        bsf = np.sqrt(1 / 2)
        d1 = d1 / 2
        d2 = np.sqrt(d2)
    else:
        raise Exception('Pulse type ("{}") is not recognized.'.format(ptype))

    return bsf, d1, d2


# following functions are used to support dzrf
def dzls(n=64, tb=4, d1=0.01, d2=0.01):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)])
    f = f / (n / 2)
    m = [1, 1, 0, 0]
    w = [1, d1 / d2]

    if n % 2 == 0:
        h = signal.firls(n + 1, f, m, w)

        # shift the filter half a sample to make it symmetric, like in MATLAB
        c = np.exp(
            1j
            * 2
            * np.pi
            / (2 * (n + 1))
            * np.concatenate([np.arange(0, n / 2 + 1, 1), np.arange(-n / 2, 0, 1)])
        )
        h = np.real(np.fft.ifft(np.multiply(np.fft.fft(h), c)))

        # lop off extra sample
        h = h[:n]
    else:
        h = signal.firls(n, f, m, w)

    return h


def dzmp(n=64, tb=4, d1=0.01, d2=0.01):
    n2 = 2 * n - 1
    di = 0.5 * dinf(2 * d1, 0.5 * d2 * d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)]) / n
    m = [1, 0]
    w = [1, 2 * d1 / (0.5 * d2 * d2)]

    hl = signal.remez(n2, f, m, w)

    h = fmp(hl)

    return h


def fmp(h):
    l = np.size(h)
    lp = 128 * np.exp(np.ceil(np.log(l) / np.log(2)) * np.log(2))
    padwidths = np.array([np.ceil((lp - l) / 2), np.floor((lp - l) / 2)])
    hp = np.pad(h, padwidths.astype(int), "constant")
    hpf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(hp)))
    hpfs = hpf - np.min(np.real(hpf)) * 1.000001
    hpfmp = mag2mp(np.sqrt(np.abs(hpfs)))
    hpmp = np.fft.ifft(np.fft.ifftshift(np.conj(hpfmp)))
    hmp = hpmp[: int((l + 1) / 2)]

    return hmp


def dzlp(n=64, tb=4, d1=0.01, d2=0.01):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)]) / n
    m = [1, 0]
    w = [1, d1 / d2]

    h = signal.remez(n, f, m, w)

    return h


def msinc(n=64, m=1) -> np.ndarray:
    x = np.arange(-n / 2, n / 2, 1) / (n / 2)
    snc = np.divide(np.sin(m * 2 * np.pi * x + 0.00001), (m * 2 * np.pi * x + 0.00001))
    ms = np.multiply(snc, 0.54 + 0.46 * np.cos(np.pi * x))
    ms = ms * 4 * m / n

    return ms


def b2rf(b, cancel_alpha_phs=False):
    a = b2a(b)
    if cancel_alpha_phs:
        b_a_phase = np.fft.fft(
            b,
        ) * np.exp(-1j * np.angle(np.fft.fft(a[np.size(a) :: -1])))
        b = np.fft.ifft(b_a_phase)
    rf = ab2rf(a, b)

    return rf


def b2a(b):
    n = np.size(b)

    npad = n * 16
    bcp = np.zeros(npad, dtype=complex)
    bcp[0:n:1] = b
    bf = np.fft.fft(bcp)
    bfmax = np.max(np.abs(bf))
    if bfmax >= 1:
        bf = bf / (1e-7 + bfmax)
    afa = mag2mp(np.sqrt(1 - np.abs(bf) ** 2))
    a = np.fft.fft(afa) / npad
    a = a[0:n:1]
    a = a[::-1]

    return a


def mag2mp(x):
    n = np.size(x)
    xl = np.log(np.abs(x))  # Log of mag spectrum
    xlf = np.fft.fft(xl)
    xlfp = xlf
    xlfp[0] = xlf[0]  # Keep DC the same
    xlfp[1 : (n // 2) : 1] = 2 * xlf[1 : (n // 2) : 1]  # Double positive frequencies
    xlfp[n // 2] = xlf[n // 2]  # keep half Nyquist the same
    xlfp[n // 2 + 1 : n : 1] = 0  # zero negative frequencies
    xlaf = np.fft.ifft(xlfp)
    a = np.exp(xlaf)  # complex exponentiation

    return a


def ab2rf(a, b):
    n = np.size(a)
    rf = np.zeros(n, dtype=complex)

    a = a.astype(complex)
    b = b.astype(complex)

    for ii in range(n - 1, -1, -1):
        cj = np.sqrt(1 / (1 + np.abs(b[ii] / a[ii]) ** 2))
        sj = np.conj(cj * b[ii] / a[ii])
        theta = np.arctan2(np.abs(sj), cj)
        psi = np.angle(sj)
        rf[ii] = 2 * theta * np.exp(1j * psi)

        # remove this rotation from polynomials
        if ii > 0:
            at = cj * a + sj * b
            bt = -np.conj(sj) * a + cj * b
            a = at[1 : ii + 1 : 1]
            b = bt[0:ii:1]

    return rf


def ab2ex(a, b=None):
    if b is None:
        b = a[1]
        a = a[0]

    mxy = 2 * np.conj(a) * b
    return mxy


def ab2inv(a, b=None):
    if b is None:
        b = a[1]
        a = a[0]

    mz = 1 - 2 * np.conj(b) * b
    return mz


def ab2sat(a, b=None):
    if b is None:
        b = a[1]
        a = a[0]

    mz = 1 - 2 * np.conj(b) * b
    return mz


def ab2se(a, b=None):
    if b is None:
        b = a[1]
        a = a[0]

    mxy = 1j * b * b
    return mxy


def ab2st(a, b=None):
    if b is not None:
        return 1j * a * a
    elif a.ndim == 2:
        return 1j * a[0] * a[0]
    else:
        raise ValueError("Invalid input")


def abr(rf, g, x, y=None):
    l = len(rf)

    if len(g) != l:
        raise ValueError("rf and g must have the same length")

    if y is None:
        y = x
    elif np.isscalar(y):
        y = y * np.ones(x.shape)
    elif len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # move f grad on second dim
    if np.iscomplexobj(g):
        g = np.stack((g.real, g.imag), axis=-1)

    a, b = abrx(rf, np.stack((x, y), axis=-1), g)
    b = -np.conj(b)

    return a, b


def abrx(rf, x, g):
    r"""
    N-dim RF pulse simulation.

    Assumes that x has inverse spatial units of g, and g has gamma*dt applied.

    Assumes dimensions x = [...,Ndim], g = [Ndim,Nt].

    Args:
         rf (array): rf waveform input.
         x (array): spatial locations.
         g (array): gradient array.

    Returns:
        2-element tuple containing

        - **a** (*array*): SLR alpha parameter.
        - **b** (*array*): SLR beta parameter.

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        'Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm'.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
    """
    xp = np
    eps = 1e-16

    a = xp.ones(xp.shape(x)[0], dtype=complex)
    b = xp.zeros(xp.shape(x)[0], dtype=complex)
    for mm in range(xp.size(rf)):
        om = x @ g[mm, :]
        phi = xp.sqrt(xp.abs(rf[mm]) ** 2 + om**2)
        n = xp.column_stack(
            (
                xp.real(rf[mm]) / (phi + eps),
                xp.imag(rf[mm]) / (phi + eps),
                om / (phi + eps),
            )
        )
        av = xp.cos(phi / 2) - 1j * n[:, 2] * xp.sin(phi / 2)
        bv = -1j * (n[:, 0] + 1j * n[:, 1]) * xp.sin(phi / 2)
        at = av * a - xp.conj(bv) * b
        bt = bv * a + xp.conj(av) * b
        a = at
        b = bt

    return a, b
