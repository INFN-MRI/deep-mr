import numpy as np
from numpy.linalg import pinv

from .._slr import ab2ex
from .._slr import abr

def ss_spect_correct(b, bsf, Nper, Noff, f, ptype, ss_type, slr, reg_factor, dbg=0):
    if ss_type not in ['Flyback', 'EP']:
        raise ValueError('ss_type must be "Flyback" or "EP"')

    nfilt = len(bsf)
    N = len(b)
    t_ref = np.arange(N)
    mult_factor = 15
    fdiff = np.diff(f)
    fsum = np.sum(fdiff[::2])
    df = fsum / (mult_factor * N)
    nband = len(f) // 2
    w = []

    for band in range(nband):
        nf = int(np.ceil((f[2*band + 1] - f[2*band]) / df)) + 1
        df_act = (f[2*band + 1] - f[2*band]) / (nf - 1)
        wband = f[2*band] + np.arange(nf) * df_act
        w = np.concatenate((w, np.pi * wband))

    # Calculate sampling positions of ref filter taps
    t_ref = np.arange(N)

    rfm = np.zeros((N, nfilt), dtype=complex)
    bm = np.zeros((N, nfilt), dtype=complex)

    for idx in range(nfilt):
        if ss_type == 'Flyback':
            t_act = t_ref + Noff[idx] / Nper
        else:
            t_act = t_ref + Noff[idx] / Nper * (-1) ** np.arange(N)

        Wref = np.exp(-1j * np.kron(w, t_ref))
        Fref = Wref.dot(b)

        # Get actual sampling positions
        Wact = np.exp(-1j * np.kron(w, t_act))

        if ss_type == 'Flyback':
            # Regularized least-squares
            Wact_pinv = pinv(Wact)
            bm[:, idx] = Wact_pinv.dot(Fref)
            rfm[:, idx] = bsf[idx] * bm[:, idx]
        else:
            # Regularized least-squares
            WactT_Wact = Wact.T.conj().dot(Wact)
            bm[:, idx] = np.linalg.solve(WactT_Wact + reg_factor * np.eye(WactT_Wact.shape[0]), Wact.T.conj().dot(Fref))
            rfm[:, idx] = 2 * np.arcsin(np.abs(bsf[idx])) * bm[:, idx]

    return rfm
