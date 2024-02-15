import numpy as np

def spec_interp(h, ni, off, f, dbg):
    N = len(h)
    mult_factor = 15
    w = np.linspace(-np.pi, np.pi, 2 * mult_factor * N)

    f = np.array(f) * np.pi
    idx_band = []
    nband = len(f) // 2
    for band in range(nband):
        idx = np.where((w >= f[band * 2 - 1]) & (w <= f[band * 2]))
        idx_band.extend(idx[0])
    wt_band = 10
    wt = np.ones(len(w))
    wt[idx_band] = wt_band

    t_ref = np.arange(N)
    Wref = np.exp(-1j * np.kron(w, t_ref))
    Fref = np.dot(Wref, h)
    Fref_wt = wt * Fref

    # ni_2 = ni // 2
    hi = np.zeros((ni, N), dtype=np.complex128)

    for idx in range(ni):
        t_act = t_ref + off + (idx / ni)

        Wact = np.exp(-1j * np.kron(w, t_act))
        Wact_wt = np.outer(wt, Wact)
        hi[idx, :] = np.linalg.pinv(Wact_wt).dot(Fref_wt)
        
    hi = hi.flatten()
    return hi

