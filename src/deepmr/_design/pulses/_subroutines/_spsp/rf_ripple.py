"""RF ripple."""

__all__ = ["rf_ripple"]

import numpy as np

def rf_ripple(de, a, ang, excitation_type):
    nband = len(de)
    d = np.zeros(nband)

    for band in range(nband):
        db = de[band]
        ab = a[band]

        if excitation_type == "ex":
            mxy_mid = np.sin(2 * np.arcsin(np.sin(ang / 2) * ab))
            mxy_pos = np.clip(db + mxy_mid, -1, 1)
            mxy_neg = np.clip(-db + mxy_mid, -1, 1)

            B_mid = np.sin(ang / 2) * ab
            B_pos = np.sin(np.arcsin(mxy_pos) / 2)
            B_neg = np.sin(np.arcsin(mxy_neg) / 2)

            d[band] = max(np.abs(B_pos - B_mid), np.abs(B_neg - B_mid)) / np.sin(
                ang / 2
            )

        elif excitation_type == "se":
            mxy_mid = (np.sin(ang / 2) * ab) ** 2
            mxy_pos = np.clip(db + mxy_mid, 0, 1)
            mxy_neg = np.clip(-db + mxy_mid, 0, 1)

            if mxy_pos >= 1 or mxy_neg >= 1:
                ab = np.sqrt(1 - np.abs(db)) / np.sin(ang / 2)
                mxy_mid = (np.sin(ang / 2) * ab) ** 2
                mxy_pos = np.clip(db + mxy_mid, 0, 1)
                mxy_neg = np.clip(-db + mxy_mid, 0, 1)
                a[band] = ab

            B_mid = np.sin(ang / 2) * ab
            B_pos = np.sqrt(mxy_pos)
            B_neg = np.sqrt(mxy_neg)

            d[band] = max(np.abs(B_pos - B_mid), np.abs(B_neg - B_mid)) / np.sin(
                ang / 2
            )

        elif excitation_type in ["inv", "sat"]:
            mz_mid = 1 - 2 * (np.sin(ang / 2) * ab) ** 2
            mz_pos = np.clip(db + mz_mid, -1, 1)
            mz_neg = np.clip(-db + mz_mid, -1, 1)

            if mz_neg <= -1 or mz_pos <= -1:
                ab = np.sqrt(1 - np.abs(db) / 2) / np.sin(ang / 2)
                mz_mid = 1 - 2 * (np.sin(ang / 2) * ab) ** 2
                mz_pos = np.clip(db + mz_mid, -1, 1)
                mz_neg = np.clip(-db + mz_mid, -1, 1)
                a[band] = ab

            B_mid = np.sin(ang / 2) * ab
            B_pos = np.sin(np.arccos(mz_pos) / 2)
            B_neg = np.sin(np.arccos(mz_neg) / 2)

            d[band] = max(np.abs(B_pos - B_mid), np.abs(B_neg - B_mid)) / np.sin(
                ang / 2
            )

    if np.max(a) != 1:
        max_a = np.max(a)
        a = a / max_a
        ang = 2 * np.arcsin(np.sin(ang / 2) * max_a)
        d = d / max_a

    return d, a, ang
