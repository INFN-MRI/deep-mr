"""
"""

import numpy as np


def ss_alias(f, a, d, f_off, fs, sym, threshold_edge=0.03):
    """
    Alias filter specifications into effective bandwidth.

    Args:
        f (tuple of floats): frequency band edges, in Hz, each band monotonically increasing,
            of length (2*number of bands,).
        a (tuple of floats): amplitude for each band [0..1].
        d (tuple of floats): ripple weighting for each band .
        f_off (float, optionall): offset frequency in Hz.
        fs (float): sampling frequency (bandwidth), in Hz.
        sym (bool): flag indicating whether frequency response should be symmetric.
        threshold_edge (float, optional): threshold of the min distance to the
            normalized spectrum edge [-1,1]. Defaults to 0.03.

    Returns:
        (tuple of floats): Aliased frequency band edges into normalized freq [-1..1].
        (tuple of floats): amplitude of aliased bands.
        (tuple of floats): ripple weighting of aliased bands.
        (float): offset frequency in Hz.

    """
    # Check input parameters
    if len(f) % 2 != 0:
        raise ValueError("Frequency vector (f) of band pairs must have an even length")

    nf = len(f) // 2
    if len(d) != nf:
        raise ValueError("Ripple vector incorrect length")
    if len(a) != nf:
        raise ValueError("Amplitude vector incorrect length")

    df = np.diff(f)
    if np.min(df[0::2]) <= 0:
        raise ValueError("Frequency bands must be monotonically increasing")

    if np.max(df[0::2]) >= fs:
        # There is one band longer than the bandwidth, impossible to design such a filter
        return [], [], [], 0

    # If sym flag then either set or check f_off
    if sym:
        if f_off is not None:
            # Check specified frequency offset to make sure it is < midpoint of the first band or > than midpoint of the top band
            if f_off > (f[1] - f[0]) / 2 and f_off < (f[-1] - f[-2]) / 2:
                raise ValueError("f_off not at band edges")
        else:
            # Set f_off to the left band edge
            f_off = (f[1] + f[0]) / 2
    else:
        # Set f_off to include as many bands as possible in one sampling interval
        if len(f_off) == 0:
            if max(f) - min(f) <= fs:
                # All bands can exist in the center spectrum
                f_off = (max(f) + min(f)) / 2
            else:
                f_l = f[0::2]
                f_u = f[1::2]
                f_off_test = np.linspace(min(f) / 2, max(f) / 2, 500)
                band_in = np.zeros((len(f_off_test), len(f_l)), dtype=bool)
                nband = np.zeros(len(f_off_test))
                edge_distance = np.zeros(len(f_off_test))
                split_flag = np.zeros(len(f_off_test))

                for off_idx in range(len(f_off_test)):
                    # Set f_off to the current test value
                    f_off = f_off_test[off_idx]

                    # Check how many bands are in the -fs/2..fs/2 range
                    band_in[off_idx, :] = (
                        (f_l >= f_off - fs / 2)
                        & (f_l <= f_off + fs / 2)
                        & (f_u >= f_off - fs / 2)
                        & (f_u <= f_off + fs / 2)
                    )
                    nband[off_idx] = np.sum(band_in[off_idx, :])

                    # Pre-calculation of normalized and aliased frequencies
                    fnorm_pre = np.zeros(len(f))
                    for idx in range(nf):
                        # Normalized frequencies
                        fa1 = (f[2 * idx] - f_off) / (fs / 2)
                        fa2 = (f[2 * idx + 1] - f_off) / (fs / 2)

                        # Get aliased frequencies
                        fa1 = (fa1 + 1) % 2 - 1
                        fa2 = (fa2 + 1) % 2 - 1

                        # Get rid of confusion about whether at -1 or 1
                        if fa2 < fa1 and fa2 == -1:
                            fa2 = 1
                        if fa2 < fa1 and fa1 == 1:
                            fa1 = -1

                        # Check whether a band is split
                        if fa2 < fa1:
                            split_flag[off_idx] = 1

                        fnorm_pre[2 * idx] = fa1
                        fnorm_pre[2 * idx + 1] = fa2

                    edge_distance[off_idx] = min(
                        [np.min(fnorm_pre[0::2] + 1), np.min(1 - fnorm_pre[1::2])]
                    )
                    if split_flag[off_idx] > 0.5:
                        edge_distance[off_idx] = 0

                # Find the optimal f_off
                idx_2 = np.where(nband == np.max(nband))[0]
                value_3, idx_3 = np.max(edge_distance[idx_2]), np.argmax(
                    edge_distance[idx_2]
                )
                if value_3 > threshold_edge:
                    f_off = f_off_test[idx_2[idx_3]]
                else:
                    idx_4 = np.where(nband >= (np.max(nband) - 1))[0]
                    idx_5 = np.argmax(edge_distance[idx_4])
                    f_off = f_off_test[idx_4[idx_5]]

    # If symmetric frequency response, then construct mirrored response, update number of bands
    if sym:
        f_h = f - f_off
        f = np.concatenate((-np.flip(f_h), f_h)) + f_off
        a = np.concatenate((np.flip(a), a))
        d = np.concatenate((np.flip(d), d))
        nf = len(f) // 2

    # Go through each band pair, determine aliased frequencies
    # in normalized space, and split if necessary
    new_idx = 0
    fnorm = np.zeros(f.shape)

    # Preallocate
    f_a = np.zeros(2 * nf + 4)
    a_a = np.zeros(nf + 2)
    d_a = np.zeros(nf + 2)

    for idx in range(nf):
        # Normalize frequencies
        fnorm[idx * 2] = (f[idx * 2] - f_off) / (fs / 2)
        fnorm[idx * 2 + 1] = (f[idx * 2 + 1] - f_off) / (fs / 2)

        # Get aliased frequencies
        fa1 = (fnorm[idx * 2] + 1) % 2 - 1
        fa2 = (fnorm[idx * 2 + 1] + 1) % 2 - 1

        # in case one band is interrupted into two ends of main spectrum
        # Check to see if endpoints can be shifted to form one single band
        if fa2 < fa1 and fa2 == -1:
            fa2 = 1
        if fa2 < fa1 and fa1 == 1:
            fa1 = -1

        if fa2 < fa1:  # % if still split, then create a new band
            f_a[new_idx * 2] = -1
            f_a[new_idx * 2 + 1] = fa2
            a_a[new_idx] = a[idx]
            d_a[new_idx] = d[idx]
            new_idx += 1

            f_a[new_idx * 2] = fa1
            f_a[new_idx * 2 + 1] = 1
            a_a[new_idx] = a[idx]
            d_a[new_idx] = d[idx]
            new_idx += 1
        else:
            f_a[new_idx * 2] = fa1
            f_a[new_idx * 2 + 1] = fa2
            a_a[new_idx] = a[idx]
            d_a[new_idx] = d[idx]
            new_idx += 1

    f_a = f_a[: 2 * new_idx]
    a_a = a_a[: 2 * new_idx]
    d_a = d_a[: 2 * new_idx]

    # Check for overlaps
    overlap_found = True
    incompatible_overlap = False

    while overlap_found and not incompatible_overlap and len(f_a) / 2 > 1:
        # Check each frequency band for overlap
        nf = int(len(f_a) / 2)

        # Preallocate
        f_tmp = np.zeros(2 * nf)
        a_tmp = np.zeros(nf)
        d_tmp = np.zeros(nf)

        for chk_idx in range(nf):
            # Copy frequency, ripple, amplitude
            f_tmp[2 * chk_idx] = f_a[2 * chk_idx]
            f_tmp[2 * chk_idx + 1] = f_a[2 * chk_idx + 1]
            d_tmp[chk_idx] = d_a[chk_idx]
            a_tmp[chk_idx] = a_a[chk_idx]

            # Check current band edges with all remaining bands for overlap
            fl_1 = f_a[2 * chk_idx]
            fu_1 = f_a[2 * chk_idx + 1]
            for rem_idx in range(chk_idx + 1, nf):
                # Initialize flags
                overlap_ok = False
                overlap_found = False

                # Check to see if overlap would be compatible, i.e., same amplitude
                if a_a[chk_idx] == a_a[rem_idx]:
                    overlap_ok = True

                # Check if band edges overlap
                fl_2 = f_a[2 * rem_idx]
                fu_2 = f_a[2 * rem_idx + 1]
                if fl_1 <= fl_2 and fu_1 >= fl_2:
                    # Fix overlapping band
                    f_tmp[2 * chk_idx] = fl_1
                    f_tmp[2 * chk_idx + 1] = max(fu_1, fu_2)
                    d_tmp[chk_idx] = min([d_a[chk_idx], d_a[rem_idx]])

                    # Copy other bands
                    idx = chk_idx
                    for cp_idx in range(chk_idx + 1, nf):
                        if cp_idx != rem_idx:
                            idx += 1
                            f_tmp[2 * idx] = f_a[2 * cp_idx]
                            f_tmp[2 * idx + 1] = f_a[2 * cp_idx + 1]
                            a_tmp[idx] = a_a[cp_idx]
                            d_tmp[idx] = d_a[cp_idx]

                    overlap_found = True

                    # cut
                    f_tmp = f_tmp[: 2 * idx + 2]
                    a_tmp = a_tmp[: idx + 1]
                    d_tmp = d_tmp[: idx + 1]

                    break  # Start over
                elif fu_1 >= fu_2 and fl_1 <= fu_2:
                    # Fix overlapping bands
                    f_tmp[2 * chk_idx] = min(fl_1, fl_2)
                    f_tmp[2 * chk_idx + 1] = fu_1
                    d_tmp[chk_idx] = min([d_a[chk_idx], d_a[rem_idx]])

                    # Copy other bands
                    idx = chk_idx
                    for cp_idx in range(chk_idx + 1, nf):
                        if cp_idx != rem_idx:
                            idx += 1
                            f_tmp[2 * idx] = f_a[2 * cp_idx]
                            f_tmp[2 * idx + 1] = f_a[2 * cp_idx + 1]
                            a_tmp[idx] = a_a[cp_idx]
                            d_tmp[idx] = d_a[cp_idx]

                    overlap_found = True

                    # cut
                    f_tmp = f_tmp[: 2 * idx + 2]
                    a_tmp = a_tmp[: idx + 1]
                    d_tmp = d_tmp[: idx + 1]

                    break  # Start over
                elif fl_1 >= fl_2 and fu_1 <= fu_2:
                    # Fix overlapping bands
                    f_tmp[2 * chk_idx] = fl_2
                    f_tmp[2 * chk_idx + 1] = fu_2
                    d_tmp[chk_idx] = min(d_a[chk_idx], d_a[rem_idx])

                    # Copy other bands
                    idx = chk_idx
                    for cp_idx in range(chk_idx + 1, nf):
                        if cp_idx != rem_idx:
                            idx += 1
                            f_tmp[2 * idx] = f_a[2 * cp_idx]
                            f_tmp[2 * idx + 1] = f_a[2 * cp_idx + 1]
                            a_tmp[idx] = a_a[cp_idx]
                            d_tmp[idx] = d_a[cp_idx]

                    overlap_found = True

                    # cut
                    f_tmp = f_tmp[: 2 * idx + 2]
                    a_tmp = a_tmp[: idx + 1]
                    d_tmp = d_tmp[: idx + 1]

                    break  # Start over

            if overlap_found:
                incompatible_overlap = not overlap_ok
                f_a = f_tmp
                d_a = d_tmp
                a_a = a_tmp
                break  # Start over from the beginning

    # If incompatible overlap, then return empty matrices
    if incompatible_overlap:
        return [], [], []

    # Sort frequencies into ascending order
    idx_sort = np.argsort(f_a)
    f_a = f_a[idx_sort]
    a_a = a_a[idx_sort[1::2] // 2]
    d_a = d_a[idx_sort[1::2] // 2]

    # If symmetric response, return only positive bands
    if sym:
        idx_pos = np.where(f_a[1::2] > 0)[0]
        f_tmp = []
        for idx in idx_pos:
            f_tmp.extend([f_a[2 * idx], f_a[2 * idx + 1]])
        f_a = np.maximum(0, np.array(f_tmp))
        a_a = a_a[idx_pos]
        d_a = d_a[idx_pos]

    return f_a, a_a, d_a, f_off
