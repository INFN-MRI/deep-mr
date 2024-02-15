"""Main spsp pulse design routine."""

__all__ = ["ss_design"]

import numpy as np

from .rf_ripple import rf_ripple
from .ss_alias import ss_alias
from .ss_ep import ss_ep
from .ss_flyback import ss_flyback
from .ss_globals import ss_globals
from .ss_grad import grad_ss

def ss_design(z_thk, z_tb, z_de, f, a_angs, de, ptype='ex', z_ftype='pm', s_ftype='min', ss_type='Flyback Whole', f_off=None, sg=None, verbose=False):
    """
    Design spectral-spatial pulse.

    Args:
        z_thk (float): Slice thickness (cm).
        z_tb (float): Spatial time-bandwidth.
        z_de (list): Spatial ripples, [pass_ripple, stop_ripple].
        f (list): Spectral band edge specification (Hz).
        a_angs (list): Spectral band flip angle specification (radians).
        de (list): Spectral band ripples.
        ptype (str, optional): Spatial pulse type: 'ex' (default), 'se', 'sat', 'inv'.
        z_ftype (str, optional): Spatial filter type: 'ms', 'ls', 'pm' (default), 'min', 'max'.
        s_ftype (str, optional): Spectral filter type: 'min' (default), 'max', 'lin'.
        ss_type (str, optional): Spectral-spatial type: 'Flyback Whole' (default),
        'Flyback Half', 'EP Whole', 'EP Half', 'EP Whole Opp-Null', 'EP Half Opp-Null'.
        f_off (float, optional): Center frequency (empty to let ss_design choose).
        verbose (bool, optional): Print debug messages: 0-none (default), 1-yes.

    Returns:
        (tuple): A tuple containing the following elements:
            g (numpy.ndarray): Gradient (G/cm).
            rf (numpy.ndarray): RF (G).

    Examples:
        See scripts in examples/ folder demonstrations of how to use this function.
    """
    if verbose:
        dbg = 1
    else:
        dbg = 0
        
    if ptype not in {'ex', 'se', 'sat', 'inv'}:
        raise ValueError(f'Spatial pulse type (ptype) not recognized: {ptype}')

    if z_ftype not in ['ms', 'ls', 'pm', 'min', 'max']:
        raise ValueError('Spatial filter type (z_ftype) not recognized: {z_ftype}')

    if s_ftype not in ['min', 'max', 'lin']:
        raise ValueError(f'Spectral filter type (s_ftype) not recognized: {s_ftype}')

    if ss_type not in ['Flyback Whole', 'Flyback Half', 'EP Whole', 'EP Half', 'EP Whole Opp-Null', 'EP Half Opp-Null']:
        raise ValueError(f'Spectral-spatial type (ss_type) not recognized: {ss_type}')

    # Initialize globals
    if sg is None:
        sg = ss_globals()

    ang = np.max(a_angs)

    if sg.SS_SLR_FLAG:
        a = np.sin(a_angs / 2) / np.sin(ang / 2)
    else:
        a = a_angs / ang
    
    d, a, ang = rf_ripple(de, a, ang, ptype)
    z_d, _, _ = rf_ripple(z_de, [1, 0], ang, ptype)

    kz_max = z_tb / z_thk
    kz_area = kz_max / sg.SS_GAMMA

    if 'EP' in ss_type:
        sg.SS_EQUAL_LOBES = 1
    else:
        sg.SS_EQUAL_LOBES = 0

    gpos, gneg, _, _, _ = grad_ss(kz_area, None, sg.SS_VERSE_FRAC, sg.SS_MXG, sg.SS_MXS, sg.SS_TS, sg.SS_EQUAL_LOBES)

    lobe = np.concatenate((gpos, gneg))
    t_lobe = len(lobe) * sg.SS_TS
    fs_scale = 1

    if 'EP' in ss_type:
        fs_max = 2 / t_lobe
        if 'Opp-Null' not in ss_type:
            fs_scale = 1 / 2
    else:
        fs_max = 1 / t_lobe
        
    if 'Half' in ss_type:
        sym_flag = True
    else:
        sym_flag = False

    fdiff = np.diff(f)
    fwidths = np.sort(fdiff[0::2])[::-1]
    fs_min = np.sum(fwidths[0:2]) / 2

    df = (fs_max - fs_min) / (sg.SS_NUM_FS_TEST - 1)
    fs_test = np.zeros(sg.SS_NUM_FS_TEST)
    fs_ok = np.zeros(sg.SS_NUM_FS_TEST)
        
    for idx in range(sg.SS_NUM_FS_TEST):
        fs_test[idx] = fs_min + idx * df
        nsamp = np.ceil(1 / (fs_test[idx] * sg.SS_TS))
        fs_test[idx] = 1 / (nsamp * sg.SS_TS)
        tmp = ss_alias(f, a, d, f_off, fs_test[idx] * fs_scale, sym_flag)
        f_a = tmp[0]

        if len(f_a) == 0:
            fs_ok[idx] = 0
        else:
            fs_ok[idx] = 1

    if not np.any(fs_ok == 1):
        fs_over = fs_test[-1]

        while f_a:
            fs_over = fs_over + df
            tmp = ss_alias(f, a, d, f_off, fs_over * fs_scale, sym_flag)
            f_a = tmp[0]

        if verbose:
            print('ss_design: Incompatible aliasing of frequency spec')
            print('at all tested frequencies.')
            print('Current max sampling is: %6.1f' % fs_max)
            print('Estimated required sampling is: %6.1f' % fs_over)
            print('Try any of the following to increase:')
            print(' - Decrease spatial TBW')
            print(' - Increase slice thickness')
            print(' - Increase VERSE fraction')
            print(' - Decrease frequency band widths')

        if ss_type in {'Flyback Whole', 'Flyback Half'}:
            if verbose:
                print(' - Try EPI type SS')
            else:
                pass

        raise Exception('No good fs')
    
    fs_bands_left = np.where(np.diff(np.concatenate(([0], fs_ok))) == 1)[0]
    fs_bands_right = np.where(np.diff(np.concatenate((fs_ok, [0]))) == -1)[0]
    fs_bands = np.concatenate((fs_bands_left, fs_bands_right))

    # Iterate on lone width trying to meet B1 requirements with minimum-time pulse
    nsolutions = 0

    if sg.SS_NUM_LOBE_ITERS == 1:
        fs_top = fs_bands[-1]
        fs_best = fs_test[fs_top]
    
        if 'Flyback' in ss_type:
            rf, g = ss_flyback(ang, z_thk, z_tb, z_d, f, a, d, fs_best, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
        else:
            rf, g = ss_ep(ang, z_thk, z_tb, z_d, f, a, d, fs_best, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
    
        if rf is not None:
            b1_best = np.max(np.abs(rf))
            dur_best = len(rf) * sg.SS_TS
            pow_best = np.sum(np.abs(rf) ** 2) * sg.SS_TS
            nsolutions = 1
            if verbose:
                print('Solution(s) exists!')
                print(f'Fs: {fs_best:6.1f} B1: {b1_best:5.3f}G Power: {pow_best:5.3e} G^2 ms Dur: {dur_best * 1e3:4.1f}ms')
        else:
            if verbose:
                print(f'Fs: {fs_best:6.1f} *** No Solution ***')
            else:
                pass
    else:
        if verbose:
            print('Iterating on spectral sampling frequency to reduce B1')
    
        dur_best = float('inf')
        b1_best = float('inf')
        nbands = len(fs_bands) // 2
        
        # lists
        rfall = []
        gall = []
        fsall = []
        infoall = []

        for band in range(nbands-1, -1, -1):
            fs_bot = fs_bands[band * 2]
            fs_top = fs_bands[band * 2 - 1]
            d_idx = int(np.floor((fs_top - fs_bot + 1) / (sg.SS_NUM_LOBE_ITERS - 1)))
            d_idx = max(1, d_idx)
            niter = int(np.ceil((fs_top - fs_bot + 1) / d_idx))
                
            for iter in range(niter-1, -1, -1):
                idx = fs_bot + iter * d_idx
                if iter == niter-1:
                    idx = fs_top
                fs = fs_test[idx]
                if 'Flyback' in ss_type:
                    rf, g = ss_flyback(ang, z_thk, z_tb, z_d, f, a, d, fs, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
                else:
                    rf, g = ss_ep(ang, z_thk, z_tb, z_d, f, a, d, fs, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
    
                if len(rf) == 0:
                    if verbose:
                        print(f'Band: {nbands - band}/{nbands} Iter: {niter - iter}/{niter} Fs: {fs:6.1f} *** No Soln ***')
                    continue

                # append current values
                rfall.append(rf)
                gall.append(g)
                fsall.append(fs)
    
                b1 = np.max(np.abs(rf))
                dur = len(rf) * sg.SS_TS
                pow = np.sum(np.abs(rf) ** 2) * sg.SS_TS
    
                infoall.append(f'Fs: {fs:6.1f} B1: {b1:5.3f}G Power: {pow:5.3e} G^2 ms Dur: {dur * 1e3:4.1f}ms')
                if verbose:
                    print(f'Band: {nbands - band}/{nbands} Iter: {niter - iter}/{niter} {infoall[nsolutions]}')
    
                if b1 <= sg.SS_MAX_B1 and dur < dur_best:
                    b1_best = b1
                    dur_best = dur
                    Ibest = nsolutions
                elif b1_best > sg.SS_MAX_B1 and b1 < b1_best:
                    b1_best = b1
                    dur_best = dur
                    Ibest = nsolutions
                
                # update
                nsolutions += 1
    
        if nsolutions > 0:
            if verbose:
                print('\n')
                print('Solution(s) exists!')
                for n in range(nsolutions):
                    print(f'{n}) {infoall[n]}')
            if nsolutions > 1:
                # Isolution = int(input('Which pulse would you like to use? (leave empty for shortest pulse) '))
                # if not Isolution or Isolution < 1 or Isolution > nsolutions:
                Isolution = Ibest
                if verbose:
                    print(f'Returning {infoall[Isolution]}')
            else:
                Isolution = 1
            rf = rfall[Isolution]
            g = gall[Isolution]
            fs_best = fsall[Isolution]
    
    if len(rf) == 0:
        if verbose:
            print('No solution found! Trying to increase band ripples to determine limiting')
            print('frequency specifications...')
    
        orig_min_order = sg.SS_MIN_ORDER
        sg.SS_MIN_ORDER = 0
        fs_top = fs_bands[-1]
        fs_best = fs_test[fs_top]
    
        d_max = np.ones(len(d))
        d_min = d
        tol_factor = 4
    
        for Id in range(0, len(d)):
            d_test = np.copy(d)
    
            while d_max[Id] - d_min[Id] > d[Id] / tol_factor:
                d_test[Id] = (d_max[Id] + d_min[Id]) / 2
    
                if 'Flyback' in ss_type:
                    rf, _ = ss_flyback(ang, z_thk, z_tb, z_d, f, a, d_test, fs_best, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
                else:
                    rf, _ = ss_ep(ang, z_thk, z_tb, z_d, f, a, d_test, fs_best, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
    
                if rf is not None:
                    d_max[Id] = d_test[Id]
                else:
                    d_min[Id] = d_test[Id]
    
        # find ripple values that create solution
        Ifix = np.where(d_max < 1)[0]
        if len(Ifix) > 0:
            Imin = np.argmin(d_max - d)
            d_fix = np.copy(d)
            d_fix[Ifix] = d_max[Ifix]
            if verbose:
                print('Solution found by increasing ripples in bands %s' % ' '.join(map(str, Ifix)))
                print('Returning pulse with increased ripple in band %d (%.1f to %.1f Hz):' % (Imin, f[2 * Imin - 2], f[2 * Imin - 1]))
        
            if 'Flyback' in ss_type:
                rf, g = ss_flyback(ang, z_thk, z_tb, z_d, f, a, d_fix, fs_best, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
            else:
                rf, g = ss_ep(ang, z_thk, z_tb, z_d, f, a, d_fix, fs_best, ptype, z_ftype, s_ftype, ss_type, f_off, dbg, sg)
    
            b1_best = np.max(np.abs(rf))
            dur_best = len(rf) * sg.SS_TS
            pow_best = np.sum(np.abs(rf) ** 2) * sg.SS_TS
            if verbose:
                print('Fs: %6.1f B1: %5.3fG Power: %5.3e G^2 ms Dur: %4.1fms' % (fs_best, b1_best, pow_best, dur_best * 1e3))
                print('\nPulse specs should be modified by reducing bandwidths or increasing ripple in bands %s' % ' '.join(map(str, Ifix)))
                print('Increasing the max pulse duration or slice thickness may also help')
    
        else:
            raise Exception('No solution found! Try reducing bandwidths, increasing ripple, increasing max duration, increasing slice thickness...')
    
        sg.SS_MIN_ORDER = orig_min_order
    
    return g, rf, fs_best




