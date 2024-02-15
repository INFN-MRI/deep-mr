"""Utils for pulse stats calculation."""

__all__ = ['calc_isodelay', 'calc_efficiency_profile']

import numpy as np
from scipy.interpolate import CubicSpline

# slice profile (from PyPulseq)
def calc_isodelay(rf, dt):
    """
    Calculate the time point of the effective rotation defined as the peak of the radio-frequency amplitude for the
    shaped pulses and the center of the pulse for the block pulses. Zero padding in the radio-frequency pulse is
    considered as a part of the shape. Delay field of the radio-frequency object is not taken into account.
    
    Args:
        rf (array): pulse envelope in time domain.
        dt: waveform dwell time in [s].
       
    Returns:
        isodelay : Time between peak and end of the radio-frequency pulse in [s].
        isodelay_idx : Corresponding position of `isodelay` in the radio-frequency pulse's envelope.
    """
    # get time
    t = dt * np.arange(len(rf))
    
    # We detect the excitation peak; if i is a plateau we take its center
    rf_max = max(abs(rf))
    i_peak = np.where(abs(rf) >= rf_max * 0.99999)[0]
    
    # get isodelay and corresponding index
    isodelay_idx = i_peak[len(i_peak) // 2]
    isodelay = t[-1] - t[isodelay_idx]

    return isodelay, isodelay_idx

# slice profile (from MyoQMRI)
def calc_efficiency_profile(rf):
    """
    Compute flip angle efficiency along z for a given pulse.
    
    Args:
        rf (array): pulse envelope in time domain.
        
    Returns:
        
    """
    # compute
    sliceprof = _calcSliceprof(rf)

    # reduce
    sliceprof = _reduceSliceProf(sliceprof, 15)

    # normalize
    sliceprof /= sliceprof[0]

    # flip
    sliceprof = np.flip(sliceprof)
    
    return np.ascontiguousarray(sliceprof)

# %% local utils    
def _calcSliceprof(pulse): 

    # alternative a bit less precise but errors ~1deg
    h = np.abs(np.fft.fft(pulse, 5210*2))
    
    sliceprof = np.abs(h[0:199])
    sliceprof = sliceprof
    
    return sliceprof

# binning of the slice profile
def _reduceSliceProf(sliceprof, nbins):
    lastVal = np.argwhere( sliceprof > 0.1 * sliceprof[0]).max()
    sliceprof = sliceprof[:lastVal]
    
    # interpolate
    x = np.linspace(0, 1, num=sliceprof.shape[0])
    xnew = np.linspace(0, 1, num=nbins)
    spl = CubicSpline(x, sliceprof)
    
    sliceprof_out = spl(xnew)
        
    return sliceprof_out