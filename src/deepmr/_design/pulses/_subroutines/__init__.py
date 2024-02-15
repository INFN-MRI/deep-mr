"""Main RF pulse design subroutines."""

from . import _slr
from . import _spsp

from ._slr import * # noqa
from ._spsp import * # noqa

__all__ = []
__all__.extend(_slr.__all__)
__all__.extend(_spsp.__all__)

# import numpy as np


# from mridesign.src.pulses import _adiabatic,  _multiband, _slr
                                    

# from mridesign.src.pulses._adiabatic import *


# __all__ = ['shinnar_leroux']
# __all__.extend(_adiabatic.__all__)


# def shinnar_leroux(dur: float = 1.8e-3, tbw: int = 4, 
#                    ptype: str = 'st', ftype: str = 'min', 
#                    n_bands: int = 1, band_sep: float = None, slice_sep: float = None, beta: float = None,
#                    passband_ripple_level: float = 0.01, stopband_ripple_level: float = 0.01, 
#                    cancel_alpha_phs=False, phs_0_pt = 'None', dt: float = 8e-6, flip: float = None):
#     """
#     Design a single or multi-band pulse using Shinnar-LeRoux algorithm
    
#     Args:
#         dur: pulse duration in [s].
#         tbw: pulse time bandwidth product (min: 2; max: 10).
#         ptype: pulse type:
#                     - 'st' (small-tip excitation);
#                     - 'ex' (pi / 2 excitation pulse); 
#                     - 'se' (spin-echo pulse);
#                     - 'inv' (inversion):
#                     - 'sat' (pi/2 saturation pulse).                  
#         ftype: type of filter to use: 
#                     - 'ms' (sinc); 
#                     - 'pm' (Parks-McClellan equal-ripple); 
#                     - 'min' (minphase using factored pm);
#                     - 'max' (maxphase using factored pm);
#                     - 'ls' (least squares);                     
#         n_bands: number of bands (default: single band).
#         band_sep: (for multi-band pulses) band separation in Hz. 
#                    Provide either this or band separation in unit of # slices (not both!).
#         slice_sep: (for multi-band pulses) band separation in unit of # slices. 
#                    Provide either this or band separation in Hz (not both!).
#         beta: (for multi-band pulses) ratio of off-resonant to on-resonant power 
#               (default: equal power for each band).
#         passband_ripple_level: passband ripple level in :math:'M_0^{-1}'.
#         stopband_ripple_level: stopband ripple level in :math:'M_0^{-1}'.
#         cancel_alpha_phs: For 'ex' pulses, absorb the alpha phase profile 
#                           from beta's profile for a flatter total phase.
#         phs_0_pt: set of phases to use for multiband. Can be:
#                     - 'phs_mod' (Wong); 
#                     - 'amp_mod' (Malik);
#                     - 'quad_mod' (Grissom);
#                     - 'None'.
#         dt: RF dwell time in [s].
#         flip: Nominal RF flip angle in [deg]. If not specified, assume 90° pulses for 'st' and 'ex' pulses and 180° for 'se', 'inv' and 'sat'.

#     band_sep = slice_sep / slice_thick * tb, where tb is time-bandwidth product
#     of the single-band pulse

#     Returns:
#         rf: designed RF pulse.

#     References:
#         Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
#         Parameter Relations for the Shinnar-LeRoux Selective Excitation
#         Pulse Design Algorithm.
#         IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
#         Wong, E. (2012). 'Optimized Phase Schedules for Minimizing Peak RF
#         Power in Simultaneous Multi-Slice RF Excitation Pulses'. Proc. Intl.
#         Soc. Mag. Reson. Med., 20 p. 2209.
#         Malik, S. J., Price, A. N., and Hajnal, J. V. (2015). 'Optimized
#         Amplitude Modulated Multi-Band RF pulses'. Proc. Intl. Soc. Mag.
#         Reson. Med., 23 p. 2398.
#     """
#     # check tbw
#     assert type(tbw) is int, "Time-Bandiwidth-Product = {tbw} must be integer"
#     assert tbw >= 2 and tbw <= 10, "Time-Bandiwidth-Product = {tbw} must be between 2 and 10"
    
#     rf_pulse = _slr.design_singleband_pulse(dur, tbw, ptype, ftype, passband_ripple_level, stopband_ripple_level, cancel_alpha_phs, dt, flip)
    
#     # remove first and last point
#     rf_pulse[0] = 0
#     rf_pulse[-1] = 0
#     rf_pulse = np.concatenate((rf_pulse, 0 * rf_pulse[-2:]))

#     # recompute duration
#     dur = dt * len(rf_pulse)

#     # add bands
#     if n_bands != 1:
#         rf_pulse = _multiband.make_multiband_pulse(rf_pulse, dt, tbw / dur, n_bands, band_sep, slice_sep, beta, phs_0_pt)
    
#     # cast to complex 64
#     rf_pulse = rf_pulse.astype(np.complex64)
    
#     # clean phase
#     if np.allclose(rf_pulse.imag, 0):
#         rf_pulse = rf_pulse.real
        
#     return rf_pulse
