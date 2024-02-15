"""Numerical routines to generate arbitrary gradient waveforms."""

__all__ = ["make_trapezoid", "make_cartesian_gradient", "make_crusher", "make_arbitrary_gradient", "compose_gradients", "derivative", "integration"]
          
import math
import warnings

from typing import Tuple

import numpy as np
import numba as nb

from scipy import interpolate
from scipy import integrate

from .misc import pad

gamma_bar = 42.575 * 1e6 # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar # rad / T / us -> rad / T / s

def make_trapezoid(area: float, gmax: float, smax: float, gdt: float, g_start: float = 0.0, rampsamp = False, npts=None):
    r"""
    General trapezoidal gradient designer for total target area (for readout and rewinders).
    
    Args:
        area: pulse area [rad / m].
        gmax: maximum gradient [mT / m].
        smax: max slew rate [T / m / s].
        gdt: sample time [us].
        g_start: initial value of gradient in [T / m] (default: 0).
        rampsamp: if False, gradient's flat part only is considered to compute area.
        
    Returns:
        (ndarray): gradient waveform in mT / m.
    """    
    # units
    gmax *= 1e-3 # [mT/m] -> [T/m]
    gdt *= 1e-6 # [us] -> [s]
            
    if np.abs(area) > 0:
                
        # if initial value is not zero, ramp to zero
        if g_start != 0:
            g0, delta_area = _make_ramp(g_start, g_end=0.0, smax=smax, gdt=gdt)
            area += delta_area
        else:
            g0 = None
        
        if rampsamp:
            ramppts = int(np.ceil(gmax / smax / gdt))
            triareamax = ramppts * gdt * gmax * gamma # rad / m
            
            if triareamax > np.abs(area):
                # triangle pulse
                newgmax = np.sqrt(np.abs(area / gamma) * smax)
                ramppts = int(np.ceil(newgmax / smax / gdt))
                ramp_up = np.linspace(0, ramppts, num=ramppts+1) / ramppts
                ramp_dn = np.linspace(ramppts, 0, num=ramppts+1) / ramppts
                pulse = np.concatenate((ramp_up, ramp_dn))
            else:
                # trapezoid pulse
                nflat = int(np.ceil(abs(area - triareamax) / gamma / gmax / gdt / 2) * 2)
                ramp_up = np.linspace(0, ramppts, num=ramppts+1) / ramppts
                ramp_dn = np.linspace(ramppts, 0, num=ramppts+1) / ramppts
                pulse = np.concatenate((ramp_up, np.ones(nflat), ramp_dn))
                
            # normalize
            trap = (pulse / pulse.sum()) * (area / gamma / gdt)
            ramppts = None
            
        else:
            trap, ramppts = _min_trap_grad(area, gmax, smax, gdt, npts)

        # append ramp
        if g0 is not None:
            trap = np.concatenate((g0, trap))
            
        # cast to mT / m
        trap *= 1e3 # T / m -> mT / m
    
    else:
        trap = np.atleast_1d([]).astype(np.float32)
        ramppts = 0
    
    return trap, ramppts

def make_cartesian_gradient(k, gmax, smax, gdt, rew_derate, fid=(0, 0), flyback=False):
    r"""
    Wrapper for make_trapezoid to build cartesian readout gradients.
    
    Build gradient waveform sampling a k-space line defined by fov and matrix size,
    satisfying physical system constraint.
    
    Args:
        k: input trajectory of shape (npts, ) in units of [rad / m].
        gmax: maximum gradient [mT / m].
        smax: max slew rate [T / m / s].
        gdt: sample time [us].
        rew_derate: slew rate derating for pns management.
        fid: number of dummy points before (fid[0]) and after (fid[1]) gradient wave.
        flyback: if True, rewind after each echo (only for nechoes > 1).
        osf: readout oversampling factor.
         
    Returns:
        (ndarray): gradient waveform in [mT / m].
        (ndarray): boolean array which is True along the readout and False elsewhere (fid and pre-rewinders).
        
    """ 
    # get kmax
    kmin = k[:, 0].min()
    kmax = k[:, -1].max()
    
    # compute gradient
    if gmax is not None and smax is not None :
        
        # calculate waveform
            
    # full spoke: area must cover 2 * kmax
        if kmin == 0:
            grad, ramppts = _min_trap_grad(kmax, gmax * 1e-3, smax, gdt * 1e-6, k.shape[0]) # noqa
        else:
            grad, ramppts = _min_trap_grad(2 * kmax, gmax * 1e-3, smax, gdt * 1e-6, k.shape[0]) # noqa

        grad = grad * 1e3 # mT / m
        
        # calculate echo index
        if kmin == 0:
            echo_idx = ramppts
        else:
            echo_idx = grad.shape[0] // 2
                
        # build adc
        adc = np.ones(grad.shape[0], dtype=int)
        adc[:ramppts+1] = 0
        adc[-ramppts-1:] = 0
        
        # add flyback if requested
        if flyback:
            if kmin == 0:
                gfly, _ = make_trapezoid(-kmax, gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
            else:
                gfly, _ = make_trapezoid(-2 * kmax, gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
        else:
            gfly = np.atleast_1d(np.asarray([], dtype=np.float32))
            
        # build pre / rewinder
        if kmin == 0:
            triarea = gamma * rew_derate * smax * ramppts * gdt * 1e-6 # rad/m
            gpre, _ = make_trapezoid(-triarea, 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
        else:
            gpre, _ = make_trapezoid(-kmax, 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
                    
        # add dummies
        grad = np.pad(grad, tuple(fid))
        adc = np.pad(adc, tuple(fid))
                
        # update echo_idx
        echo_idx += (fid[0] + gpre.shape[-1])
        
        # safety checks
        # gradient
        act_gmax = grad[0].max()
        if act_gmax > gmax:
            warnings.warn(f"actual gmax = {act_gmax} [mT/m] exceed maximum gradient = {gmax} [mT/m]")
        
        # slew rate
        s0 = derivative(grad, gdt, axis=-1, direction=-1) * 1e3 # T / m / s
        act_smax = s0[0].max()
        if act_smax > smax:
            warnings.warn(f"actual smax = {act_smax} [T/m/s] exceed maximum slew rate = {smax} [T/m/s]")
            
        # put together
        grad = {"read": grad, "pre": gpre, "flybck": gfly}           
    else:
        grad = None
        adc = None
        if kmin == 0:
            echo_idx = 0
        else:
            echo_idx = int(k.shape[-1] // 2)
        
    return grad, adc, echo_idx

def make_crusher(ncycles, voxelsize, gmax, smax, gdt):
    r"""
    Wrapper for make_trapezoid to build crushers.
    
    Build gradient waveform sampling a k-space line defined by fov and matrix size,
    satisfying physical system constraint.
    
    Args:
        ncycles: number of cycles per voxel.
        voxelsize: voxelsize in mm.
        gmax: maximum gradient [mT / m].
        smax: max slew rate [T / m / s].
        gdt: sample time [us].
                 
    Returns:
        (ndarray): gradient waveform in [mT / m].
        
    """
    area = ncycles * np.pi / (voxelsize * 1e-3)
    gcrush, _ = make_trapezoid(area, gmax, smax, gdt, rampsamp=True) # noqa
    
    return gcrush
  
def make_arbitrary_gradient(k, gmax, smax, gdt, rew_derate, fid=(0, 0), balance=False):
    r"""
    Wrapper for min_time_gradient.
    
    Build gradient waveform corresponding to trajectory 'k' while 
    satisfying physical system constraint.
    
    Args:
        k: input trajectory of shape (ndim, ...) in units of [rad / m].
        gmax: maximum gradient [mT / m].
        smax: max slew rate [T / m / s].
        gdt: sample time [us].
        rew_derate: slew rate derating for pns management.
        fid: number of dummy points before (fid[0]) and after (fid[1]) gradient wave.
        balance: if True, append rewinders.
        
    Returns:
        (ndarray): actual trajectory corresponding to system limit in [rad / m].
        (ndarray): gradient waveform in [mT / m].
        (ndarray): boolean array which is True along the readout and False elsewhere (fid and pre-rewinders).
        
    """
    if gmax is not None and smax is not None :
        # calculate waveform
        k, grad = _min_time_gradient(k.T, 0.9 * gmax, 0.8 * smax, gdt, g_start=0.0, g_end=0.0) # noqa
        
        # calculate echo index
        echo_idx = _find_echo(k)
                
        # build adc
        adc = np.ones(grad.shape[1], dtype=bool)
        
        # build rewinder
        if balance:
           
            # rewinder
            grewx, _ = make_trapezoid(-k[0][-1], 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
            grewy, _ = make_trapezoid(-k[1][-1], 0.9 * gmax, rew_derate * smax, gdt, rampsamp=True) # noqa
                        
            # get length of rewinder
            grew_length = max(grewx.shape[-1], grewy.shape[-1])
            
            # pad rewinder
            grewx = pad(grewx, grew_length)
            grewy = pad(grewy, grew_length)
                        
            # build rewinder and append to gradient
            grew = np.stack((grewx, grewy), axis=0)
                    
        # add dummies
        grad = np.pad(grad, ((0, 0), tuple(fid)))
        adc = np.pad(adc, tuple(fid))
        
        # update echo_idx
        echo_idx += fid[0]
        
        # safety checks
        # gradient
        if balance:
            gcheck = np.concatenate((grad, grew), axis=-1)
        else:
            gcheck = grad
        act_gmax = ((gcheck**2).sum(axis=0)**0.5).max()
        if act_gmax > gmax:
            warnings.warn(f"actual gmax = {act_gmax} [mT/m] exceed maximum gradient = {gmax} [mT/m]")
        
        # slew rate
        s0 = derivative(gcheck, gdt, axis=-1, direction=-1) * 1e3 # T / m / s
        act_smax = ((s0**2).sum(axis=0)**0.5).max()
        if act_smax > smax:
            warnings.warn(f"actual smax = {act_smax} [T/m/s] exceed maximum slew rate = {smax} [T/m/s]")
            
        # balanced
        if balance:
            k0 = integration(gcheck * 1e-3, gamma * gdt * 1e-6, axis=-1) # mT
            act_kend = abs(k0[:, -1]).max()
            if act_kend > 1.0:
                warnings.warn(f"max end k = {act_kend} [rad/m]: failed to balance!")
                
        # put together
        if balance:
            grad = {"read": grad, "rew": grew, "pre": None}
        else:
            grad = {"read": grad, "rew": None, "pre": None}
    else:
        grad = None
        adc = None
        echo_idx = _find_echo(k)
        
    return k, grad, adc, echo_idx


def compose_gradients(side, gx=None, gy=None, gz=None, **kwargs):
    """
    Pad and stack gradients.
    """
    # parse gdt
    if "gdt" in kwargs:
        gdt = kwargs["gdt"] * 1e-3 # ms
    else:
        gdt = 0.0
    if gx is None:
        gx = np.atleast_1d(np.asarray([], dtype=np.float32))
    if gy is None:
        gy = np.atleast_1d(np.asarray([], dtype=np.float32))
    if gz is None:
        gz = np.atleast_1d(np.asarray([], dtype=np.float32))
        
    # get length of rewinder
    grad_length = max(gx.shape[-1], gy.shape[-1], gz.shape[-1])
    
    # get delta_te
    delta_te = gdt * (grad_length - gx.shape[-1]) # >= 0
    
    # pad rewinder
    gx = pad(gx, grad_length, side)
    gy = pad(gy, grad_length, side)
    gz = pad(gz, grad_length, side)
                
    # build rewinder and append to gradient
    return np.stack((gx, gy, gz), axis=0), delta_te
       
def derivative(input, dp=1, axis=0, direction=1):
    """
    Compute numerical finite difference derivative.
    """
    # first swap axis
    input = input.swapaxes(axis, 0)
    
    if direction > 0: # forward difference
        output = (np.concatenate((input[1:], input[[-1]])) - input) / dp
        output[-1] = output[-2]
        
    elif direction < 0: # backward difference
        output = (input - np.concatenate((input[[0]], input[:-1]))) / dp
        output[0] = output[1]
        
    # swap back
    input = input.swapaxes(axis, 0)
    output = output.swapaxes(axis, 0)
    
    return output

def integration(input, dp=1, axis=0):
    """
    Compute numerical integration using trapezoidal rule.
    """
    return integrate.cumtrapz(input, initial=0, axis=axis) * dp

#%% local sub-routines
def _find_echo(k): # noqa
    # k (ndim, ...)
    
    # find kabs
    kabs = (k**2).sum(axis=0)**0.5
    kabs = kabs.flatten()
    
    return np.where(kabs==0)[0][0]
    
    
def _make_ramp(g_start: float, g_end: float, smax: float, gdt: float) -> Tuple[np.ndarray, float]:
    """
    Built a gradient ramp connecting g_start and g_end.
    
    Args:
        g_start: initial gradient value [T / m].
        g_end: final gradient value [T / m].
        smax: max slew rate [T / m / s].
        gdt: sample time [s].
    
    Returns:
        2-element tuple containing
        - **ramp**: gradient waveform in [T / m].
        - **area**: area of the ramp gradient in [T * s / m].
    """
    if g_start != g_end:
        # compute number of points given height and slew rate
        delta_g = g_start - g_end
        npts = int(abs(np.ceil(delta_g / smax / gdt)))
    
        # build gradient
        ramp = np.linspace(g_start, g_end, npts)
        
        # compute area
        area = ramp.sum() * gdt * gamma
    else:
        ramp = None
        area = 0.0
    
    return ramp, area

def _min_trap_grad(area: float, gmax: float, smax: float, gdt: float, pts: int = None):
    r"""
    Minimal duration trapezoidal gradient designer. Design for target area
    under the flat portion (for non-ramp-sampled pulses).
    
    Args:
        area: pulse area [rad / m].
        gmax: maximum gradient [T / m].
        smax: max slew rate [T / m / s].
        gdt: sample time [s].
        pts: number of points in the flat part (if None, assume minimum duration).

    Returns:
        (ndarray): gradient waveform in [T / m].
    """    
    # get sign
    if np.abs(area) > 0:      
        if pts is None:
            # we get the solution for plateau amp by setting derivative of
            # duration as a function of amplitude to zero and solving
            a = np.sqrt(smax * np.abs(area / gamma) / 2)
            
            # finish design with discretization
            # make a flat portion of magnitude a and enough area for the swath
            pts = int(np.floor(np.abs(area / gamma) / a / gdt))
        else:
            a = np.abs(area / gamma) / pts / gdt

        flat = np.ones((1, pts))
        flat = flat / np.sum(flat) * np.abs(area / gamma) / gdt
        if np.max(flat) > gmax:
            flat = np.ones((1, int(np.ceil(np.abs(area / gamma) / gmax / gdt))))
            flat = flat / np.sum(flat) * np.abs(area / gamma) / gdt

        # make attack and decay ramps
        ramppts = int(np.ceil(np.max(flat) / smax / gdt))
        ramp_up = np.linspace(0, ramppts, num=ramppts+1) / ramppts * np.max(flat)
        ramp_dn = np.linspace(ramppts, 0, num=ramppts+1) / ramppts * np.max(flat)

        trap = np.concatenate((ramp_up, np.squeeze(flat), ramp_dn))
               
    else:
        trap = 0.0
        ramptts = 0

    return trap, ramppts

def _min_time_gradient(c: np.ndarray, gmax: float, smax: float, gdt: float, g_start: float = 0.0, g_end: float = None, verbose: bool = False):
    r"""
    Given a k-space trajectory c(n), gradient and slew constraints. This
    function will return a new parametrization that will meet these
    constraint while getting from one point to the other in minimum time.
    
    Args:
        c: Curve in k-space given in any parametrization [rad / m] (..., ndim) real array.
        gmax: Maximum gradient [mT / m].
        smax: Maximum slew [T / m / s]
        gdt: Sampling time interval [us].
        g_start: Initial gradient amplitude (leave empty for g0 = 0) in [T / m].
        g_end: Gradient value at the end of the trajectory. If not
                possible, the result would be the largest possible
                ampltude. (Leave empty if you don't care to get
                maximum gradient.).
        
    Returns:
        tuple: (g, k, s, t) tuple containing
        - **g** : gradient waveform [T / m]
        - **k** : exact k-space coordinates corresponding to gradient g in  [rad / m].
        - **s** : slew rate [T / m / s]
        - **time** :  sampled time [ms].
        
    References:
        Lustig M, Kim SJ, Pauly JM. A fast method for designing time-optimal
        gradient waveforms for arbitrary k-space trajectories. IEEE Trans Med
        Imaging. 2008;27(6):866-873. doi:10.1109/TMI.2008.922699
        
    Note:
        Arc length code translated from matlab
          (c) Michael Lustig 2005
          modified 2006 and 2007
          Rewritten in Python in 2020 by Kevin Johnson
          Modified in 2023 by Matteo Cencini
    """    
    # units
    gmax *= 1e-3 # [mT/m] -> [T/m]
    gdt *= 1e-6 # [us] -> [s]
    
    # fix dim
    c = c.copy()
    c = c.reshape(-1, c.shape[-1])
    
    # get ndim
    ndim = c.shape[-1]
    
    # stack z axis if it is not present
    if ndim == 1:
        cropdim = True
        c = np.concatenate((c, 0 * c[..., [-1]], 0 * c[..., [-1]]), axis=-1)        
    elif ndim == 2:
        cropdim = True
        c = np.concatenate((c, 0 * c[..., [-1]]), axis=-1)
    else:
        cropdim = False
                
    # rescale gyromagnetic factors
    _gamma_bar = gamma_bar * 1e-6 # Hz / T -> MHz / T
    _gamma_bar /= 10
    
    # convert units
    c = c / (2 * np.pi * 100) # rad / m -> 1 / cm
    gmax *= 100 # T / m -> G / cm
    smax /= 10 # T / m / s -> G / cm / s
    gdt *= 1e3 # s -> ms
             
    if verbose:
        print('Const arc-length parametrization')
        
    # represent the curve using spline with parametrization p
    num_p = c.shape[0]
    p = np.linspace(0, num_p-1, num_p)
    pp = interpolate.CubicSpline(p, c, axis=0)

    # interpolate curve for gradient accuracy
    dp = 1e-1
    cc = pp(np.linspace(0, num_p-1, int((num_p - 1) / dp + 1)))

    # find length of the curve
    cp = derivative(cc, dp)
    s_of_p = integration(np.linalg.norm(cp, axis=1), dp)
    curve_length = s_of_p[-1]

    # decide ds and compute st for the first point
    stt0 = (_gamma_bar * smax)  # always assumes first point is max slew
    st0 = stt0 * gdt / 2  # start at 1/2 the gradient for accuracy close to g=0
    s0 = st0 * gdt
    ds = s0 / 1.5  # smaller step size for numerical accuracy

    # s is arc length at high resolution
    s = np.linspace(0, curve_length, int(curve_length / ds))
    s_half = np.linspace(0, curve_length, 2 * int(curve_length / ds))
    
    # get the start
    sta = 0 * s
    sta[0] = min(g_start * _gamma_bar + st0, _gamma_bar * gmax)

    # cubic spline at s positions (s of p)
    interp1 = interpolate.CubicSpline(s_of_p, np.linspace(0, num_p-1, int((num_p - 1) / dp + 1)))
    p_of_s_half = interp1(s_half)
    p_of_s = p_of_s_half[::2]
        
    if verbose:
        print('Compute geometry dependent constraints')

    # compute constraints (forbidden line curve)
    phi, k = _sdotmax(pp, p_of_s_half, s_half, gmax, smax)
    k = np.pad(k, (0, 2), 'edge')  # extend for the Runge-Kutte method

    if verbose:
        print('Solve ODE forward')

    # solve ODE forward
    _forward_ode_solver(sta, s, ds, k, phi, smax)
    
    # prepare for backward            
    stb = 0 * s
    if g_end is None:
        stb[-1] = sta[-1]
    else:
        stb[-1] = min(max(g_end * _gamma_bar, st0), _gamma_bar * gmax)
        
    if verbose:
        print('Solve ODE backwards')

    # solve ODE backwards
    _backward_ode_solver(stb, s, ds, k, phi, smax)
    
    # fix last point which is indexed a bit off
    kpos = [2, 1, 0]
    dstds = _runge_kutta(ds, stb[1], k[kpos], smax)
    if dstds is None:
        stb[0] = phi[0]
    else:
        tmpst = stb[1] + dstds
        stb[0] = min(tmpst, phi[0])
        
    if verbose:
        print('Final Interpolations')

    # take the minimum of the curves
    ds = s[1] - s[0]
    st_of_s = np.minimum(sta, stb)

    # compute time
    t_of_s = integration(1. / st_of_s, ds)
    t = np.linspace(0, t_of_s[-1], int(t_of_s[-1] / gdt))
    
    # compute curve
    interp1 = interpolate.CubicSpline(t_of_s, s)
    s_of_t = interp1(t)
    interp1 = interpolate.CubicSpline(s, p_of_s)
    p_of_t = interp1(s_of_t)    
    c = np.squeeze(pp(p_of_t))
    
    # compute gradient
    g = derivative(c, _gamma_bar * gdt, direction=-1)
        
    # compute trajectory
    k = integration(g, _gamma_bar * gdt, axis=0)
    
    # # fix dim
    k = np.ascontiguousarray(k.transpose())
    g = np.ascontiguousarray(g.transpose())

    if cropdim:
        k = k[:ndim]    
        g = g[:ndim]
    
    # rescale
    k = k * (2 * np.pi * 100) # 1 / cm -> rad / m
    g *= 10 # G / cm -> mT / m
        
    return k, g

def _sdotmax(pp: interpolate.CubicSpline, p_of_s: np.ndarray, s: np.ndarray, gmax: float, smax: float):
    """
    # [sdot, k, ] = sdotMax(PP, p_of_s, s, gmax, smax)
    #
    # Given a k-space curve C (in [1/cm] units), maximum gradient amplitude
    # (in G/cm) and maximum slew-rate (in G/(cm*ms)).
    # This function calculates the upper bound for the time parametrization
    # sdot (which is a non scaled max gradient constaint) as a function
    # of s.
    #
    #   pp      --  spline polynomial
    #   p_of_s  --  parametrization vs arclength
    #   s       --  arclength parametrization (0->1)
    #   gmax    --  maximum gradient (G/cm)
    #   smax    --  maximum slew rate (G/ cm*ms)
    #
    #   returns the maximum sdot (1st derivative of s) as a function of
    #   arclength s
    #   Also, returns curvature as a function of s and length of curve (L)
    #
    #  (c) Michael Lustig 2005
    #  last modified 2006
    """
    # rescale gyromagnetic factor
    _gamma_bar = gamma_bar * 1e-6 # Hz / T -> MHz / T
    _gamma_bar /= 10
    
    # preserve and flatten arclength:
    s = s.copy().flatten()
    
    # absolute value of 2nd derivative in curve space using cubic splines:
    dp_p = derivative(p_of_s, direction=1)
    dp_m = derivative(p_of_s, direction=-1)
    ds_p = derivative(s, direction=1)
    ds_m = derivative(s, direction=-1)
    
    cs_p = (pp(p_of_s + dp_p) - pp(p_of_s)) / ds_p[:, None]  # evaluate along arc length
    cs_m = (pp(p_of_s) - pp(p_of_s - dp_m)) / ds_m[:, None]  # evaluate along arc length
    cs = (cs_p - cs_m) / (ds_m / 2 + ds_p / 2)[:, None]
    
    # get magnitude
    k = np.linalg.norm(cs, axis=1)
    
    # fix edge numerical problems
    k[-1] = k[-2]
    k[0] = k[1]

    # calc I constraint curve (maximum gradient)
    sdot1 = _gamma_bar * gmax * np.ones(s.shape, s.dtype)

    # calc II constraint curve (curve curvature dependent)
    sdot2 = np.sqrt(_gamma_bar * smax / (k + np.finfo(float).eps))

    # calc total constraint
    sdot = np.minimum(sdot1, sdot2)

    return sdot, k

@nb.njit(cache=True, fastmath=True)  # pragma: no cover
def _forward_ode_solver(sta, s, ds, k, phi, smax):
    for n in range(1, s.shape[0]):
        kpos = 2 * (n - 1)
        dstds = _runge_kutta(ds, sta[n - 1], k[kpos:kpos + 3], smax)

        if dstds is None:
            sta[n] = phi[2 * n]
        else:
            tmpst = sta[n - 1] + dstds
            sta[n] = min(tmpst, phi[2 * n])
            
@nb.njit(cache=True, fastmath=True)  # pragma: no cover
def _backward_ode_solver(stb, s, ds, k, phi, smax):
    for n in range(s.shape[0] - 2, 0, -1):
        kpos = 2 * (n + 1)
        dstds = _runge_kutta(ds, stb[n + 1], k[kpos:(kpos - 3):-1], smax)

        if dstds is None:
            stb[n] = phi[2 * n]
        else:
            tmpst = stb[n + 1] + dstds
            stb[n] = min(tmpst, phi[2 * n])
            
@nb.njit(cache=True, fastmath=True)  # pragma: no cover
def _runge_kutta(ds: float, st: float, kvals: np.ndarray, smax: float = None):
    r"""
    Runge-Kutta 4 for curve constrained.
    
    Args:
        ds (float): spacing in arc length space
        st (float): output shape.
        kvals (array): 3 points of curve.
        smax (float): maximum slew
        gamma (float): gyromagnetic ratio
        
    Returns:
        float or None: step size dsdt or None
    """
    # rescale gyromagnetic factor
    _gamma_bar = gamma_bar * 1e-6 # Hz / T -> MHz / T
    _gamma_bar /= 10
        
    temp = (_gamma_bar ** 2 * smax ** 2 - abs(kvals[0]) ** 2 * st ** 4)
    if temp < 0.0:
        return None
    k1 = ds / st * math.sqrt(temp)

    temp = (_gamma_bar ** 2 * smax ** 2 - abs(kvals[1]) ** 2 * (st + ds * k1 / 2) ** 4)
    if temp < 0.0:
        return None
    k2 = ds / (st + ds * k1 / 2) * math.sqrt(temp)

    temp = (_gamma_bar ** 2 * smax ** 2 - abs(kvals[1]) ** 2 * (st + ds * k2 / 2) ** 4)
    if temp < 0.0:
        return None
    k3 = ds / (st + ds * k2 / 2) * math.sqrt(temp)

    temp = (_gamma_bar ** 2 * smax ** 2 - abs(kvals[2]) ** 2 * (st + ds * k3) ** 4)
    if temp < 0.0:
        return None
    k4 = ds / (st + ds * k3) * math.sqrt(temp)

    return k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

