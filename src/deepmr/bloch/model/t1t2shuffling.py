"""T1-T2 Shuffling simulator"""

__all__ = ["t1t2shuffling"]

import numpy as np

import dacite
from dacite import Config

from .. import blocks
from .. import ops
from . import base

def t1t2shuffling(flip, phases, ESP, TR, T1, T2, sliceprof=False, diff=None, device="cpu", **kwargs):
    """
    Simulate a T1T2Shuffling Spin Echo sequence. Only single-pool for now.

    Args:
        flip (float, array-like): refocusing angle in [deg] of shape (npulses,) or (npulses, nmodes).
        phases (float, array-like): refocusing angle phases in [deg] of shape (npulses,) or (npulses, nmodes).
        ESP (float): Echo spacing in [ms].
        TR (array-like): Variable TR list in [ms].
        T1 (float, array-like): Longitudinal relaxation time in [ms].
        T2 (float, array-like): Transverse relaxation time in [ms].
        sliceprof (optional, bool or array-like): excitation slice profile (i.e., flip angle scaling across slice).
            If False, pulse are non selective. If True, pulses are selective but ideal profile is assumed.
            If array, flip angle scaling along slice is simulated. Defaults to False.
        diff (optional, str, tuple[str]): Arguments to get the signal derivative with respect to.
            Defaults to None (no differentation).
        device (optional, str): Computational device. Defaults to "cpu".

    Kwargs (simulation):
        nstates (optional, int): Maximum number of EPG states to be retained during simulation.
            High numbers improve accuracy but decrease performance. Defaults to 10.
        max_chunk_size (optional, int): Maximum number of atoms to be simulated in parallel.
            High numbers increase speed and memory footprint. Defaults to natoms.
        nlocs (optional, int): Maximum number of spatial locations to be simulated (i.e., for slice profile effects).
            Defaults to 15 for slice selective and 1 for non-selective / ideal profile acquisitions.
        verbose (optional, bool): If true, prints execution time for signal (and gradient) calculations.

     Kwargs (System):
         B1 (optional, float, array-like): flip angle scaling factor (1.0 := nominal flip angle).
             Defaults to None (nominal flip angle).
         B0 (optional, float, array-like): Bulk off-resonance in [Hz]. Defaults to None.
         B1Tx2 (optional, Union[float, npt.NDArray[float], torch.FloatTensor]): flip angle scaling factor for secondary RF mode (1.0 := nominal flip angle). Defaults to None.
         B1phase (optional, Union[float, npt.NDArray[float], torch.FloatTensor]): B1 relative phase in [deg]. (0.0 := nominal rf phase). Defaults to None.

    """
    # constructor
    init_params = {"flip": flip, "phases": phases, "ESP": ESP, "TR": TR, "T1": T1, "T2": T2, "diff": diff, "device": device, **kwargs}

    # get verbosity
    if "verbose" in init_params:
        verbose = init_params["verbose"]
    else:
        verbose = False

    # get verbosity
    if "asnumpy" in init_params:
        asnumpy = init_params["asnumpy"]
    else:
        asnumpy = True
        
    # get selectivity:
    if sliceprof:
        selective = True
    else:
        selective = False
            
    # add moving pool if required
    if selective and "v" in init_params:
        init_params["moving"] = True

    # excitation pulse properties
    exc_props = {"slice_selective": selective}
    if "exc_flip" in kwargs:
        exc_props["flip"] = kwargs["exc_flip"]
    else:
        exc_props["flip"] = 90.0
    
    # refocusing pulse properties
    rf_props = {"slice_selective": selective}    
    if np.isscalar(sliceprof) is False:
        rf_props["slice_profile"] = kwargs["sliceprof"]
        
    # get nlocs
    if "nlocs" in init_params:
        nlocs = init_params["nlocs"]
    else:
        if selective:
            nlocs = 15
        else:
            nlocs = 1

    # interpolate slice profile:
    if "slice_profile" in rf_props:
        nlocs = min(nlocs, len(rf_props["slice_profile"]))
    else:
        nlocs = 1
        
    # assign nlocs
    init_params["nlocs"] = nlocs
    
    # put all properties together
    props = {"exc_props": exc_props, "rf_props": rf_props}

    # initialize simulator
    simulator = dacite.from_dict(T1T2Shuffling, init_params, config=Config(check_types=False))
            
    # run simulator
    if diff:
        # actual simulation
        sig, dsig = simulator(flip=flip, phases=phases, ESP=ESP, TR=TR, props=props)
        
        # flatten
        sig, dsig = sig.swapaxes(-1, -2), dsig.swapaxes(-1, -2) 
        sig = sig.reshape(-1, sig.shape[-1] * sig.shape[-2])
        dsig = dsig.reshape(-1, sig.shape[-1] * sig.shape[-2])

        # post processing
        if asnumpy:
            sig = sig.detach().cpu().numpy()
            dsig = dsig.detach().cpu().numpy()

        # prepare info
        info = {"trun": simulator.trun, "tgrad": simulator.tgrad}
        if verbose:
            return sig, dsig, info
        else:
            return sig, dsig
    else:
        # actual simulation
        sig = simulator(flip=flip, phases=phases, ESP=ESP, TR=TR, props=props)
        
        # flatten
        sig, dsig = sig.swapaxes(-1, -2)
        sig = sig.reshape(-1, sig.shape[-1] * sig.shape[-2])
        
        # post processing
        if asnumpy:
            sig = sig.cpu().numpy()

        # prepare info
        info = {"trun": simulator.trun}
        if verbose:
            return sig, info
        else:
            return sig

#%% utils
class T1T2Shuffling(base.BaseSimulator):
    """Class to simulate T1-T2 Shuffling."""

    @staticmethod
    def sequence(flip, phases, ESP, TR, props, T1, T2, B1, states, signal):
        
        # parsing pulses and grad parameters
        exc_props = props["exc_props"]
        rf_props = props["rf_props"]
        
        # get number of frames and echoes
        npulses = flip.shape[0]
        
        # define preparation
        Exc = blocks.ExcPulse(states, B1, exc_props)

        # prepare RF pulse
        RF = blocks.ExcPulse(states, B1, rf_props)

        # prepare free precession period
        Xpre, Xpost = blocks.FSEStep(states, ESP, T1, T2)
        
        # magnetization prep
        states = Exc(states, exc_props["flip"])
        
        # get recovery times
        rectime = TR - (npulses + 1) * ESP # T + 1 to account for fast recovery

        # actual sequence loop
        for n in range(npulses):
            
            # relax, recover and shift for half echo spacing
            states = Xpre(states)
                        
            # apply refocusing
            states = RF(states, flip[n], phases[n])

            # relax, recover and spoil for half echo spacing
            states = Xpost(states)
            
            # observe magnetization
            signal[n] = ops.observe(states, RF.phi)

        return ops.t1sat(signal  * 1j, rectime, T1)


