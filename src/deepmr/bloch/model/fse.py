"""Fast Spin Echo simulator"""

__all__ = ["fse"]

import warnings
import numpy as np

import dacite
from dacite import Config

from .. import blocks
from .. import ops
from . import epg


def fse(flip, phases, ESP, T1, T2, sliceprof=False, diff=None, device="cpu", **kwargs):
    """
    Simulate a Fast Spin Echo sequence.

    Parameters
    ----------
    flip : float | np.ndarray | torch.Tensor
        Flip angle in ``[deg]`` of shape ``(npulses,)`` or ``(npulses, nmodes)``.
    phases : float | np.ndarray | torch.Tensor
        Refocusing angle phases in ``[deg]`` of shape ``(npulses,)`` or ``(npulses, nmodes)``.
    ESP : float
        Echo spacing in [ms].      
    T1 : float | np.ndarray | torch.Tensor
        Longitudinal relaxation time for main pool in ``[ms]``.
    T2 : float | np.ndarray | torch.Tensor
        Transverse relaxation time for main pool in ``[ms]``.
    sliceprof : float | np.ndarray | torch.Tensor
        Excitation slice profile (i.e., flip angle scaling across slice).
        If ``False``, pulse are non selective. If ``True``, pulses are selective but ideal profile is assumed.
        If array, flip angle scaling along slice is simulated. Defaults to ``False``.  
    spoil_inc : float, optional 
        RF spoiling increment in ``[deg]``. Defaults to ``117°``.      
    diff : str | tuple[str], optional
        String or tuple of strings, saying which arguments 
        to get the signal derivative with respect to. 
        Defaults to ``None`` (no differentation).
    device : str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
        Defaults to ``cpu``.

    Other Parameters
    ----------------
    nstates : int, optional 
        Maximum number of EPG states to be retained during simulation. 
        High numbers improve accuracy but decrease performance. 
        Defaults to ``10``.
    max_chunk_size : int, optional
        Maximum number of atoms to be simulated in parallel. 
        High numbers increase speed and memory footprint. 
        Defaults to ``natoms``.
    verbose : bool, optional
        If ``True``, prints execution time for signal (and gradient) calculations.
        Defaults to ``False``.
    B1sqrdTau : float, optional 
        Refocusing pulse energies in ``[uT**2 * ms]`` when ``flip = 1.0 [deg]``.
    exc_flip : float 
        Excitation flip angle. Defaults to ``90 [deg]``.
    exc_B1sqrdTau: float 
        Excitation pulse energy in ``[uT**2 * ms]``.
    grad_tau : float, optional
        Gradient lobe duration in ``[ms]``.
    grad_amplitude : float, optional
        Gradient amplitude along unbalanced direction in ``[mT / m]``.
        If total_dephasing is not provided, this is used to compute diffusion and flow effects.
    grad_dephasing : float, optional 
        Total gradient-induced dephasing across a voxel (in grad direction).
        If gradient_amplitude is not provided, this is used to compute diffusion and flow effects.
    voxelsize : str | list | tuple | np.ndarray | torch.Tensor, optional  
        Voxel size (``dx``, ``dy``, ``dz``) in ``[mm]``. 
        If scalar, assume isotropic voxel. Defaults to ``None``.
    grad_orient : str | list | tuple | np.ndarray | torch.Tensor, optional 
        Gradient orientation (``"x"``, ``"y"``, ``"z"`` or ``versor``). Defaults to ``"z"``.
    slice_orient : str | list | tuple | np.ndarray | torch.Tensor, optional 
        Slice orientation (``"x"``, ``"y"``, ``"z"`` or ``versor``).
        Ignored if pulses are non-selective. Defaults to ``"z"``.
    B1 : float | np.ndarray | torch.Tensor , optional
        Flip angle scaling factor (``1.0 := nominal flip angle``). 
        Defaults to ``None``.
    B0 : float | np.ndarray | torch.Tensor , optional 
        Bulk off-resonance in [Hz]. Defaults to ``None``
    B1Tx2 : float | np.ndarray | torch.Tensor 
        Flip angle scaling factor for secondary RF mode (``1.0 := nominal flip angle``). 
        Defaults to ``None``.
    B1phase : float | np.ndarray | torch.Tensor
        B1 relative phase in ``[deg]``. (``0.0 := nominal rf phase``). 
        Defaults to ``None``.      
    T2star : float | np.ndarray | torch.Tensor
        Effective relaxation time for main pool in ``[ms]``. 
        Defaults to ``None``.
    D : float | np.ndarray | torch.Tensor
        Apparent diffusion coefficient in ``[um**2 / ms]``. 
        Defaults to ``None``.
    v : float | np.ndarray | torch.Tensor
        Spin velocity ``[cm / s]``. Defaults to ``None``.  
    chemshift : float | np.ndarray | torch.Tensor 
        Chemical shift for main pool in ``[Hz]``. 
        Defaults to ``None``.
    T1bm : float | np.ndarray | torch.Tensor
        Longitudinal relaxation time for secondary pool in ``[ms]``. 
        Defaults to ``None``.
    T2bm : float | np.ndarray | torch.Tensor
        Transverse relaxation time for main secondary in ``[ms]``. 
        Defaults to ``None``.
    kbm  : float | np.ndarray | torch.Tensor 
        Nondirectional exchange between main and secondary pool in ``[Hz]``. 
        Defaults to ``None``.
    weight_bm  : float | np.ndarray | torch.Tensor
        Relative secondary pool fraction. 
        Defaults to ``None``.
    chemshift_bm : float | np.ndarray | torch.Tensor
        Chemical shift for secondary pool in ``[Hz]``. 
        Defaults to ``None``.
    kmt : float | np.ndarray | torch.Tensor 
        Nondirectional exchange between free and bound pool in ``[Hz]``.
        If secondary pool is defined, exchange is between secondary and bound pools 
        (i.e., myelin water and macromolecular), otherwise exchange 
        is between main and bound pools. 
        Defaults to ``None``.
    weight_mt : float | np.ndarray | torch.Tensor
        Relative bound pool fraction. 
        Defaults to ``None``.

    """
    # constructor
    init_params = {
        "flip": flip,
        "phases": phases,
        "ESP": ESP,
        "T1": T1,
        "T2": T2,
        "diff": diff,
        "device": device,
        **kwargs,
    }

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
    if "exc_B1sqrdTau" in kwargs:
        exc_props["b1rms"] = kwargs["exc_B1sqrdTau"] ** 0.5
        exc_props["duration"] = 1.0

    # refocusing pulse properties
    rf_props = {"slice_selective": selective}
    if "B1sqrdTau" in kwargs:
        rf_props["b1rms"] = kwargs["B1sqrdTau"] ** 0.5
        rf_props["duration"] = 1.0

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

    # unbalanced gradient properties
    grad_props = {}
    if "grad_tau" in kwargs:
        grad_props["duration"] = kwargs["grad_tau"]
    if "grad_dephasing" in kwargs:
        grad_props["total_dephasing"] = kwargs["grad_dephasing"]
    if "voxelsize" in kwargs:
        grad_props["voxelsize"] = kwargs["voxelsize"]
    if "grad_amplitude" in kwargs:
        grad_props["grad_amplitude"] = kwargs["grad_amplitude"]
    if "grad_orient" in kwargs:
        grad_props["grad_direction"] = kwargs["grad_orient"]
    if "slice_orient" in kwargs:
        grad_props["slice_direction"] = kwargs["slice_orient"]

    # check for possible inconsistencies:
    if "total_dephasing" in rf_props and "grad_amplitude" in rf_props:
        warnings.warn(
            "Both total_dephasing and grad_amplitude are provided - using the first"
        )

    # put all properties together
    props = {"exc_props": exc_props, "rf_props": rf_props, "grad_props": grad_props}

    # initialize simulator
    simulator = dacite.from_dict(FSE, init_params, config=Config(check_types=False))

    # run simulator
    if diff:
        # actual simulation
        sig, dsig = simulator(flip=flip, phases=phases, ESP=ESP, props=props)

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
        sig = simulator(flip=flip, phases=phases, ESP=ESP, props=props)

        # post processing
        if asnumpy:
            sig = sig.cpu().numpy()

        # prepare info
        info = {"trun": simulator.trun}
        if verbose:
            return sig, info["trun"]
        else:
            return sig


# %% utils
spin_defaults = {"D": None, "v": None}


class FSE(epg.EPGSimulator):
    """Class to simulate Fast Spin Echo."""

    @staticmethod
    def sequence(
        flip,
        phases,
        ESP,
        props,
        T1,
        T2,
        B1,
        df,
        weight,
        k,
        chemshift,
        D,
        v,
        states,
        signal,
    ):
        # parsing pulses and grad parameters
        exc_props = props["exc_props"]
        rf_props = props["rf_props"]
        grad_props = props["grad_props"]

        # get number of frames and echoes
        npulses = flip.shape[0]

        # define preparation
        Exc = blocks.ExcPulse(states, B1, exc_props)

        # prepare RF pulse
        RF = blocks.ExcPulse(states, B1, rf_props)

        # prepare free precession period
        Xpre, Xpost = blocks.FSEStep(
            states, ESP, T1, T2, weight, k, chemshift, D, v, grad_props
        )

        # magnetization prep
        states = Exc(states, exc_props["flip"])

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

        return signal * 1j
