"""Base EPG simulation Class."""

__all__ = ["EPGSimulator"]

import math
import gc
import inspect
import time

from dataclasses import dataclass, fields
from functools import partial, wraps
from typing import Tuple, Union

import numpy.typing as npt
import torch
from torch.func import jacfwd, vmap

from .. import ops

eps = torch.finfo(torch.float32).eps
spin_properties = (
    "T1",
    "T2",
    "T2star",
    "chemshift",
    "D",
    "v",
    "T1bm",
    "T2bm",
    "chemshift_bm",
    "kbm",
    "weight_bm",
    "kmt",
    "weight_mt",
    "B0",
    "B1",
    "B1Tx2",
    "B1phase",
)
allowed = ("T1", "T2", "k", "weight", "chemshift", "D", "v", "B1", "df", "props")


@dataclass
class EPGSimulator:
    """
    Base class for all EPG-based Bloch simulators.

    Users can define a new simulator by subclassing this
    and overloading "sequence" method. Base class already handle spin parameters (e.g., T1, T2, ...)
    as well as simulation properties (e.g., computational device, max number of batches...) so the user
    has only to care about specific sequence arguments (e.g., flip angle, TR, ... for GRE or flip angle, ETL, for FSE).

    In order to work properly, "sequence" method must be a 'staticmethod', and the arguments must follow this order:

        1. sequence parameters (flip angle, TE, TR, nrepetitions, ...)
        2. spin parameters (T1, T2, B1, ...)
        3. buffer for EPG states and output signal (mandatory): states, signal

    Examples
    --------
    >>> from deepmr import bloch
    >>> from deepmr.bloch import ops

    >>> class SSFP(bloch.EPGSimulator):
    >>>
    >>>     @staticmethod
    >>>     def signal(flip, TR, T1, T2, states, signal):
    >>>
    >>>         # get device and sequence length
    >>>         device = flip.device
    >>>         npulses = flip.shape[-1]
    >>>
    >>>         # define operators
    >>>         T = ops.RFPulse(device, alpha=flip) # RF pulse
    >>>         E = ops.Relaxation(device, TR, T1, T2) # relaxation until TR
    >>>         S = ops.Shift() # gradient spoil
    >>>
    >>>         # apply sequence
    >>>         for n in range(npulses):
    >>>             states = T(states)
    >>>             signal[n] = ops.observe(states)
    >>>             states = E(states)
    >>>             states = S(states)
    >>>
    >>>             # return output
    >>>             return signal

    The resulting class can be used to perform simulation by instantiating an object (spin properties as input)
    and using the '__call__' method (sequence properties as input).

    >>> ssfp = SSFP(device=device, T1=T1, T2=T2)
    >>> signal = ssfp(flip=flip, TR=TR)

    For convenience, simulator instantiation and actual simulation can (and should) be wrapped in a wrapper function.

    >>> def simulate_ssfp(flip, TR, T1, T2, device="cpu"):
    >>>     mysim = SSFP(device=device, T1=T1, T2=T2)
    >>>     return ssfp(flip=flip, TR=TR)

    The class also enable automatic forward differentiation wrt to input spin parameters via "diff" argument.

    >>> import numpy as np
    >>>
    >>> def simulate_ssfp(flip, TR, T1, T2, diff=None, device="cpu"):
    >>>     ssfp = SSFP(device=device, T1=T1, T2=T2, diff=diff)
    >>>     return ssfp(flip=flip, TR=TR)
    >>>
    >>> # this will return signal only (evolution towards steady state of unbalanced SSFP sequence)
    >>> signal = simulate_ssfp(flip=10.0*np.ones(1000, dtype=np.float32), TR=4.5, T1=500.0, T2=50.0)
    >>>
    >>> # this will also return derivatives
    >>> signal, dsignal = simulate_ssfp(flip=10.0*np.ones(1000, dtype=np.float32), TR=8.5, T1=500.0, T2=50.0, diff=("T1", "T2"))
    >>>
    >>> # dsignal[0] = dsignal / dT1 (derivative of signal wrt T1)
    >>> # dsignal[1] = dsignal / dT2 (derivative of signal wrt T2)

    This is useful e.g. for nonlinear fitting and for calculating objective functions (CRLB) for sequence optimization.

    Parameters
    ----------
    T1 : float | np.ndarray | torch.Tensor
        Longitudinal relaxation time for main pool in ``[ms]``.
    T2 : float | np.ndarray | torch.Tensor
        Transverse relaxation time for main pool in ``[ms]``.
    diff : str | tuple[str], optional
        String or tuple of strings, saying which arguments 
        to get the signal derivative with respect to. 
        Defaults to ``None`` (no differentation).
    device : str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
    B1 : float | np.ndarray | torch.Tensor , optional
        Flip angle scaling factor (``1.0 := nominal flip angle``). 
        Defaults to ``None``.
    B0 : float | np.ndarray | torch.Tensor , optional 
        Bulk off-resonance in [Hz]. Defaults to ``None``

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
    nlocs : int, optional
        Number of spatial locations to be simulated (i.e., for slice profile effects). 
        Defaults to ``1``.
    T2star : float | np.ndarray | torch.Tensor
        Effective relaxation time for main pool in ``[ms]``. 
        Defaults to ``None``.
    D : float | np.ndarray | torch.Tensor
        Apparent diffusion coefficient in ``[um**2 / ms]``. 
        Defaults to ``None``.
    v : float | np.ndarray | torch.Tensor
        Spin velocity ``[cm / s]``. Defaults to ``None``.
    moving : bool, optional 
        If True, simulate an in-flowing spin pool. 
        Defaults to ``False``.
    chemshift  : float | np.ndarray | torch.Tensor 
        Chemical shift for main pool in ``[Hz]``. 
        Defaults to ``None``.
    T1bm : float | np.ndarray | torch.Tensor
        Longitudinal relaxation time for secondary pool in ``[ms]``. 
        Defaults to ``None``.
    T2bm : float | np.ndarray | torch.Tensor
        Transverse relaxation time for main secondary in ``[ms]``. 
        Defaults to ``None``.
    kbm : float | np.ndarray | torch.Tensor 
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
    B1Tx2 : float | np.ndarray | torch.Tensor 
        Flip angle scaling factor for secondary RF mode (``1.0 := nominal flip angle``). 
        Defaults to ``None``.
    B1phase : float | np.ndarray | torch.Tensor
        B1 relative phase in ``[deg]``. (``0.0 := nominal rf phase``). 
        Defaults to ``None``.

    """

    # main properties
    T1: Union[float, npt.NDArray[float], torch.FloatTensor]  # ms
    T2: Union[float, npt.NDArray[float], torch.FloatTensor]  # ms
    diff: Union[str, Tuple[str]] = None
    device: str = "cpu"
    B1: Union[float, npt.NDArray[float], torch.FloatTensor] = None
    B0: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # Hz

    # other simulation parameters
    nlocs: int = 1
    nstates: int = 10
    max_chunk_size: int = None

    # other main pool properties
    T2star: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # ms
    D: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # um**2 / ms
    v: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # cm / s
    moving: bool = False
    chemshift: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # Hz

    # bloch-mcconnell parameters
    T1bm: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # ms
    T2bm: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # ms
    kbm: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # Hz
    weight_bm: Union[float, npt.NDArray[float], torch.FloatTensor] = None
    chemshift_bm: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # Hz

    # bloch-mt parameters
    kmt: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # Hz
    weight_mt: Union[float, npt.NDArray[float], torch.FloatTensor] = None

    # fields
    B1Tx2: Union[float, npt.NDArray[float], torch.FloatTensor] = None
    B1phase: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # deg

    @staticmethod
    def sequence():  # noqa
        """Base method to be overridden to define new signal simulators."""
        pass

    def __post_init__(self):  # noqa
        # gather properties
        props = {}
        for f in fields(self):  # iterate over class fields
            fname = f.name
            if fname in spin_properties:
                fvalue = getattr(self, fname)  # get current value
                props[fname] = fvalue

        # cast to tensors
        props = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in props.items()
            if v is not None
        }

        # expand
        props = {
            k: torch.atleast_1d(v.squeeze())[..., None] + eps for k, v in props.items()
        }

        # broadcast
        bprops = torch.broadcast_tensors(*props.values())
        props = dict(zip(props.keys(), bprops))

        # replace properties
        for fname in props.keys():  # iterate over class fields
            setattr(self, fname, props[fname])

        # diff
        if self.diff is None:
            self.diff = []
        elif isinstance(self.diff, str):
            self.diff = [self.diff]

        # bloch-mcconnell
        self.model = None
        if self.T1bm is not None:
            self.model = "bm"
            assert self.T2bm is not None, "T2 for secondary free pool not provided!"
            assert (
                self.weight_bm is not None
            ), "Weight for secondary free pool not provided!"
            assert (
                self.kbm is not None
            ), "Exchange rate for secondary free pool not provided!"
            self.T1 = torch.cat((self.T1, self.T1bm), axis=-1) + eps
        if self.T2bm is not None:
            assert self.T1bm is not None, "T1 for secondary free pool not provided!"
            self.T2 = torch.cat((self.T2, self.T2bm), axis=-1) + eps

        # MT
        if self.kmt is not None:
            if self.model is not None:
                self.model = "bm-mt"
            else:
                self.model = "mt"
            assert self.weight_mt is not None, "Weight for bound pool not provided!"
        if self.weight_mt is not None:
            assert self.kmt is not None, "Exchange rate for bound pool not provided!"

        # build fraction
        self.weight = None
        if self.model == "bm":
            self.weight = torch.cat((1 - self.weight_bm, self.weight_bm), axis=-1)
        if self.model == "mt":
            self.weight = torch.cat((1 - self.weight_mt, self.weight_mt), axis=-1)
        if self.model == "bm-mt":
            weight_free = torch.cat((1 - self.weight_bm, self.weight_bm), axis=-1)
            self.weight = torch.cat(
                ((1 - self.weight_mt) * weight_free, self.weight_mt), axis=-1
            )

        # build exchange matrix
        if self.model == "bm":
            self.k = self.kbm
        elif self.model == "mt":
            self.k = self.kmt
        elif self.model == "bm-mt":
            self.k = torch.cat((self.kbm, self.kmt), axis=-1)
        else:
            self.k = None

        # finalize exchange
        if self.k is not None:
            # single pool voxels do not exchange
            idx = torch.isclose(self.weight, torch.tensor(1.0)).sum(axis=-1) == 1
            self.k[idx, :] = 0.0

        # chemical shift
        if self.model is not None and "bm" in self.model:
            if self.chemshift is not None and self.chemshift_bm is None:
                self.chemshift = torch.cat(
                    (self.chemshift, 0 * self.chemshift), axis=-1
                )
            elif self.chemshift is None and self.chemshift_bm is not None:
                self.chemshift = torch.cat(
                    (0 * self.chemshift_bm, self.chemshift_bm), axis=-1
                )
            elif self.chemshift is not None and self.chemshift_bm is not None:
                self.chemshift = torch.cat((self.chemshift, self.chemshift_bm), axis=-1)
            else:
                self.chemshift = torch.zeros(
                    self.T1.shape, dtype=torch.float32, device=self.device
                )
        else:
            if self.chemshift is None:
                self.chemshift = torch.zeros(
                    self.T1.shape, dtype=torch.float32, device=self.device
                )

        # B0
        if self.B0 is None:
            self.B0 = torch.zeros(
                self.T1.shape, dtype=torch.float32, device=self.device
            )

        # total (complex) field variation
        if self.T2star is not None and self.model is not None:
            R2prime = 1 / self.T2star - 1 / self.T2[..., -1]
            T2prime = 1 / R2prime
            T2prime = torch.nan_to_num(T2prime, posinf=0.0, neginf=0.0) + eps
            self.df = R2prime + 1j * 2 * math.pi * self.B0
        elif self.T2star is None and self.model is None:
            self.df = 1j * 2 * math.pi * (self.B0 + self.chemshift)
        elif self.T2star is not None and self.model is None:
            R2star = 1 / self.T2star
            R2star = torch.nan_to_num(R2star, posinf=0.0, neginf=0.0) + eps
            self.df = R2star + 1j * 2 * math.pi * (self.B0 + self.chemshift)
        else:
            self.df = 1j * 2 * math.pi * self.B0
        self.df = torch.stack((self.df.real, self.df.imag), axis=-1)

        # B1
        if self.B1Tx2 is not None:
            assert self.B1 is not None, "B1 not provided!"
            if self.B1phase is None:
                self.B1phase = 0.0 * self.B1
            else:
                self.B1phase = torch.deg2rad(self.B1phase)
            self.B1 = torch.cat(
                (self.B1, self.B1Tx2 * torch.exp(1j * self.B1phase)), axis=-1
            )

        # get sizes
        self.batch_size, self.npools = self.T1.shape
        if self.max_chunk_size is None:
            self.max_chunk_size = self.batch_size

        # set differentiation
        for f in fields(self):  # iterate over class fields
            fname = f.name
            if fname in self.diff:
                fvalue = getattr(self, fname)  # get current value
                if fvalue.requires_grad is False:
                    fvalue.requires_grad = True
                    setattr(self, fname, fvalue)

    def initialize_buffer(self):  # noqa
        return ops.EPGstates(
            self.device,
            self.batch_size,
            self.nstates,
            self.nlocs,
            self.seqlength,
            self.npools,
            self.weight,
            self.model,
            self.moving,
        )

    def get_sim_inputs(self, modelsig):  # noqa
        output = {
            "T1": self.T1,
            "T2": self.T2,
            "k": self.k,
            "weight": self.weight,
            "chemshift": self.chemshift,
            "df": self.df,
            "D": self.D,
            "v": self.v,
            "B1": self.B1,
        }

        # clean up
        output = {k: v for k, v in output.items() if k in modelsig}

        return output

    def reformat(self, input):  # noqa
        # handle tuples
        if isinstance(input, (list, tuple)):
            output = [item[..., 0, :] + 1j * item[..., -1, :] for item in input]
            # output = [torch.diagonal(item, dim1=-2, dim2=-1) if len(item.shape) == 4 else item for item in output]
            output = [
                item.reshape(*item.shape[:2], -1) if len(item.shape) == 4 else item
                for item in output
            ]

            # stack
            if len(output) == 1:
                output = output[0]
            else:
                output = torch.concatenate(output, dim=-1)
                output = output.permute(2, 0, 1)
        else:
            output = input[..., 0] + 1j * input[..., -1]

        return output

    def __call__(self, **seq_kwargs):  # noqa
        # clean memory
        gc.collect()

        # inspect signature
        modelparams = inspect_signature(self.sequence)
        assert (
            "signal" in modelparams
        ), "Error! Please design the model to accept a 'signal' argument."
        assert (
            "states" in modelparams
        ), "Error! Please design the model to accept a 'states' argument."

        # get sequence and tissue parameters
        seqparams = list(seq_kwargs.keys())
        tissueparams = self.get_sim_inputs(modelparams)

        # check validity of sequence properties
        assert set(seqparams).issubset(
            set(modelparams)
        ), f"Error! Function call ({seqparams}) does not match model signature ({modelparams})."

        # check validity of tissue properties
        candidate = set(modelparams).difference(set(seqparams + ["signal", "states"]))
        rem = [c for c in candidate if c not in allowed]
        assert (
            len(rem) == 0
        ), f"Error! Model signature contains unrecognized arguments = {rem} - valid arguments are {allowed}"

        # convert to 1D tensors when possible
        for k, v in seq_kwargs.items():
            if isinstance(v, dict) is False and v is not None:
                seq_kwargs[k] = torch.atleast_1d(
                    torch.as_tensor(v, dtype=torch.float32, device=self.device)
                )

        # get shapes
        seqlength = [
            arg.shape[0] for arg in seq_kwargs.values() if isinstance(arg, torch.Tensor)
        ]
        seqlength = list(dict.fromkeys(seqlength))
        self.seqlength = [el for el in seqlength if el != 1]

        # prepare inputs
        buffer = self.initialize_buffer()
        inputs = tissueparams
        inputs["signal"] = buffer["signal"]
        inputs["states"] = buffer["states"]
        inputs, inputnames = _sort_signature(inputs, inspect_signature(self.sequence))
        seqparams, _ = _sort_signature(seq_kwargs, inspect_signature(self.sequence))

        # workaround for "None" inputs
        in_dims = [0] * len(inputs)
        for n in range(len(in_dims)):
            if inputs[n] is None:
                in_dims[n] = None
        in_dims = tuple(in_dims)

        # actual body
        modelfunc = jacadapt(self.sequence)
        modelfunc = partial(modelfunc, *seqparams)
        run = vmap(modelfunc, in_dims=in_dims, chunk_size=self.max_chunk_size)

        # prepare function
        if self.diff:
            argnums = [inputnames.index(arg) for arg in self.diff]
            argnums = tuple(argnums)

            # prepare derivative
            drun = vmap(
                jacfwd(modelfunc, argnums=argnums),
                in_dims=in_dims,
                chunk_size=self.max_chunk_size,
            )

            # run
            t0 = time.time()
            sig = run(*inputs)
            gc.collect()
            t1 = time.time()
            self.trun = t1 - t0
            t0 = time.time()
            dsig = drun(*inputs)
            gc.collect()
            t1 = time.time()
            self.tgrad = t1 - t0

            # reformat
            sig = self.reformat(sig)
            dsig = self.reformat(dsig)

            return sig.squeeze(), dsig.squeeze()
        else:
            with torch.no_grad():
                t0 = time.time()
                sig = run(*inputs)
                gc.collect()
                t1 = time.time()
                self.trun = t1 - t0

            # reformat
            sig = self.reformat(sig)

            return sig.squeeze()


# %% local utils
def inspect_signature(input):
    return list(inspect.signature(input).parameters)

def jacadapt(func):
    @wraps(func)
    def wrapper(*args):
        # replace
        args = list(args)
        args[-1] = real2complex(args[-1], "signal")
        args[-2] = real2complex(args[-2], "states")

        # run function
        output = func(*args)

        # replace
        return complex2real(output)

    return wrapper

def real2complex(input, what):
    if what == "signal":
        return input["real"] + 1j * input["imag"]
    elif what == "states":
        F = input["F"]["real"] + 1j * input["F"]["imag"]
        Z = input["Z"]["real"] + 1j * input["Z"]["imag"]
        out = {"F": F, "Z": Z}

        if "moving" in input is not None:
            Fmoving = input["moving"]["F"]["real"] + 1j * input["moving"]["F"]["imag"]
            Zmoving = input["moving"]["Z"]["real"] + 1j * input["moving"]["Z"]["imag"]
            out["moving"] = {}
            out["moving"]["F"] = Fmoving
            out["moving"]["Z"] = Zmoving

        if "Zbound" in input is not None:
            Zbound = input["Zbound"]["real"] + 1j * input["Zbound"]["imag"]
            out["Zbound"] = Zbound

        return out

def complex2real(input):
    return torch.stack((input.real, input.imag), dim=-1)

def _sort_signature(input, reference):
    out = {k: input[k] for k in reference if k in input}
    return list(out.values()), list(out.keys())
