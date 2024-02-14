"""Base analytical simulation Class."""

__all__ = ["AnalyticalSimulator"]

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

eps = torch.finfo(torch.float32).eps
spin_properties = (
    "T1",
    "T2",
    "B0",
    "B1",
)
allowed = ("T1", "T2", "B1", "df", "props")


@dataclass
class AnalyticalSimulator:
    """
    Base class for all analytical Bloch simulators.

    Users can define a new simulator by subclassing this
    and overloading "sequence" method. Base class already handle spin parameters (e.g., T1, T2, ...)
    as well as simulation properties (e.g., computational device, max number of batches...) so the user
    has only to care about specific sequence arguments (e.g., flip angle, TR, ... for GRE or flip angle, ETL, for FSE).

    In order to work properly, "sequence" method must be a 'staticmethod', and the arguments must follow this order:

        1. sequence parameters (flip angle, TE, TR, nrepetitions, ...)
        2. spin parameters (T1, T2, B1, ...)
        3. buffer for EPG states and output signal (mandatory): signal

    Examples
    --------
    >>> import torch
    >>> from deepmr import bloch

    >>> class SPGR(bloch.AnalyticalSimulator):
    >>>
    >>>     @staticmethod
    >>>     def signal(flip, TR, TE, T1, T2, signal):
    >>>         
    >>>         # factors
    >>>         E1 = torch.exp(-TR/T1)
    >>>         E2 = torch.exp(-TE/T2) # here, T2 actually represents T2*
    >>>         sina = torch.sin(flip)
    >>>         cosa = torch.cos(flip)
    >>>
    >>>         # actual signal
    >>>         signal = sina * (1 - E1) / (1 - cosa * E1) * E2
    >>>         
    >>>         return signal

    The resulting class can be used to perform simulation by instantiating an object (spin properties as input)
    and using the '__call__' method (sequence properties as input).

    >>> spgt = SPGR(device=device, T1=T1, T2=T2)
    >>> signal = spgr(flip=flip, TR=TR, TE=TE)

    For convenience, simulator instantiation and actual simulation can (and should) be wrapped in a wrapper function.

    >>> def simulate_spgr(flip, TR, TE, T1, T2, device="cpu"):
    >>>     mysim = SPGR(device=device, T1=T1, T2=T2)
    >>>     return spgr(flip=flip, TR=TR, TE=TE)

    The class also enable automatic forward differentiation wrt to input spin parameters via "diff" argument.

    >>> import numpy as np
    >>>
    >>> def simulate_spgr(flip, TR, T1, T2, diff=None, device="cpu"):
    >>>     ssfp = SPGR(device=device, T1=T1, T2=T2, diff=diff)
    >>>     return spgr(flip=flip, TR=TR, TE=TE)
    >>>
    >>> # this will return signal only
    >>> signal = simulate_spgr(flip=10.0*np.ones(1000, dtype=np.float32), TR=4.5, TE=1.0, T1=500.0, T2=20.0)
    >>>
    >>> # this will also return derivatives
    >>> signal, dsignal = simulate_spgr(flip=10.0*np.ones(1000, dtype=np.float32), TR=8.5, T1=500.0, T2=20.0, diff=("T1", "T2"))
    >>>
    >>> # dsignal[0] = dsignal / dT1 (derivative of signal wrt T1)
    >>> # dsignal[1] = dsignal / dT2* (derivative of signal wrt T2*)

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
    max_chunk_size : int, optional
        Maximum number of atoms to be simulated in parallel. 
        High numbers increase speed and memory footprint. 
        Defaults to ``natoms``.
    nlocs : int, optional
        Number of spatial locations to be simulated (i.e., for slice profile effects). 
        Defaults to ``1``.

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
    max_chunk_size: int = None

    # other main pool properties
    T2star: Union[float, npt.NDArray[float], torch.FloatTensor] = None  # ms

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

        # B0
        if self.B0 is None:
            self.B0 = torch.zeros(
                self.T1.shape, dtype=torch.float32, device=self.device
            )

        # frequency offset
        self.df = 1j * 2 * math.pi * self.B0
        self.df = torch.stack((self.df.real, self.df.imag), axis=-1)

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
        buffer = torch.zeros((self.batch_size, self.seqlength), dtype=torch.complex64, device=self.device)
        return {"real": buffer.real, "imag": buffer.imag}
    
    def get_sim_inputs(self, modelsig):  # noqa
        output = {
            "T1": self.T1,
            "T2": self.T2,
            "df": self.df,
            "B1": self.B1,
        }

        # clean up
        output = {k: v for k, v in output.items() if k in modelsig}

        return output

    def reformat(self, input):  # noqa
        # handle tuples
        if isinstance(input, (list, tuple)):
            output = [item[..., 0, :] + 1j * item[..., -1, :] for item in input]
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

        # get sequence and tissue parameters
        seqparams = list(seq_kwargs.keys())
        tissueparams = self.get_sim_inputs(modelparams)

        # check validity of sequence properties
        assert set(seqparams).issubset(
            set(modelparams)
        ), f"Error! Function call ({seqparams}) does not match model signature ({modelparams})."

        # check validity of tissue properties
        candidate = set(modelparams).difference(set(seqparams + ["signal"]))
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
        inputs["signal"] = buffer
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

        # run function
        output = func(*args)

        # replace
        return complex2real(output)

    return wrapper

def real2complex(input):
    return input["real"] + 1j * input["imag"]

def complex2real(input):
    return torch.stack((input.real, input.imag), dim=-1)

def _sort_signature(input, reference):
    out = {k: input[k] for k in reference if k in input}
    return list(out.values()), list(out.keys())
