# Bloch Simulation

```{eval-rst}
.. automodule:: deepmr.bloch
```

## Numerical Models
```{eval-rst}
.. currentmodule:: deepmr.bloch
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.bloch.mprage
	deepmr.bloch.memprage
	deepmr.bloch.bssfpmrf
	deepmr.bloch.ssfpmrf
	deepmr.bloch.fse
	deepmr.bloch.t1t2shuffling
```

## Custom signal models

DeepMR also contains helper classes to define custom signal models. The main class for analytical and numerical models are `deepmr.bloch.AnalyticalSimulator`
and `deepmr.bloch.EPGSimulator`, respectively.

Users can define a custom signal model by subclassing it and overloading ``sequence`` method. Base class already handle spin parameters (e.g., ``T1``, ``T2``, ...)
as well as simulation properties (e.g., computational ``device``, maximum ``number of batches``...) so the user has only to care about specific sequence arguments (e.g., ``flip angle``, ``TR``, ... for GRE or ``flip angle``, ``ETL``, for FSE). In order to work properly, ``sequence`` method must be a ``staticmethod`` and the arguments must follow this order:

1. sequence parameters (``flip angle``, ``TE``, ``TR``, ``nrepetitions``, ...)
2. spin parameters (``T1``, ``T2``, ``B1``, ...)
3. (mandatory) buffer for output signal (analytical) or EPG states and output signal (numerical):  ``signal`` / `` states``, ``signal``

```python
from deepmr import bloch
from deepmr.bloch import ops

class SSFP(bloch.EPGSimulator):

    @staticmethod
    def signal(flip, TR, T1, T2, states, signal):

        # get device and sequence length
        device = flip.device
        npulses = flip.shape[-1]

        # define operators
        T = ops.RFPulse(device, alpha=flip) # RF pulse
        E = ops.Relaxation(device, TR, T1, T2) # relaxation until TR
        S = ops.Shift() # gradient spoil

        # apply sequence
        for n in range(npulses):
            states = T(states)
            signal[n] = ops.observe(states)
            states = E(states)
            states = S(states)

		# return output
		return signal
```

The resulting class can be used to perform simulation by instantiating an object (spin properties as input)and using the ``__call__`` method (sequence properties as input):

```python
ssfp = SSFP(device=device, T1=T1, T2=T2) # build simulator
signal = ssfp(flip=flip, TR=TR) # run simulation
```

For convenience, simulator instantiation and actual simulation can (and should) be wrapped in a wrapper function:

```python
def simulate_ssfp(flip, TR, T1, T2, device="cpu"):
    mysim = SSFP(device=device, T1=T1, T2=T2)
    return ssfp(flip=flip, TR=TR)
```

The class also enable automatic forward differentiation wrt to input spin parameters via ``diff`` argument:

```python
import numpy as np

def simulate_ssfp(flip, TR, T1, T2, diff=None, device="cpu"):
	ssfp = SSFP(device=device, T1=T1, T2=T2, diff=diff)
	return ssfp(flip=flip, TR=TR)

# this will return signal only (evolution towards steady state of unbalanced SSFP sequence)
signal = simulate_ssfp(flip=10.0*np.ones(1000, dtype=np.float32), TR=4.5, T1=500.0, T2=50.0)

# this will also return derivatives
signal, dsignal = simulate_ssfp(flip=10.0*np.ones(1000, dtype=np.float32), TR=8.5, T1=500.0, T2=50.0, diff=("T1", "T2"))

# dsignal[0] = dsignal / dT1 (derivative of signal wrt T1)
# dsignal[1] = dsignal / dT2 (derivative of signal wrt T2)
```

This is useful e.g. for nonlinear fitting and for calculating objective functions (CRLB) for sequence optimization.

To facilitate the development of signal models, we include basic sequence building blocks (e.g., Inversion Preparation, SSFP Propagator) and low-level Extended Phase Graphs operators:


## Sequence Blocks

```{eval-rst}
.. currentmodule:: deepmr.bloch
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.bloch.InversionPrep
	deepmr.bloch.T2Prep
	deepmr.bloch.ExcPulse
	deepmr.bloch.bSSFPStep
	deepmr.bloch.SSFPFidStep
	deepmr.bloch.SSFPEchoStep
	deepmr.bloch.FSEStep
```

## Low-level Operators

```{eval-rst}
.. currentmodule:: deepmr.bloch
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.bloch.EPGstates
	deepmr.bloch.RFPulse
	deepmr.bloch.AdiabaticPulse
	deepmr.bloch.Relaxation
	deepmr.bloch.Shift
	deepmr.bloch.Spoil
	deepmr.bloch.DiffusionDamping
	deepmr.bloch.FlowDephasing
	deepmr.bloch.FlowWash
	deepmr.bloch.observe
	deepmr.bloch.susceptibility
	deepmr.bloch.t1sat
```
