# Image Reconstructors

```{eval-rst}
.. automodule:: deepmr.recon.alg
```

## Building blocks

DeepMR contains a convenient builder routines to automatically initialize the MR encoding
operator for different reconstruction problems 
(e.g., Cartesian vs Non-Cartesian, single- vs multi-channel, single- vs multi-contrast).

This can be used inside conventional reconstruction algorithms, or inside neural network architectures.

```{eval-rst}
.. currentmodule:: deepmr 
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.recon.EncodingOp
```

## Classical Image Reconstruction

DeepMR contains convenient wrappers around generic standard reconstruction.

The provided routine can perform zero-filled or iterative reconstruction
with Tikonhov, L1Wavelet or Total Variation regularization for both Cartesian
and Non-Cartesian (single-/multi- contrast/channel) data, depending on input arguments.

```{eval-rst}
.. currentmodule:: deepmr 
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.recon.recon_lstsq
```