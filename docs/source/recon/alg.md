# Image Reconstructors

```{eval-rst}
.. automodule:: deepmr.recon.alg
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