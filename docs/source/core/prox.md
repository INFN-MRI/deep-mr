# Proximal Operators

```{eval-rst}
.. automodule:: deepmr.prox
```

## Classical Denoisers

Classical (i.e., non-trainable) denoisers can be used inside
iterative reconstructions as regularizars (aka, PnP reconstruction).

DeepMR expose denoisers both as ``torch.nn`` objects (to be chained in DL-based reconstructions)
and functional forms (e.g., for simple standalone denoising).

Currently available denoisers are soft-thresholded L1 Wavelet (``WaveletPrior``), soft-thresholded
Local Low Rank (``LLRPrior``) and Total Variation (``TVPrior``).

Both ``WaveletPrior`` and ``TVPrior`` are wrappers around the corresponding [DeepInverse](https://deepinv.github.io/deepinv/) implementations.

```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.prox.WaveletPrior
	deepmr.prox.TVPrior
	deepmr.prox.LLRPrior

	deepmr.prox.wavelet_denoise
	deepmr.prox.tv_denoise
	deepmr.prox.llr_denoise
```


