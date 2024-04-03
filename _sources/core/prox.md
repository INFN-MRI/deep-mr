# Proximal Operators

```{eval-rst}
.. automodule:: deepmr.prox
```

## Classical Denoisers

Classical (i.e., non-trainable) denoisers can be used inside
iterative reconstructions as regularizars (aka, PnP reconstruction).

DeepMR expose denoisers both as ``torch.nn`` objects (to be chained in DL-based reconstructions)
and functional forms (e.g., for simple standalone denoising).

Currently available denoisers are Wavelet (``WaveletDenoiser``, ``WaveletDictDenoiser``),
Local Low Rank (``LLRDenoiser``) and Total (Generalized) Variation (``TVDenoiser``, ``TGVDenoiser`).

Both ``WaveletDenoiser`` / ``WaveletDictDenoiser`` and ``TVDenoiser`` / ``TGVDenoiser`` are adapted for complex-value inputs from the corresponding [DeepInverse](https://deepinv.github.io/deepinv/) implementations.

```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.prox.WaveletDenoiser
	deepmr.prox.WaveletDictDenoiser
	deepmr.prox.TVDenoiser
	deepmr.prox.TGVDenoiser
	deepmr.prox.LLRDenoiser

	deepmr.prox.wavelet_dict_denoise
	deepmr.prox.wavelet_denoise
	deepmr.prox.tv_denoise
	deepmr.prox.tgv_denoise
	deepmr.prox.llr_denoise
```


