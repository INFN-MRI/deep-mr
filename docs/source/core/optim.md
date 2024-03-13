# Iterative Algorithms

```{eval-rst}
.. automodule:: deepmr.optim
```

## Optimization Steps

Operators representing single iterations of classical optimization algorithms.

DeepMR expose these operators as ``torch.nn`` objects to be chained e.g., in unrolled Neural Network architectures.

Currently available optimizers are wrappers around the corresponding [DeepInverse](https://deepinv.github.io/deepinv/) implementations.

```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.optim.ADMMIteration
	deepmr.optim.PGDIteration
	deepmr.optim.GDIteration
	deepmr.optim.CPIteration
	deepmr.optim.DRSIteration
	deepmr.optim.HQSIteration
```
