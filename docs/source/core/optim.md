# Iterative Algorithms

```{eval-rst}
.. automodule:: deepmr.optim
```

## Optimization Steps

Operators representing single iterations of classical optimization algorithms.

DeepMR expose these operators as ``torch.nn`` objects to be chained e.g., in unrolled Neural Network architectures
and the corresponding functional versions for standalone usage.

```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.optim.ADMMStep
	deepmr.optim.PGDStep

	deepmr.optim.admm_solve
	deepmr.optim.pgd_solve
```
