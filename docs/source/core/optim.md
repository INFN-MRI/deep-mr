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
	
	deepmr.optim.CGStep
	deepmr.optim.ADMMStep
	deepmr.optim.PGDStep

	deepmr.optim.cg_solve
	deepmr.optim.admm_solve
	deepmr.optim.pgd_solve
```

In addition, we provide utils to estimate matrix-free operator properties, such as maximum eigenvalue.

## Linop linear algebra

```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.optim.power_method
```

