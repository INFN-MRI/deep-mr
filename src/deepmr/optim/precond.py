"""Polynomial preconditioning subroutines."""

__all__ = ["create_polynomial_preconditioner"]

import numpy as np
import torch

import sympy

from .._external.chebyshev import polynomial as chebpoly
from .. import linops as _linops


def create_polynomial_preconditioner(precond_type, degree, T, l=0, L=1, verbose=False):
    """
    Create polynomial preconditioner as in Srinivasan et al [1].
    Code adapted from https://github.com/sidward/ppcs/tree/main?tab=readme-ov-file

    Parameters
    ----------
    precond_type : str
        Type of preconditioner.

        * ``l_2``: ``l_2`` optimized polynomial.
        * ``l_inf``: ``l_inf`` optimized polynomial.

    degree : int
        Degree of polynomial to use.
    T : deepmr.linops.Linop
        Normal linear operator.
    l : float, optional
        Lower bound of interval. The default is ``0.0``.
    L : float, optional
        Upper bound of interval. The default is ``1.0``.
    verbose : bool, optional
        Print information. The default is ``False``.

    Returns
    -------
    P : deepmr.linops.Linop
        Polynomial preconditioner.

    References
    ----------
    [1] Srinivasan et al., Polynomial Preconditioners for Regularized Linear Inverse Problems, Arxiv preprint 2204.10252, (2022),
        DOI: https://doi.org/10.48550/arXiv.2204.10252

    """
    assert degree >= 0

    if precond_type == "l_2":
        c = l_2_opt(degree, l, L, verbose=verbose)
    elif precond_type == "l_inf":
        c = l_inf_opt(degree, l, L, verbose=verbose)
    else:
        raise Exception("Unknown norm option.")
    
    if isinstance(T, _linops.Linop) is False:
        _T = _linops.Linop(ndim=1)
        _T.forward = T
    else:
        _T = T
    I = _linops.Identity(_T.ndim)

    def phelper(c):
        if c.size()[0] == 1:
            return c[0] * I
        return c[0] * I + _T * phelper(c[1:])

    # recursively build
    P = phelper(c)

    return P


# %% local utils
def l_inf_opt(degree, l=0.0, L=1.0, verbose=False):
    """
    Calculate polynomial p(x) that minimizes the supremum of |1 - x p(x)|
    over (l, L).

    Based on Equation 50 of:
       Shewchuk, J. R.
       An introduction to the conjugate gradient method without the agonizing
       pain, Edition 1¼.

    Uses the following package:
      https://github.com/mlazaric/Chebyshev/
      DOI: 10.5281/zenodo.5831845

    Parameters
    ----------
    degree : int
        Degree of polynomial to calculate.
    l : float, optional
        Lower bound of interval. The default is ``0.0``.
    L : float, optional
        Upper bound of interval. The default is ``1.0``.
    verbose : bool, optional
        Print information. The default is ``False``.

    Returns
    -------
    coeffs : torch.Tensor
        Coefficients of optimized polynomial.

    """
    assert degree >= 0

    if verbose:
        print("L-infinity optimized polynomial.")
        print("> Degree:   %d" % degree)
        print("> Spectrum: [%0.2f, %0.2f]" % (l, L))

    T = chebpoly.get_nth_chebyshev_polynomial(degree + 1)

    y = sympy.symbols("y")
    P = T((L + l - 2 * y) / (L - l))
    P = P / P.subs(y, 0)
    P = sympy.simplify((1 - P) / y)

    if verbose:
        print("> Resulting polynomial: %s" % repr(P))

    if degree > 0:
        points = sympy.stationary_points(P, y, sympy.Interval(l, L))
        vals = np.array(
            [P.subs(y, point) for point in points] + [P.subs(y, l)] + [P.subs(y, L)]
        )
        assert np.abs(vals).min() > 1e-8, "Polynomial not injective."

    c = sympy.Poly(P).all_coeffs()[::-1] if degree > 0 else (sympy.Float(P),)

    return torch.as_tensor(np.array(c, dtype=np.float32))


def l_2_opt(degree, l=0.0, L=1.0, weight=1, verbose=False):
    """

    Calculate polynomial p(x) that minimizes the following:

    ..math:
      int_l^l w(x) (1 - x p(x))^2 dx

    To incorporate priors, w(x) can be used to weight regions of the
    interval (l, L) of the expression above.

    Based on:
      Polynomial Preconditioners for Conjugate Gradient Calculations
      Olin G. Johnson, Charles A. Micchelli, and George Paul
      DOI: 10.1137/0720025

    Parameters
    ----------
    degree : int
        Degree of polynomial to calculate.
    l : float, optional
        Lower bound of interval. The default is ``0.0``.
    L : float, optional
        Upper bound of interval. The default is ``1.0``.
    weight : SymPy
        Sympy expression to include prior weight.
    verbose : bool, optional
        Print information. The default is ``False``.

    Returns
    -------
    coeffs : torch.Tensor
        Coefficients of optimized polynomial.

    """
    if verbose:
        print("L-2 optimized polynomial.")
        print("> Degree:   %d" % degree)
        print("> Spectrum: [%0.2f, %0.2f]" % (l, L))

    c = sympy.symbols("c0:%d" % (degree + 1))
    x = sympy.symbols("x")

    p = sum([(c[k] * x**k) for k in range(degree + 1)])
    f = weight * (1 - x * p) ** 2
    J = sympy.integrate(f, (x, l, L))

    mat = [[0] * (degree + 1) for _ in range(degree + 1)]
    vec = [0] * (degree + 1)

    for edx in range(degree + 1):
        eqn = sympy.diff(J, c[edx])
        tmp = eqn.copy()
        # Coefficient index
        for cdx in range(degree + 1):
            mat[edx][cdx] = float(sympy.Poly(eqn, c[cdx]).coeffs()[0])
            tmp = tmp.subs(c[cdx], 0)
        vec[edx] = float(-tmp)

    mat = np.array(mat, dtype=np.double)
    vec = np.array(vec, dtype=np.double)
    res = np.array(np.linalg.pinv(mat) @ vec, dtype=np.float32)

    poly = sum([(res[k] * x**k) for k in range(degree + 1)])
    if verbose:
        print("> Resulting polynomial: %s" % repr(poly))

    if degree > 0:
        points = sympy.stationary_points(poly, x, sympy.Interval(l, L))
        vals = np.array(
            [poly.subs(x, point) for point in points]
            + [poly.subs(x, l)]
            + [poly.subs(x, L)]
        )
        assert vals.min() > 1e-8, "Polynomial is not positive."

    return torch.as_tensor(res)
