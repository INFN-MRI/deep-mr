"""Preconditioned Olug-and-Play FISTA algorithm."""

__all__ = ["pnp_fista"]

import time

from tqdm import tqdm

from numpy.typing import NDArray
import numpy as np

from mrops import _sigpy as sp

from ..precond import create_polynomial_preconditioner


def pnp_fista(
    num_iters: int,
    ptol: float,
    A: sp.linop.Linop,
    b: NDArray[complex],
    proxg: sp.prox.Prox,
    x0: NDArray[complex] | None = None,
    precond_type: str | None = None,
    pdeg: int | None = None,
    accelerate: bool = True,
    l: float = 0,
    a: float = 2.1,
    stepfn: callable | None = None,
    ref: NDArray[complex] | None = None,
    save: str | None = None,
    verbose: bool = True,
    idx: tuple | None = None,
) -> NDArray[complex]:
    """
    Proximal Gradient Descent (FISTA).

    Solves the following optimization problem using proximal gradient descent:

    .. math::
        \min_x \frac{1}{2} \| A x - b \|_2^2 + g(x)

    Assumes MaxEig(A.H * A) = 1.

    Parameters
    ----------
    num_iters : int
        Maximum number of iterations.
    ptol : float
        l1-percentage tolerance between iterates.
    A : Linop
        Forward model operator.
    b : NDArray[complex]
        Measurement vector.
    proxg : Prox
        Proximal operator of g.
    x0 : NDArray[complex], optional
        Initial guess. If None, will be set to `A.H(b)`.
    precond_type : str, optional
        Type of preconditioner:
        - "l_2" : l_2 optimized polynomial.
        - "l_inf" : l_inf optimized polynomial.
        - "ifista" : from DOI: 10.1137/140970537.
    pdeg : int, optional
        Degree of polynomial preconditioner to use. If None, no preconditioner is used.
    accelerate : bool, optional
        If True, uses Nesterov acceleration (FISTA).
    l : float, optional
        If known, the minimum eigenvalue of A.H * A.
    a : float, optional
        Parameter from DOI: 10.1561/2400000003.
    stepfn : callable, optional
        If specified, determines the variable step size per iteration.
    ref : NDArray[complex], optional
        Reference to compare against for error calculation.
    save : str, optional
        If specified, path to save iterations and timings.
    verbose : bool, optional
        If True, print information during iterations.
    idx : tuple, optional
        If provided, slice iterates before saving.

    Returns
    -------
    NDArray[complex]
        The reconstruction vector.
    """
    if precond_type == "l_inf" and l == 0:
        raise ValueError("If l == 0, l_inf polynomial cannot be used.")

    device = sp.get_device(b)
    if verbose:
        print("Proximal Gradient Descent.")
        if accelerate:
            print(
                "> Variable step size."
                if l == 0
                else f"> Fixed step size derived from l = {l:.2f}"
            )
        print(
            "> No preconditioning used."
            if precond_type is None
            else f"> {precond_type}-preconditioning is used."
        )

    P = (
        sp.linop.Identity(A.ishape)
        if pdeg is None
        else create_polynomial_preconditioner(
            precond_type, pdeg, A.N, l, 1, verbose=verbose
        )
    )

    with device:
        lst_time = []
        lst_err = None
        if ref is not None:
            ref = sp.to_device(ref, device)
            lst_err = ([], [])

        # Set-up time.
        start_time = time.perf_counter()
        AHb = A.H(b)
        x = AHb if x0 is None else sp.to_device(x0, device)
        z = x.copy()
        end_time = time.perf_counter()

        lst_time.append(end_time - start_time)
        if lst_err is not None:
            lst_err[0].append(calc_perc_err(ref, x, ord=1))
            lst_err[1].append(calc_perc_err(ref, x, ord=2))
        save_helper(save, x, 0, lst_time, lst_err, idx)

        if verbose:
            pbar = tqdm(total=num_iters, desc="PGD", leave=True)

        for k in range(num_iters):
            start_time = time.perf_counter()

            x_old = x.copy()
            if accelerate:
                x = z.copy()

            gr = A.N(x) - AHb
            x = proxg(1, x - P(gr))

            if accelerate:
                if l > 0:
                    # DOI: 10.1007/978-3-319-91578-4_2
                    step = (1 - l**0.5) / (1 + l**0.5)
                elif stepfn is None:
                    # DOI: 10.1561/2400000003
                    step = k / (k + a + 1)
                else:
                    step = stepfn(k)
                z = x + step * (x - x_old)

            end_time = time.perf_counter()

            lst_time.append(end_time - start_time)
            if lst_err is not None:
                lst_err[0].append(calc_perc_err(ref, x, ord=1))
                lst_err[1].append(calc_perc_err(ref, x, ord=2))
            save_helper(save, x, k + 1, lst_time, lst_err, idx)

            calc_tol = calc_perc_err(x_old, x, ord=1)
            if verbose:
                pbar.set_postfix(ptol=f"{calc_tol:.2f}%")
                pbar.update()
                pbar.refresh()

            if calc_tol <= ptol:
                break

        if verbose:
            pbar.close()

    return x


# %% utils
def calc_perc_err(
    ref: NDArray[complex],
    x: NDArray[complex],
    ord: int = 2,
    auto_normalize: bool = True,
) -> float:
    """
    Calculate the percentage error between two vectors.

    Parameters
    ----------
    ref : NDArray[complex]
        The reference vector.
    x : NDArray[complex]
        The vector to compare against.
    ord : int, optional
        The order of the norm. Default is 2 (Euclidean norm).
    auto_normalize : bool, optional
        If True, the vectors are normalized before computing the error.

    Returns
    -------
    float
        The percentage error between `ref` and `x`.
    """
    dev = sp.get_device(x)
    xp = dev.xp
    with dev:
        if auto_normalize:
            p = ref / xp.linalg.norm(ref.ravel(), ord=ord)
            q = x / xp.linalg.norm(x.ravel(), ord=ord)
            err = xp.linalg.norm((p - q).ravel(), ord=ord)
        else:
            err = xp.linalg.norm((ref - x).ravel(), ord=ord)
            err /= xp.linalg.norm(ref.ravel(), ord=ord)
        err = sp.to_device(err, sp.cpu_device)

    if np.isnan(err) or np.isinf(err):
        return 100
    return 100 * err


def save_helper(
    save: str,
    x: NDArray[complex],
    itr: int,
    lst_time: list,
    lst_err: list,
    idx: tuple,
    obj: NDArray[complex] = None,
):
    """
    Save the iteration results, including the solution and error.

    Parameters
    ----------
    save : str
        Directory path to save results.
    x : NDArray[complex]
        The solution vector to save.
    itr : int
        The current iteration number.
    lst_time : list
        List of elapsed times for each iteration.
    lst_err : list
        List of error values for each iteration.
    idx : tuple
        Index or slice of iterates to save.
    obj : NDArray[complex], optional
        The objective values to save.
    """
    if save is None:
        return

    tp = sp.get_array_module(x)
    np.save(f"{save}/time.npy", np.cumsum(lst_time))
    if idx is None:
        tp.save(f"{save}/iter_{itr:03d}.npy", x)
    else:
        tp.save(f"{save}/iter_{itr:03d}.npy", x[idx])

    if lst_err is not None:
        np.save(f"{save}/err.npy", lst_err)
