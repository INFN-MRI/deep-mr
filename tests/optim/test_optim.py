"""Test optimization routines."""

import itertools
import pytest

import numpy as np
import numpy.testing as npt

import torch
import deepmr

dtype = [torch.float32, torch.float64, torch.complex64, torch.complex128]
device = ["cpu"]
if torch.cuda.is_available():
    device += ["cuda"]

# tolerance
tol = 1e-4


@pytest.mark.parametrize("dtype, device", list(itertools.product(*[dtype, device])))
def test_power_method(dtype, device):
    # setup problem
    n = 5
    A, _ = Ax_setup(n, dtype, device)
    x_hat = torch.rand(n, 1, dtype=dtype, device=device)

    # define function
    def AHA(x):
        return A.T @ A @ x

    # actual calculation
    x = deepmr.optim.power_method(A, x_hat, AHA=AHA, device=device, niter=30)
    x_numpy = np.linalg.svd(A.numpy(force=True), compute_uv=False)[0]

    # check
    npt.assert_allclose(x, x_numpy, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype, device", list(itertools.product(*[dtype, device])))
def test_conjugate_gradient(dtype, device):
    # setup problem
    n = 5
    lamda = 0.1
    A, x = Ax_setup(n, dtype, device)
    y = A @ x

    # rescale
    LL = np.linalg.svd(
        (A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device)).numpy(force=True),
        compute_uv=False,
    )[0]
    A = (1 / LL) ** 0.5 * A
    y = y / torch.linalg.norm(y)

    # calculate solution
    x_torch = torch.linalg.solve(
        A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device), A.T @ y
    )

    # define function
    AHA = _as_linop(A)

    # actual calculation
    x, _ = deepmr.optim.cg_solve(A.T @ y, AHA, niter=1000, lamda=lamda)

    # check
    npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype, device", list(itertools.product(*[dtype, device])))
def test_polynomial_inversion(dtype, device):
    # setup problem
    n = 5
    lamda = 0.1
    A, x = Ax_setup(n, dtype, device)
    y = A @ x

    # rescale
    LL = np.linalg.svd(
        (lamda * A.T @ A + torch.eye(n, dtype=dtype, device=device)).numpy(force=True),
        compute_uv=False,
    )[0]
    A = (1 / LL) ** 0.5 * A
    y = y / torch.linalg.norm(y)

    # calculate solution
    x_torch = torch.linalg.solve(
        lamda * A.T @ A + torch.eye(n, dtype=dtype, device=device), lamda * A.T @ y
    )

    # define function
    AHA = _as_linop(A)

    # actual calculation
    x = deepmr.optim.polynomial_inversion(A.T @ y, AHA, niter=10, lamda=lamda)

    # check
    npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "dtype, device, accelerate",
    list(itertools.product(*[dtype, device, [True, False]])),
)
def test_proximal_gradient(dtype, device, accelerate):
    # setup problem
    n = 5
    lamda = 0.1
    A, x = Ax_setup(n, dtype, device)
    y = A @ x

    # rescale
    LL = np.linalg.svd(
        (A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device)).numpy(force=True),
        compute_uv=False,
    )[0]
    A = (1 / LL) ** 0.5 * A
    y = y / torch.linalg.norm(y)

    # calculate solution
    x_torch = torch.linalg.solve(
        A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device), A.T @ y
    )

    # define function
    _AHA = _as_linop(A)
    if lamda != 0.0:
        AHA = _AHA + lamda * deepmr.linops.Identity()
    else:
        AHA = _AHA

    # prepare denoiser
    D = Denoiser(ths=lamda)

    # actual calculation
    x, _ = deepmr.optim.pgd_solve(
        A.T @ y, 1.0, AHA, D, niter=1000, accelerate=accelerate
    )

    # check
    npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "dtype, device, accelerate",
    list(itertools.product(*[dtype, device, [True, False]])),
)
def test_polynomial_preconditioned_proximal_gradient(dtype, device, accelerate):
    # setup problem
    l = 4
    n = 5
    lamda = 0.1
    A, x = Ax_setup(n, dtype, device)
    y = A @ x

    # rescale
    LL = np.linalg.svd(
        (A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device)).numpy(force=True),
        compute_uv=False,
    )[0]
    A = (1 / LL) ** 0.5 * A
    y = y / torch.linalg.norm(y)

    # calculate solution
    x_torch = torch.linalg.solve(
        A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device), A.T @ y
    )

    # define function
    _AHA = _as_linop(A)
    if lamda != 0.0:
        AHA = _AHA + lamda * deepmr.linops.Identity()
    else:
        AHA = _AHA

    # prepare denoiser
    D = Denoiser(ths=lamda)

    # define preconditioner
    P = deepmr.optim.precond.create_polynomial_preconditioner("l_2", l, AHA)

    # actual calculation
    x, _ = deepmr.optim.pgd_solve(
        A.T @ y, 1.0, AHA, D, P=P, niter=250, accelerate=accelerate
    )

    # check
    npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)


# @pytest.mark.parametrize("dtype, device", list(itertools.product(*[dtype, device])))
# def test_admm(dtype, device):
#     # setup problem
#     n = 5
#     step = 1.0
#     A, x_torch, y = Ax_y_setup(n, 0.0, dtype, device)

#     # define function
#     def AHA(x):
#         return A.T @ A @ x  # + step * x

#     # prepare denoiser
#     def D(x):
#         return x  # / (1.0 + lamda / step)

#     # actual calculation
#     x, _ = deepmr.optim.admm_solve(A.T @ y, step, AHA, D, niter=1000, dc_niter=1000)

#     # check
#     npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)


# @pytest.mark.parametrize("dtype, device", list(itertools.product(*[dtype, device])))
# def test_lstsq(dtype, device):
#     # setup problem
#     n = 5
#     lamda = 0.0
#     A, x_torch, y = Ax_y_setup(n, lamda, dtype, device)

#     # define function
#     def AH(x):
#         return A.T @ x

#     def AHA(x):
#         return A.T @ A @ x

#     # prepare denoiser
#     D = Denoiser(ths=lamda)

#     # CG
#     x_cg, _ = deepmr.optim.lstsq(y, AH, AHA, niter=1000, lamda=lamda, ndim=1)

#     # check
#     npt.assert_allclose(x_cg.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)

#     # PGD
#     x_pgd, _ = deepmr.optim.lstsq(y, AH, AHA, prior=D, niter=1000, lamda=lamda)

#     # check
#     npt.assert_allclose(
#         x_pgd.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol
#     )

#     # PP-PGD
#     # x_pppgd, _ = deepmr.optim.lstsq(y, AH, AHA, prior=D, niter=1, lamda=lamda, ndim=1, use_precond=True, precond_degree=4)

#     # # check
#     # npt.assert_allclose(x_pppgd.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)

#     # ADMM
#     x_admm, _ = deepmr.optim.lstsq(y, AH, AHA, prior=[D], niter=1000, lamda=lamda)

#     # check
#     npt.assert_allclose(
#         x_admm.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol
#     )


# %% local subroutines
def Ax_setup(n, dtype, device):
    A = torch.eye(n) + 0.1 * torch.ones([n, n])
    A = A.to(dtype=dtype)
    A = A.to(device=device)
    x = torch.arange(n, device=device)
    x = x.to(dtype=dtype)
    return A, x


class Denoiser(torch.nn.Module):
    def __init__(self, ths=1.0):
        super().__init__()
        self.ths = ths

    def forward(self, x):
        return x  # / (1.0 + self.ths)


def _as_linop(A):
    Aop = deepmr.linops.Linop()
    AHop = deepmr.linops.Linop()

    _Aop = deepmr.linops.Linop()
    _AHop = deepmr.linops.Linop()

    Aop.forward = lambda x: A.T @ A @ x
    AHop.forward = lambda x: A.T @ A @ x

    _Aop.forward = lambda x: A.T @ A @ x
    _AHop.forward = lambda x: A.T @ A @ x

    Aop._adjoint_linop = lambda: _AHop
    AHop._adjoint_linop = lambda: _Aop

    return Aop
