"""Test optimization routines."""

import itertools
import pytest

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
def test_conjugate_gradient(dtype, device):
    # setup problem
    n = 5
    lamda = 0.1
    A, x_torch, y = Ax_y_setup(n, lamda, dtype, device)

    # define function
    def AHA(x):
        return A.T @ A @ x

    # actual calculation
    x = deepmr.optim.cg_solve(A.T @ y, AHA, niter=1000, lamda=lamda, ndim=2)

    # check
    npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)
    

@pytest.mark.parametrize("dtype, device", list(itertools.product(*[dtype, device])))
def test_lsmr(dtype, device):
    # setup problem
    n = 5
    lamda = 0.1
    A, x_torch, y = Ax_y_setup(n, lamda, dtype, device)

    # define function
    def AHA(x):
        return A.T @ A @ x

    # actual calculation
    x = deepmr.optim.lsmr_solve(A.T @ y, AHA, niter=1000, lamda=lamda, ndim=2)

    # check
    npt.assert_allclose(x.detach().cpu(), x_torch.detach().cpu(), rtol=tol, atol=tol)


# %% local subroutines
def Ax_setup(n, dtype, device):
    A = torch.eye(n) + 0.1 * torch.ones([n, n])
    A = A.to(dtype=dtype)
    A = A.to(device=device)
    x = torch.arange(n, device=device)
    x = x.to(dtype=dtype)
    return A, x


def Ax_y_setup(n, lamda, dtype, device):
    A, x = Ax_setup(n, dtype, device)
    y = A @ x
    x_torch = torch.linalg.solve(
        A.T @ A + lamda * torch.eye(n, dtype=dtype, device=device), A.T @ y
    )

    return A, x_torch, y
