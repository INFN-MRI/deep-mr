"""Utils for linop testing."""

import torch
import numpy.testing as npt

import pytest

seed = 42
atol = 1e-3
rtol = 1e-3


class Helpers:  # noqa
    @staticmethod
    def check_linop_unitary(A, data):  # noqa
        torch.manual_seed(seed)
        x = torch.rand(*data.shape, dtype=data.dtype, device=data.device)
        AHA = A.H * A
        npt.assert_allclose(
            AHA(x).numpy(force=True), x.numpy(force=True), atol=atol, rtol=rtol
        )

    @staticmethod
    def check_linop_linear(A, data):  # noqa
        torch.manual_seed(seed)
        a = torch.rand(1, dtype=data.dtype, device=data.device)
        x = torch.rand(*data.shape, dtype=data.dtype, device=data.device)
        y = torch.rand(*data.shape, dtype=data.dtype, device=data.device)
        npt.assert_allclose(
            A(a * x + y).numpy(force=True),
            (a * A(x) + A(y)).numpy(force=True),
            atol=atol,
            rtol=rtol,
        )

    @staticmethod
    def check_linop_adjoint(A, idata, odata):  # noqa
        torch.manual_seed(seed)
        x = torch.rand(*idata.shape, dtype=idata.dtype, device=idata.device)
        y = torch.rand(*odata.shape, dtype=odata.dtype, device=odata.device)

        lhs = _vdot(A(x), y)
        rhs = _vdot(A.H.H(x), y)
        npt.assert_allclose(
            lhs.numpy(force=True), rhs.numpy(force=True), atol=atol, rtol=rtol
        )

        rhs = _vdot(x, A.H(y))
        npt.assert_allclose(lhs.numpy(force=True), rhs.numpy(force=True), rtol=rtol)


@pytest.fixture
def helpers():  #  noqa
    return Helpers


# %% local utils
def _vdot(a, b):
    return (a.conj().flatten() * b.flatten()).sum()
