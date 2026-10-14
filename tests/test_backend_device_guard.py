"""Test device dispatch without requiring an accelerator runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def backend():
    source = Path(__file__).parents[1] / "src/deepmr/_utils/backend.py"
    spec = importlib.util.spec_from_file_location("backend_under_test", source)
    module = importlib.util.module_from_spec(spec)
    # Isolate the conversion module from the package's reconstruction imports.
    with patch.dict("sys.modules", {"torch": Mock(), "numba": Mock()}):
        spec.loader.exec_module(module)
    return module


def test_cpu_conversion_keeps_the_tensor_conversion_path(backend):
    tensor = Mock(device=SimpleNamespace(type="cpu"))
    expected = object()
    tensor.detach.return_value.contiguous.return_value.numpy.return_value = expected
    assert backend.pytorch2numba(tensor) is expected
    backend.nb.cuda.as_cuda_array.assert_not_called()


@pytest.mark.parametrize("device_type", ["mps", "xpu"])
def test_unsupported_device_fails_before_conversion(backend, device_type):
    tensor = Mock(device=SimpleNamespace(type=device_type))
    with pytest.raises(ValueError, match="select CPU execution and CPU tensors"):
        backend.pytorch2numba(tensor)
    tensor.detach.assert_not_called()
    backend.nb.cuda.as_cuda_array.assert_not_called()
