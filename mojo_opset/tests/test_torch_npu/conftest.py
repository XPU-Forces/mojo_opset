"""Shared fixtures for torch_npu operator unit tests."""

import pytest
import torch

from mojo_opset.tests.utils import get_torch_device
from mojo_opset.utils.platform import get_platform


@pytest.fixture(scope="module")
def npu_device():
    """Current NPU ``torch.device``; skips if no Ascend NPU."""
    pytest.importorskip("torch_npu")
    if get_platform() != "npu" or not torch.npu.is_available():
        pytest.skip("Requires Ascend NPU (e.g. A2, A5/950PR).")
    return torch.device(get_torch_device())
