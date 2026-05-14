"""Helpers for torch_npu unit tests."""

from __future__ import annotations

import pytest
import torch

from mojo_opset.tests.utils import get_torch_device
from mojo_opset.utils.platform import get_platform


def require_npu() -> torch.device:
    pytest.importorskip("torch_npu")
    if get_platform() != "npu" or not torch.npu.is_available():
        pytest.skip("Requires Ascend NPU (A2, A5/950PR, …).")
    return torch.device(get_torch_device())


def assert_close_npu(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    equal_nan: bool = True,
) -> None:
    torch.testing.assert_close(
        got.detach().cpu().float(),
        ref.detach().cpu().float(),
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )
