import importlib
import os

import pytest
import torch

os.environ["MOJO_DISABLE_TENSOR_GUARD"] = "No"
# WARNING(liuyuan): The following reload is necessary when running with other tests
from mojo_opset.backends.ttx.kernels import npu

importlib.reload(npu)
from mojo_opset.backends.ttx.kernels.npu.convolution import causal_conv1d_fwd
from mojo_opset.backends.ttx.kernels.npu.kv_cache import store_paged_kv_impl


def test_tensor_device_guard():
    with pytest.raises(TypeError, match="Found cpu tensor.*triton kernel."):
        store_paged_kv_impl(*[torch.empty(*[8] * 4)] * 5, torch.tensor([2, 4]), torch.tensor([2, 2]))

    with pytest.raises(TypeError, match="Found cpu tensor.*triton kernel."):
        causal_conv1d_fwd(*[torch.empty(*[256] * 3)] * 4)
