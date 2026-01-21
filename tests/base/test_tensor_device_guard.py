import torch
import pytest
import inspect
from mojo_opset.backends.ttx.kernels.npu.kv_cache import store_paged_kv_impl

def test_tensor_device_guard():
    with pytest.raises(TypeError, match="Found cpu tensor.*triton kernel."):
        store_paged_kv_impl(
            *[torch.empty(*[8]*4)]
            * len(inspect.signature(store_paged_kv_impl).parameters)
        )
