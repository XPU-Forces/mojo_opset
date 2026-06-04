import os
import inspect

import pytest

from mojo_opset import MojoApplyRoPE
from mojo_opset import MojoExperts
from mojo_opset import MojoEmbedding
from mojo_opset import MojoGemm
from mojo_opset import MojoGroupRMSNorm
from mojo_opset import MojoMoE
from mojo_opset import MojoMoECombine
from mojo_opset import MojoMoEDispatch
from mojo_opset import MojoMoEGating
from mojo_opset import MojoOverEncodingNGram
from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoRotaryEmbedding
from mojo_opset import MojoStorePagedKVCache


TORCH_NPU_KERNEL_OPERATORS = [
    MojoGemm,
    MojoRMSNorm,
    MojoGroupRMSNorm,
    MojoApplyRoPE,
    MojoPagedPrefillGQA,
    MojoPagedDecodeGQA,
    MojoMoEGating,
    MojoExperts,
]

TORCH_NPU_ROUTE_ONLY_OPERATORS = [
    # RotaryEmbedding generates cos/sin tables; npu_rotary_mul is used by MojoApplyRoPE.
    MojoRotaryEmbedding,
    # Registered under torch_npu, but still keeps the core block-table write semantics.
    MojoStorePagedKVCache,
    # Composite MoE route shells. Dedicated torch_npu kernels live in gating/experts.
    MojoMoE,
    MojoMoEDispatch,
    MojoMoECombine,
]

TORCH_FALLBACK_OPERATORS = [
    MojoEmbedding,
    MojoOverEncodingNGram,
    MojoPagedPrefillSWA,
    MojoPagedDecodeSWA,
]


def _assert_bound_to_torch_npu_backend(op_cls):
    impl = op_cls._registry.get("torch_npu")
    impl_file = os.path.realpath(inspect.getfile(impl))

    assert impl.__module__.startswith("mojo_opset.backends.torch_npu.")
    assert "/mojo_opset/backends/torch_npu/" in impl_file


@pytest.mark.parametrize("op_cls", TORCH_NPU_KERNEL_OPERATORS)
def test_torch_npu_kernel_backend_binding(op_cls):
    _assert_bound_to_torch_npu_backend(op_cls)


@pytest.mark.parametrize("op_cls", TORCH_NPU_ROUTE_ONLY_OPERATORS)
def test_torch_npu_route_only_backend_binding(op_cls):
    _assert_bound_to_torch_npu_backend(op_cls)


@pytest.mark.parametrize("op_cls", TORCH_FALLBACK_OPERATORS)
def test_excluded_ops_fallback_to_torch(op_cls):
    assert "torch_npu" not in op_cls._registry._registry

    impl = op_cls._registry.get("torch")
    assert getattr(impl, "_backend", None) == "torch"
