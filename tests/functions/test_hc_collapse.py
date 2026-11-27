# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Case matrix adapted for Ascend NPU by the mojo_opset contributors.
import pytest
import torch

from mojo_opset import functions
from mojo_opset import preload
from tests._checks import assert_close
from tests._compile import compile_fullgraph

# Original vLLM mHC kernel case matrix (tests/kernels/mhc/test_mhc_kernels.py):
# token tails (0/1/33), full tiles (128/1024), a single stream, a wide
# hc_mult, a non-power-of-two hidden size and a very wide token axis with a
# tiny hidden size (sequence-parallel tail blocks).
HC_COLLAPSE_CASES = [
    (0, 4, 5120),
    (1, 4, 5120),
    (33, 4, 5137),
    (128, 4, 5120),
    (1024, 4, 5120),
    (2, 1, 513),
    (7, 8, 5137),
    (65537, 4, 65),
]


@pytest.mark.api("functions.hc_collapse")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens,hc_mult,hidden_size", HC_COLLAPSE_CASES)
@pytest.mark.parametrize("strided", [False, True])
def test_hc_collapse_preserves_weighted_residual_sum(
    accuracy_backend, num_tokens, hc_mult, hidden_size, strided
):
    """Handle tile tails and strided streams without changing FP32 mixing."""
    implementation, _, device = accuracy_backend
    torch.manual_seed(0)
    x = torch.randn(
        num_tokens,
        hc_mult,
        hidden_size * (2 if strided else 1),
        dtype=torch.bfloat16,
        device=device,
    )
    pre = torch.rand(num_tokens, hc_mult, device=device)
    if strided:
        x = x[..., ::2]
        pre = pre.t().contiguous().t()
    if num_tokens:
        pre[0].zero_()
        pre[0, -1] = 1
    if num_tokens > 1 and hc_mult == 4:
        pre[1].fill_(1)
        x[1, 1] = -x[1, 0]
        x[1, 3] = -x[1, 2]
    expected = functions.hc_collapse(x, pre, implementation="torch")
    actual = functions.hc_collapse(x, pre, implementation=implementation)
    assert actual.is_contiguous()
    # BF16 rounding only: contraction of the FP32 mix may differ by one BF16
    # ulp, so compare within a ulp and pin the exact cases below.
    assert_close(actual, expected, torch.bfloat16, rtol=1e-2, atol=1.6e-2)
    if num_tokens:
        assert_close(actual[0], x[0, -1], rtol=0, atol=0)
    if num_tokens > 1 and hc_mult == 4:
        assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)


@pytest.mark.api("functions.hc_collapse")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_hc_collapse_before_gather_is_bitwise_equal(accuracy_backend, num_tokens, tp_size):
    """Simulate sequence parallelism on one device, with padding and ranks
    holding no valid tokens: collapsing before or after the gather must be
    bitwise identical because every token is collapsed locally."""
    implementation, _, device = accuracy_backend
    torch.manual_seed(0)
    padded_tokens = num_tokens + (-num_tokens) % tp_size
    streams = torch.randn(padded_tokens, 4, 5120, dtype=torch.bfloat16, device=device)
    pre_mix = torch.rand(padded_tokens, 4, device=device)
    stream_shards = streams.chunk(tp_size)
    mix_shards = pre_mix.chunk(tp_size)
    before = functions.hc_collapse(
        torch.cat(stream_shards)[:num_tokens],
        torch.cat(mix_shards)[:num_tokens],
        implementation=implementation,
    )
    after = torch.cat(
        [
            functions.hc_collapse(x, pre, implementation=implementation)
            for x, pre in zip(stream_shards, mix_shards)
        ]
    )[:num_tokens]
    assert after.shape == (num_tokens, 5120)
    assert_close(after, before, rtol=0, atol=0)


@pytest.mark.api("functions.hc_collapse")
@pytest.mark.accuracy
def test_hc_collapse_custom_op_supports_compile(accuracy_backend):
    """The registered custom op keeps schema/fake metadata and graph capture."""
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu"):
        pytest.skip("NPU Triton custom-op compile contract")
    preload("hc_collapse", implementation="triton")
    namespace = "mojo_npu_triton_a5" if target.startswith("npu.a5") else "mojo_npu_triton_a2"
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16, device=device)
    pre = torch.rand(2, 4, device=device)
    torch.library.opcheck(
        getattr(torch.ops, namespace).hc_collapse_fwd.default,
        (x, pre),
        test_utils=("test_schema", "test_faketensor"),
    )

    def run(x, pre):
        return functions.hc_collapse(x, pre, implementation="triton")

    try:
        eager = run(x, pre).clone()
        compiled = compile_fullgraph(run)
        actual = compiled(x, pre)
        assert_close(actual, eager, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
