# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Case matrix adapted for Ascend NPU by the mojo_opset contributors.
import pytest
import torch

from mojo_opset import functions
from mojo_opset import preload
from tests._checks import assert_close
from tests._compile import compile_fullgraph

# Case matrix anchored to the vLLM mHC kernel suite
# (tests/kernels/mhc/test_mhc_kernels.py): token tails, full tiles, a single
# stream, a wide hc_mult, a non-power-of-two hidden size and a very wide
# token axis with a tiny hidden size. The (4, 4096) rows are the GLM-5.3-Flash
# model spec (mhc=true, hc_mult=4, hidden_size=4096): decode tails through
# prefill tiles.
MHC_POST_CASES = [
    (0, 4, 4096),
    (1, 4, 4096),
    (33, 4, 5137),
    (128, 4, 4096),
    (1024, 4, 4096),
    (2, 1, 513),
    (7, 8, 5137),
    (8, 4, 7168),
    (65537, 4, 65),
]


@pytest.mark.api("functions.mhc_post")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens,hc_mult,hidden_size", MHC_POST_CASES)
@pytest.mark.parametrize("strided", [False, True])
def test_mhc_post_preserves_weighted_residual_mix(accuracy_backend, num_tokens, hc_mult, hidden_size, strided):
    """Handle tile tails and strided streams without changing FP32 mixing."""
    implementation, _, device = accuracy_backend
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size * (2 if strided else 1), dtype=torch.bfloat16, device=device)
    residual = torch.randn(
        num_tokens,
        hc_mult,
        hidden_size * (2 if strided else 1),
        dtype=torch.bfloat16,
        device=device,
    )
    post = torch.rand(num_tokens, hc_mult, 1, device=device)
    comb = torch.rand(num_tokens, hc_mult, hc_mult, device=device)
    if strided:
        x = x[:, ::2]
        residual = residual[..., ::2]
        post = torch.rand(num_tokens, 2 * hc_mult, 1, device=device)[..., ::2, :]
        comb = torch.rand(num_tokens, 2 * hc_mult, 2 * hc_mult, device=device)[..., ::2, ::2]
    if num_tokens:
        # Pinned rows: identity comb with a zero post term reproduces the
        # residual streams exactly; a zero comb with a unit post term
        # reproduces the sublayer output exactly.
        comb[0] = torch.eye(hc_mult, device=device)
        post[0].zero_()
    if num_tokens > 1:
        comb[1].zero_()
        post[1].fill_(1)
    expected = functions.mhc_post(x, residual, post, comb, implementation="torch")
    actual = functions.mhc_post(x, residual, post, comb, implementation=implementation)
    assert actual.is_contiguous()
    # BF16 rounding only: contraction of the FP32 mix may differ by one BF16
    # ulp, so compare within a ulp and pin the exact cases below.
    assert_close(actual, expected, torch.bfloat16, rtol=1e-2, atol=1.6e-2)
    if num_tokens:
        assert_close(actual[0], residual[0], rtol=0, atol=0)
    if num_tokens > 1:
        assert_close(actual[1], x[1].unsqueeze(0).expand(hc_mult, -1), rtol=0, atol=0)


@pytest.mark.api("functions.mhc_post")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_mhc_post_before_gather_is_bitwise_equal(accuracy_backend, num_tokens, tp_size):
    """Simulate sequence parallelism on one device, with padding and ranks
    holding no valid tokens: updating streams before or after the gather must
    be bitwise identical because every token is updated locally."""
    implementation, _, device = accuracy_backend
    torch.manual_seed(0)
    hc_mult, hidden_size = 4, 4096  # GLM-5.3-Flash spec.
    padded_tokens = num_tokens + (-num_tokens) % tp_size
    x = torch.randn(padded_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(padded_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    post = torch.rand(padded_tokens, hc_mult, 1, device=device)
    comb = torch.rand(padded_tokens, hc_mult, hc_mult, device=device)

    def gather(tensors):
        return torch.cat([t[:num_tokens] if i == 0 else t for i, t in enumerate(tensors)])[:num_tokens]

    x_shards = x.chunk(tp_size)
    residual_shards = residual.chunk(tp_size)
    post_shards = post.chunk(tp_size)
    comb_shards = comb.chunk(tp_size)
    before = functions.mhc_post(
        gather(x_shards),
        gather(residual_shards),
        gather(post_shards),
        gather(comb_shards),
        implementation=implementation,
    )
    after = torch.cat(
        [
            functions.mhc_post(x, res, post, comb, implementation=implementation)
            for x, res, post, comb in zip(x_shards, residual_shards, post_shards, comb_shards)
        ]
    )[:num_tokens]
    assert after.shape == (num_tokens, hc_mult, hidden_size)
    assert_close(after, before, rtol=0, atol=0)


@pytest.mark.api("functions.mhc_post")
@pytest.mark.accuracy
def test_mhc_post_accepts_2d_post_layer_mix(accuracy_backend):
    """The vLLM model passes [T, hc_mult, 1]; a [T, hc_mult] gate also works."""
    implementation, _, device = accuracy_backend
    torch.manual_seed(0)
    x = torch.randn(7, 4096, dtype=torch.bfloat16, device=device)
    residual = torch.randn(7, 4, 4096, dtype=torch.bfloat16, device=device)
    post = torch.rand(7, 4, device=device)
    comb = torch.rand(7, 4, 4, device=device)
    actual = functions.mhc_post(x, residual, post, comb, implementation=implementation)
    expected = functions.mhc_post(x, residual, post.unsqueeze(-1), comb, implementation="torch")
    assert_close(actual, expected, torch.bfloat16, rtol=1e-2, atol=1.6e-2)


@pytest.mark.api("functions.mhc_post")
@pytest.mark.accuracy
def test_mhc_post_custom_op_supports_compile(accuracy_backend):
    """The registered custom op keeps schema/fake metadata and graph capture."""
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu"):
        pytest.skip("NPU Triton custom-op compile contract")
    preload("mhc_post", implementation="triton")
    namespace = "mojo_npu_triton_a5" if target.startswith("npu.a5") else "mojo_npu_triton_a2"
    x = torch.randn(2, 4096, dtype=torch.bfloat16, device=device)
    residual = torch.randn(2, 4, 4096, dtype=torch.bfloat16, device=device)
    post = torch.rand(2, 4, 1, device=device)
    comb = torch.rand(2, 4, 4, device=device)
    torch.library.opcheck(
        getattr(torch.ops, namespace).mhc_post_fwd.default,
        (x, residual, post, comb),
        test_utils=("test_schema", "test_faketensor"),
    )

    def run(x, residual, post, comb):
        return functions.mhc_post(x, residual, post, comb, implementation="triton")

    try:
        eager = run(x, residual, post, comb).clone()
        compiled = compile_fullgraph(run)
        actual = compiled(x, residual, post, comb)
        assert_close(actual, eager, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
