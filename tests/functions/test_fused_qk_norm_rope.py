from functools import partial

import pytest
import torch
import torch.nn.functional as F

from mojo_opset import functions
from tests._checks import assert_repeatable

HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-5
ATOL = 0.0625
RTOL = 0.02


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    *,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = rope_table.shape[-1]
    half = rotary_dim // 2

    def rotate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.shape[-1],), weight, EPS)
        compute_dtype = torch.float32 if interleaved else x.dtype
        values = x.to(compute_dtype)
        cos = rope_table[..., :half].to(compute_dtype).unsqueeze(1)
        sin = rope_table[..., half:].to(compute_dtype).unsqueeze(1)
        if interleaved:
            first = values[..., :rotary_dim:2]
            second = values[..., 1:rotary_dim:2]
        else:
            first = values[..., :half]
            second = values[..., half:rotary_dim]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                values[..., rotary_dim:],
            ),
            dim=-1,
        ).to(x.dtype)

    return rotate(q, q_weight), rotate(k, k_weight)


def _kernel_rounding_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    *,
    round_normalized: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = ROTARY_DIM // 2

    def rotate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if round_normalized:
            normalized = F.rms_norm(x, (HEAD_DIM,), weight, EPS)
            first = normalized[..., :half].float()
            second = normalized[..., half:ROTARY_DIM].float()
            passthrough = normalized[..., ROTARY_DIM:].float()
        else:
            values = x.float()
            inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPS)
            normalized = values * inv_rms * weight.float()
            first = normalized[..., :half]
            second = normalized[..., half:ROTARY_DIM]
            passthrough = normalized[..., ROTARY_DIM:]
        cos = rope_table[..., :half].float().unsqueeze(1)
        sin = rope_table[..., half:].float().unsqueeze(1)
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                passthrough,
            ),
            dim=-1,
        ).to(x.dtype)

    return rotate(q, q_weight), rotate(k, k_weight)


def _make_inputs(
    device: str,
    tokens: int,
    heads: int,
    *,
    strided: bool,
    table_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if strided:
        packed = torch.randn(tokens, heads * 3, HEAD_DIM, device=device, dtype=torch.bfloat16)
        q = packed[:, :heads]
        k = packed[:, heads : heads * 2]
    else:
        q = torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
        k = torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    angles = torch.randn(tokens, ROTARY_DIM // 2, device=device)
    rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(table_dtype)
    return q, k, q_weight, k_weight, rope_table


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.accuracy
@pytest.mark.parametrize("tokens", [1, 257, 1024, 4096, 16384, 65536])
@pytest.mark.parametrize("heads", [8, 56])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "packed-qkv-view"])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32], ids=["bf16-table", "fp32-table"])
def test_fused_qk_norm_rope_accuracy(accuracy_backend, tokens, heads, strided, table_dtype):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("the fused Triton implementation targets Ascend NPU")
    q, k, q_weight, k_weight, rope_table = _make_inputs(
        device,
        tokens,
        heads,
        strided=strided,
        table_dtype=table_dtype,
    )
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        implementation=implementation,
    )
    assert actual_q.shape == q.shape and actual_k.shape == k.shape
    assert actual_q.is_contiguous() and actual_k.is_contiguous()
    torch.testing.assert_close(actual_q, expected_q, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual_k, expected_k, atol=ATOL, rtol=RTOL)


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.accuracy
@pytest.mark.parametrize("tokens", [257, 4096])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "packed-qkv-view"])
def test_fused_qk_norm_rope_tail_heads(accuracy_backend, tokens, strided):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("the fused Triton implementation targets Ascend NPU")
    q, k, q_weight, k_weight, rope_table = _make_inputs(
        device,
        tokens,
        57,
        strided=strided,
        table_dtype=torch.bfloat16,
    )
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        implementation=implementation,
    )
    torch.testing.assert_close(actual_q, expected_q, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual_k, expected_k, atol=ATOL, rtol=RTOL)


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.bitwise
def test_fused_qk_norm_rope_nonrotary_passthrough_is_bf16_exact(accuracy_backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("the fused Triton implementation targets Ascend NPU")
    q, k, q_weight, k_weight, rope_table = _make_inputs(
        device,
        257,
        56,
        strided=True,
        table_dtype=torch.bfloat16,
    )
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        implementation=implementation,
    )
    expected_q = F.rms_norm(q, (HEAD_DIM,), q_weight, EPS)
    expected_k = F.rms_norm(k, (HEAD_DIM,), k_weight, EPS)
    torch.testing.assert_close(actual_q[..., ROTARY_DIM:], expected_q[..., ROTARY_DIM:], atol=0, rtol=0)
    torch.testing.assert_close(actual_k[..., ROTARY_DIM:], expected_k[..., ROTARY_DIM:], atol=0, rtol=0)


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.bitwise
def test_fused_qk_norm_rope_bf16_round_and_shared_stats_are_exact(accuracy_backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("the fused Triton implementation targets Ascend NPU")
    q, k, q_weight, k_weight, rope_table = _make_inputs(
        device,
        17,
        57,
        strided=True,
        table_dtype=torch.float32,
    )
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        implementation=implementation,
    )
    expected_q, expected_k = _kernel_rounding_reference(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        round_normalized=True,
    )
    unrounded_q, unrounded_k = _kernel_rounding_reference(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        round_normalized=False,
    )
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert not torch.equal(actual_q, unrounded_q)
    assert not torch.equal(actual_k, unrounded_k)


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.accuracy
@pytest.mark.parametrize("tokens", [1, 257])
def test_fused_qk_norm_rope_is_deterministic(accuracy_backend, tokens):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("the fused Triton implementation targets Ascend NPU")
    inputs = _make_inputs(
        device,
        tokens,
        57,
        strided=True,
        table_dtype=torch.float32,
    )
    assert_repeatable(
        partial(
            functions.fused_qk_norm_rope,
            eps=EPS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            implementation=implementation,
        ),
        inputs,
        backward=False,
    )


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "shape,strided",
    [
        ((64, 64), False),
        ((64, 64), True),
        ((128, 128), False),
        ((128, 128), True),
        ((256, 128), False),
        ((256, 128), True),
    ],
)
def test_fused_qk_norm_rope_interleaved_fallback(accuracy_backend, shape, strided):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation not in (None, "triton"):
        pytest.skip("Ascend fallback coverage")
    head_dim, rotary_dim = shape
    tokens, heads = 31, 8
    if strided:
        packed = torch.randn(tokens, heads * 3, head_dim, device=device, dtype=torch.bfloat16)
        q = packed[:, :heads]
        k = packed[:, heads : heads * 2]
    else:
        q = torch.randn(tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    angles = torch.randn(tokens, rotary_dim // 2, device=device)
    rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    expected_q, expected_k = _reference(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        interleaved=True,
    )
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        interleaved=True,
        implementation=implementation,
    )
    torch.testing.assert_close(actual_q, expected_q, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual_k, expected_k, atol=ATOL, rtol=RTOL)


@pytest.mark.reference
def test_fused_qk_norm_rope_torch_fallback_cpu():
    q, k, q_weight, k_weight, rope_table = _make_inputs(
        "cpu",
        17,
        8,
        strided=True,
        table_dtype=torch.float32,
    )
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        implementation="torch",
    )
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)


@pytest.mark.reference
def test_fused_qk_norm_rope_empty_tokens_cpu():
    q, k, q_weight, k_weight, rope_table = _make_inputs(
        "cpu",
        0,
        8,
        strided=False,
        table_dtype=torch.bfloat16,
    )
    actual_q, actual_k = functions.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        EPS,
        implementation="torch",
    )
    assert actual_q.shape == q.shape and actual_k.shape == k.shape
