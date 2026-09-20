import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules
from tests._checks import assert_close


def make_varlen_fa_inputs(device, dtype, lengths, heads):
    qlens, klens = lengths
    q = torch.randn(sum(qlens), heads[0], 128, dtype=dtype, device=device)
    k = torch.randn(sum(klens), heads[1], 128, dtype=dtype, device=device)
    v = torch.randn_like(k)
    cq, ck = [torch.tensor([0, *ls], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32) for ls in lengths]
    return q, k, v, cq, ck


def make_varlen_fa_module_inputs(device):
    # Equal lengths exercise the module's optional cu_k_lens argument.
    return make_varlen_fa_inputs(device, torch.bfloat16, ([96, 128], [96, 128]), (5, 3))[:4]


SWA_INFER_CASES = [
    pytest.param(2, 16, 4, 128, 1024, 0, torch.bfloat16, id="bf16"),
    pytest.param(2, 16, 4, 96, 1024, 0, torch.bfloat16, id="padded-dim"),
    pytest.param(2, 8, 1, 128, 1024, 2048, torch.bfloat16, id="cached-kv"),
]


SWA_INFER_WINDOWS = [("ABAB", 4, 255), ("AABB", 4, 1023)]


def assert_swa_infer_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    # Original check_tol_diff compares in FP32 using these operator-specific limits.
    rtol, atol = (1e-6, 1e-5) if actual.dtype == torch.float32 else (2e-2, 2e-2)
    assert_close(actual.float(), expected.float(), torch.float32, rtol=rtol, atol=atol)


def make_swa_infer_case(batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, device):
    # generate_sdpa_data in Mojo's operator test creates lengths and tensors on CPU.
    q_lens = torch.randint(max_q // 2, max_q, (batch,), dtype=torch.int32).clamp(min=1)
    kv_lens = q_lens
    if max_cache:
        kv_lens = q_lens + torch.randint(max_cache // 2, max_cache, (batch,), dtype=torch.int32)
    cq, ck = (
        torch.cat((torch.zeros(1, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32))) for lens in (q_lens, kv_lens)
    )
    q = torch.randn(int(cq[-1]), q_heads, dim, dtype=dtype)
    k = torch.randn(int(ck[-1]), kv_heads, dim, dtype=dtype)
    v = torch.randn_like(k)
    return tuple(x.to(device) for x in (q, k, v, cq, ck))


@pytest.mark.api("modules.SWAInfer", ops=["swa_infer"])
@pytest.mark.parametrize("batch,q_heads,kv_heads,dim,max_q,max_cache,dtype", SWA_INFER_CASES)
@pytest.mark.parametrize("layout,global_window,local_window", SWA_INFER_WINDOWS)
@pytest.mark.accuracy
def test_swa_infer(
    accuracy_backend, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, layout, global_window, local_window
):
    implementation, _, device = accuracy_backend
    inputs = make_swa_infer_case(batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, device)
    module = modules.SWAInfer(True, layout, global_window, local_window, implementation=implementation)
    actual = module(*inputs, softmax_scale=dim**-0.5)
    expected = F.swa_infer(
        *inputs,
        local_window_size=local_window,
        global_window_size=global_window,
        gqa_interleave=layout == "ABAB",
        softmax_scale=dim**-0.5,
        implementation="torch_reference",
    )
    assert not actual.requires_grad
    assert_swa_infer_close(actual, expected)


@pytest.mark.api("modules.VarlenFAInfer", ops=["varlen_fa_infer"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.accuracy
def test_varlen_fa(accuracy_backend, causal, interleave):
    implementation, _, device = accuracy_backend
    inputs = make_varlen_fa_module_inputs(device)
    module = modules.VarlenFAInfer(causal, interleave, implementation=implementation)
    actual = module(*inputs)
    expected = F.varlen_fa_infer(*inputs, is_causal=causal, gqa_interleave=interleave, implementation="torch_reference")
    assert_close(actual, expected, inputs[0].dtype)
