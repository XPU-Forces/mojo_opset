import pytest
import torch

from mojo_opset import MojoCompressor
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _make_minimal_inputs(
    *,
    device: str,
    dtype: torch.dtype,
    hidden_size: int = 1024,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    cmp_ratio: int = 128,
    coff: int = 1,
    block_size: int = 128,
    b: int = 1,
    s: int = 1,
):
    # Use a numerically stable input set to avoid NaNs and ensure deterministic behavior.
    # (The compressor kernel is sensitive to extreme/random state_cache values.)
    torch.manual_seed(0)
    scale = 0.01
    x = ((torch.rand(b * s, hidden_size, device=device, dtype=dtype) * 2.0 - 1.0) * scale).contiguous()
    wkv = ((torch.rand(coff * head_dim, hidden_size, device=device, dtype=dtype) * 2.0 - 1.0) * scale).contiguous()
    wgate = ((torch.rand(coff * head_dim, hidden_size, device=device, dtype=dtype) * 2.0 - 1.0) * scale).contiguous()
    ape = torch.zeros((cmp_ratio, coff * head_dim), device=device, dtype=torch.float32)
    norm_weight = torch.ones((head_dim,), device=device, dtype=dtype)
    # Identity-ish rotary: sin=0, cos=1.
    rope_sin = torch.zeros((1, rope_head_dim), device=device, dtype=dtype)
    rope_cos = torch.ones((1, rope_head_dim), device=device, dtype=dtype)

    # Minimal continuous-cache layout: one block per batch, matching doc example.
    block_table = torch.ones((b, 1), device=device, dtype=torch.int32)
    state_cache = torch.zeros((2, block_size, 2 * coff * head_dim), device=device, dtype=torch.float32)
    start_pos = torch.zeros((b,), device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, b * s], device=device, dtype=torch.int32)
    return (
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        block_table,
        cu_seqlens,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@bypass_not_implemented
def test_compressor_eager(dtype):
    device = get_torch_device()
    if device != "npu":
        pytest.skip("Ascend NPU only.")

    (
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        block_table,
        cu_seqlens,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
    ) = _make_minimal_inputs(device=device, dtype=dtype)

    before_cache = state_cache.clone()
    state_cache_ref = state_cache.clone()

    mojo_op = MojoCompressor()
    print(f"{type(mojo_op)=}")
    ref_op = MojoCompressor._registry.get("torch")()

    out = mojo_op.forward(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        state_block_table=block_table,
        cu_seqlens=cu_seqlens,
        seqused=None,
        start_pos=start_pos,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )

    ref_out = ref_op.forward(
        x,
        wkv,
        wgate,
        state_cache_ref,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        state_block_table=block_table,
        cu_seqlens=cu_seqlens,
        seqused=None,
        start_pos=start_pos,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )

    assert out.shape == ref_out.shape
    assert out.dtype == ref_out.dtype
    torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
    assert not torch.equal(before_cache, state_cache), "state_cache should be updated in-place."


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_compressor_graph(dtype):
    device = get_torch_device()
    if device != "npu":
        pytest.skip("Graph-mode test only runs on Ascend NPU.")

    try:
        import torchair  # noqa: F401
        from torchair.configs.compiler_config import CompilerConfig
    except Exception as e:
        pytest.skip(f"torchair is not available: {e}")

    try:
        import custom_ops  # noqa: F401
    except Exception as e:
        pytest.skip(f"custom_ops is not available: {e}")

    if not hasattr(torch.ops.custom, "compressor"):
        pytest.skip("torch.ops.custom.compressor is not registered in this environment.")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch build.")

    (
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        block_table,
        cu_seqlens,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
    ) = _make_minimal_inputs(device=device, dtype=dtype)

    before_cache = state_cache.clone()

    class Network(torch.nn.Module):
        def forward(
            self,
            x_in,
            wkv_in,
            wgate_in,
            state_cache_in,
            ape_in,
            norm_weight_in,
            rope_sin_in,
            rope_cos_in,
            block_table_in,
            cu_seqlens_in,
            start_pos_in,
        ):
            return torch.ops.custom.compressor(
                x_in,
                wkv_in,
                wgate_in,
                state_cache_in,
                ape_in,
                norm_weight_in,
                rope_sin_in,
                rope_cos_in,
                rope_head_dim=rope_head_dim,
                cmp_ratio=cmp_ratio,
                state_block_table=block_table_in,
                cu_seqlens=cu_seqlens_in,
                seqused=None,
                start_pos=start_pos_in,
                coff=coff,
                norm_eps=1e-6,
                rotary_mode=2,
                cache_mode=1,
            )

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    model = Network().to(device)
    try:
        model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    except TypeError:
        model = torch.compile(model, fullgraph=True, backend=npu_backend)
    except Exception as e:
        pytest.skip(f"torch.compile failed on this environment: {e}")

    out = model(x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos, block_table, cu_seqlens, start_pos)

    # Smoke validation: shape/dtype and cache mutated.
    assert out.dtype == dtype
    assert out.dim() == 2 and out.size(-1) == norm_weight.numel()
    assert not torch.equal(before_cache, state_cache), "state_cache should be updated in-place in graph mode."
