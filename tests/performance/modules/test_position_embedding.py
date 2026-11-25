import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.RoPEInfer", ops=["rope_infer"])
@pytest.mark.parametrize(
    "batch,seq_len,q_heads,k_heads,dtype,layout",
    [
        (1, 128, 8, 2, torch.bfloat16, "contiguous"),
        (32, 8192, 32, 8, torch.bfloat16, "contiguous"),
        *[
            pytest.param(
                32, 8192, qh, kh, dtype, "transposed",
                marks=pytest.mark.api("modules.RoPECosSin", ops=["rope_cos_sin"]),
            )
            for qh, kh in [(32, 32), (32, 8), (16, 1), (1, 1)]
            for dtype in [torch.float32, torch.float16, torch.bfloat16]
        ],
    ],
)
def test_apply_rope(benchmark, perf_environment, batch, seq_len, q_heads, k_heads, dtype, layout):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.RoPEInfer(implementation=implementation)
        if layout == "transposed":
            embedding = modules.RoPECosSin(
                rope_theta=10000.0, rope_dim=128, init_max_length=seq_len, implementation="torch"
            ).to(device)
            x = torch.randn(batch, seq_len, q_heads * 128, device=device, dtype=dtype)
            cos, sin = embedding(x)
            q, k = [
                torch.randn(batch, seq_len, heads, 128, device=device, dtype=dtype).transpose(1, 2)
                for heads in (q_heads, k_heads)
            ]
        else:
            q, k = [torch.randn(batch, heads, seq_len, 128, device=device, dtype=dtype) for heads in (q_heads, k_heads)]
            cos = torch.randn(seq_len, 128, device=device)
            sin = torch.randn_like(cos)
        return lambda: module(q, k, cos, sin)

    benchmark(
        factory=factory,
        op="rope_infer",
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        k_heads=k_heads,
        dtype=str(dtype),
        layout=layout,
        phase="forward",
    )


@pytest.mark.api(
    "modules.VisionRoPE2DInfer",
    "modules.VisionRoPECosSin2D",
    ops=["vision_rope_2d_infer", "vision_rope_cos_sin_2d"],
)
@pytest.mark.parametrize("grid", [((4, 4),), ((8, 6),), ((8, 8), (4, 6))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vision_rope(benchmark, perf_environment, grid, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.VisionRoPE2DInfer(implementation=implementation)
        embedding = modules.VisionRoPECosSin2D(
            rope_theta=10000.0, rope_dim=64, adapooling_factor=2, implementation="torch"
        ).to(device)
        cos, sin = embedding(torch.tensor(grid, dtype=torch.int32, device=device))
        q, k = [torch.randn(sum(h * w for h, w in grid), 20, 64, dtype=dtype, device=device) for _ in range(2)]
        return lambda: module(q, k, cos, sin)

    benchmark(
        factory=factory,
        op="vision_rope_2d_infer",
        grid=grid,
        heads=20,
        head_dim=64,
        dtype=str(dtype),
        phase="forward",
    )


@pytest.mark.api("modules.MultimodalRoPEInfer", ops=["multimodal_rope_infer"])
@pytest.mark.parametrize(
    "tokens,q_heads,k_heads,section,interleaved",
    [
        (128, 28, 4, [16, 24, 24], False),
        (128, 32, 8, [24, 20, 20], True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multimodal_rope_infer(benchmark, perf_environment, tokens, q_heads, k_heads, section, interleaved, dtype):
    _, device, _, implementation = perf_environment
    dim = 128

    def factory():
        module = modules.MultimodalRoPEInfer(implementation=implementation)
        rotary_dim = sum(section) * 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32) / rotary_dim))
        freqs = torch.outer(torch.arange(2048, dtype=torch.float32), inv_freq).repeat_interleave(2, -1)
        cache_cos, cache_sin = freqs.cos().to(device), freqs.sin().to(device)
        positions = torch.randint(1000, (3, tokens), device=device)
        cos, sin = cache_cos[positions], cache_sin[positions]
        q = torch.randn(tokens, q_heads * dim, device=device, dtype=dtype)
        k = torch.randn(tokens, k_heads * dim, device=device, dtype=dtype)
        return lambda: module(q, k, cos, sin, mrope_section=section, is_interleaved=interleaved, head_dim=dim)

    benchmark(
        factory=factory,
        op="multimodal_rope_infer",
        tokens=tokens,
        q_heads=q_heads,
        k_heads=k_heads,
        head_dim=dim,
        section=section,
        interleaved=interleaved,
        dtype=str(dtype),
        phase="forward",
    )

