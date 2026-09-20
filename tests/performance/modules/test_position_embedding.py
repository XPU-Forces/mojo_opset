import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.ApplyRoPEInfer", ops=["apply_rope_infer"])
@pytest.mark.parametrize(
    "batch,seq_len,q_heads,k_heads,dtype,layout",
    [
        (1, 128, 8, 2, torch.bfloat16, "contiguous"),
        (32, 8192, 32, 8, torch.bfloat16, "contiguous"),
        *[
            pytest.param(
                32, 8192, qh, kh, dtype, "transposed",
                marks=pytest.mark.api("modules.RotaryEmbedding", ops=["rotary_embedding"]),
            )
            for qh, kh in [(32, 32), (32, 8), (16, 1), (1, 1)]
            for dtype in [torch.float32, torch.float16, torch.bfloat16]
        ],
    ],
)
def test_apply_rope(benchmark, perf_environment, batch, seq_len, q_heads, k_heads, dtype, layout):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.ApplyRoPEInfer(implementation=implementation)
        if layout == "transposed":
            embedding = modules.RotaryEmbedding(
                rope_theta=10000.0, rope_dim=128, init_max_length=seq_len, implementation="torch_reference"
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
        op="apply_rope_infer",
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        k_heads=k_heads,
        dtype=str(dtype),
        layout=layout,
        phase="forward",
    )


@pytest.mark.api(
    "modules.ApplyVisionRoPE2DInfer",
    "modules.VisionRotaryEmbedding2D",
    ops=["apply_vision_rope2d_infer", "vision_rotary_embedding2d"],
)
@pytest.mark.parametrize("grid", [((4, 4),), ((8, 6),), ((8, 8), (4, 6))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vision_rope(benchmark, perf_environment, grid, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.ApplyVisionRoPE2DInfer(implementation=implementation)
        embedding = modules.VisionRotaryEmbedding2D(
            rope_theta=10000.0, rope_dim=64, adapooling_factor=2, implementation="torch_reference"
        ).to(device)
        cos, sin = embedding(torch.tensor(grid, dtype=torch.int32, device=device))
        q, k = [torch.randn(sum(h * w for h, w in grid), 20, 64, dtype=dtype, device=device) for _ in range(2)]
        return lambda: module(q, k, cos, sin)

    benchmark(
        factory=factory,
        op="apply_vision_rope2d_infer",
        grid=grid,
        heads=20,
        head_dim=64,
        dtype=str(dtype),
        phase="forward",
    )
