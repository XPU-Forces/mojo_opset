# SPDX-License-Identifier: Apache-2.0
"""Precision suite for vision_pos_embed_interp on Ascend NPU.

NPU port of upstream tests/kernels/core/test_vit_bilinear_pos_embed.py
(vllm main@aa31fa0) plus the task-book additions:
- large video grid (16,100,168);
- generalised G=14 (already upstream) and tail-block hidden=1000;
- merged multi-image single-launch bitwise equality.

Tolerances and assertions follow the task book and MUST NOT be modified.

Run:  pytest tests/functions/test_vision_pos_embed_interp.py
"""

import os

import pytest
import torch

_DEVICE = "npu"
_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _NPU_AVAILABLE,
    reason="torch_npu / Ascend NPU device not available",
)

from mojo_opset.functions.vision_pos_embed_interp import (  # noqa: E402
    pos_embed_interpolate_native,
    triton_pos_embed_interpolate,
    triton_pos_embed_interpolate_merged,
)

DTYPES = [torch.float32, torch.bfloat16]
# Qwen3-VL default
NUM_GRID_PER_SIDE = 48
SPATIAL_MERGE_SIZE = 2
HIDDEN_DIM = 1152

# 4 square + 4 non-square grids (h, w divisible by spatial_merge_size=2)
SQUARE_GRIDS = [(1, 4, 4), (1, 16, 16), (1, 32, 32), (1, 48, 48)]
NON_SQUARE_GRIDS = [(1, 8, 16), (1, 14, 20), (1, 32, 48), (1, 60, 80)]
ALL_GRIDS = SQUARE_GRIDS + NON_SQUARE_GRIDS

LARGE_VIDEO_GRID = (16, 100, 168)
TAIL_HIDDEN = 1000
MULTI_IMAGE_LIST = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]


def _make_embed(G, hidden, dtype, device=_DEVICE):
    torch.manual_seed(42)
    return (
        torch.randn(G * G, hidden, device=device, dtype=dtype) * 0.25
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", ALL_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in ALL_GRIDS])
def test_triton_matches_native(grid_thw, dtype):
    """P01-P16: Triton kernel output must match the native implementation."""
    t, h, w = grid_thw
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype)

    native_out = pos_embed_interpolate_native(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    triton_out = triton_pos_embed_interpolate(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    assert native_out.shape == triton_out.shape, (
        f"Shape mismatch: native {native_out.shape} vs triton {triton_out.shape}"
    )

    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_temporal_repeat(dtype):
    """P17/P18: t>1 output must bitwise-equal t=1 repeated (atol=0)."""
    h, w = 16, 16
    t_single, t_multi = 1, 3
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype)

    out_single = triton_pos_embed_interpolate(
        embed_weight, t_single, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    out_multi = triton_pos_embed_interpolate(
        embed_weight, t_multi, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    expected = out_single.repeat(t_multi, 1)
    torch.testing.assert_close(out_multi, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_large_video_grid(dtype):
    """P19/P20: large video grid (16,100,168) must match native (no timeout)."""
    t, h, w = LARGE_VIDEO_GRID
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype)

    native_out = pos_embed_interpolate_native(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    triton_out = triton_pos_embed_interpolate(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_generalised_grid_side(dtype):
    """P21/P22: G=14 (non power of two) guards against hardcoded G=48."""
    t, h, w = 1, 16, 16
    G = 14
    embed_weight = _make_embed(G, HIDDEN_DIM, dtype)

    native_out = pos_embed_interpolate_native(
        embed_weight, t, h, w, G, SPATIAL_MERGE_SIZE, dtype
    )
    triton_out = triton_pos_embed_interpolate(
        embed_weight, t, h, w, G, SPATIAL_MERGE_SIZE, dtype
    )

    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 16, 16), (1, 32, 48)], ids=["16x16", "32x48"])
def test_tail_block_hidden(dtype, grid_thw):
    """P23-P26: non-power-of-two hidden dim tail block (mask, no OOB write)."""
    t, h, w = grid_thw
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, TAIL_HIDDEN, dtype)

    native_out = pos_embed_interpolate_native(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    triton_out = triton_pos_embed_interpolate(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    assert triton_out.shape == native_out.shape
    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_merged_multi_image_bitwise(dtype):
    """P27/P28: merged single launch == per-image launch + torch.cat (atol=0)."""
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype)

    per_image = torch.cat(
        [
            triton_pos_embed_interpolate(
                embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
            )
            for t, h, w in MULTI_IMAGE_LIST
        ],
        dim=0,
    )
    merged = triton_pos_embed_interpolate_merged(
        embed_weight, MULTI_IMAGE_LIST, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    assert merged.shape == per_image.shape
    torch.testing.assert_close(merged, per_image, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_merged_single_image_equivalence(dtype):
    """Merged API with a one-image list must equal the single-image API."""
    t, h, w = 3, 16, 16
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype)

    single = triton_pos_embed_interpolate(
        embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    merged = triton_pos_embed_interpolate_merged(
        embed_weight, [(t, h, w)], NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    torch.testing.assert_close(merged, single, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_merged_matches_native(dtype):
    """Merged path must also match the native benchmark within tolerance."""
    embed_weight = _make_embed(NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype)

    native_out = torch.cat(
        [
            pos_embed_interpolate_native(
                embed_weight, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
            )
            for t, h, w in MULTI_IMAGE_LIST
        ],
        dim=0,
    )
    merged = triton_pos_embed_interpolate_merged(
        embed_weight, MULTI_IMAGE_LIST, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )

    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(merged, native_out, atol=atol, rtol=rtol)
