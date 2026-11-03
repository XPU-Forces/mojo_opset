"""模块层契约测试：``modules.VisionPosEmbedInterp``（任务书 :141 的 modules/ 路径）。

定位：与 ``tests/functions/test_pos_embed_interp.py`` 分工——
functions 层管数值正确性，本文件只管**封装契约**：
  1. 静态参数（G / m_size / dtype）持有并正确转发；
  2. 单图 ``forward`` 与多图 ``forward_multi`` 两条转发路径；
  3. 构造参数校验（G ≥ 2、m_size ≥ 1）；
  4. 与 functions 层同 implementation 调用结果逐位一致。
"""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as MojoModules

NUM_GRID_PER_SIDE = 48
M_SIZE = 2
HIDDEN_DIM = 1152
DTYPES = (torch.float32, torch.bfloat16)


def _embed(dtype):
    torch.manual_seed(42)
    return (
        torch.randn(NUM_GRID_PER_SIDE * NUM_GRID_PER_SIDE, HIDDEN_DIM, dtype=dtype)
        * 0.25
    ).contiguous()


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 16, 16), (1, 32, 48)], ids=str)
def test_forward_matches_function(grid_thw, dtype):
    """模块单图转发必须与 functions 层逐位一致。"""
    embed = _embed(dtype)
    module = MojoModules.VisionPosEmbedInterp(
        NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation="torch"
    )
    got = module(embed, *grid_thw)
    want = F.vision_pos_embed_interp(
        embed, *grid_thw, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation="torch"
    )
    assert torch.equal(got, want)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_forward_multi_matches_cat(dtype):
    """模块多图转发必须等于逐图调用 + cat。"""
    embed = _embed(dtype)
    module = MojoModules.VisionPosEmbedInterp(
        NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation="torch"
    )
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    got = module.forward_multi(embed, grids)
    want = torch.cat([module(embed, *g) for g in grids])
    assert torch.equal(got, want)


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_holds_config_as_members():
    """三个静态参数作为成员持有，而非每次重传。"""
    module = MojoModules.VisionPosEmbedInterp(
        NUM_GRID_PER_SIDE, M_SIZE, torch.bfloat16, implementation="torch"
    )
    assert module.num_grid_per_side == NUM_GRID_PER_SIDE
    assert module.m_size == M_SIZE
    assert module.dtype is torch.bfloat16
    assert module.implementation == "torch"


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_rejects_invalid_config():
    with pytest.raises(ValueError):
        MojoModules.VisionPosEmbedInterp(1, M_SIZE)
    with pytest.raises(ValueError):
        MojoModules.VisionPosEmbedInterp(NUM_GRID_PER_SIDE, 0)


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_rejects_non_divisible_grid():
    """h/w 不被 m_size 整除时，functions 层抛 ValueError。"""
    embed = _embed(torch.bfloat16)
    module = MojoModules.VisionPosEmbedInterp(
        NUM_GRID_PER_SIDE, M_SIZE, torch.bfloat16, implementation="torch"
    )
    with pytest.raises(ValueError):
        module(embed, 1, 15, 16)
