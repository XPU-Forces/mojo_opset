"""Test script for MojoSparseAttnSharedkv and MojoSparseAttnSharedkvMetadata"""
import os
import torch
import numpy as np
import math
import pytest

os.environ['MOJO_PLATFORM'] = 'npu'

from mojo_opset import MojoSparseAttnSharedkv, MojoSparseAttnSharedkvMetadata

def test_mojo_sparse_attn_sharedkv_metadata():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        pytest.skip("NPU not available!")

    torch.npu.set_device(13)

    b = 1
    s1 = 128
    s2 = 512
    n1 = 64
    n2 = 1
    dn = 512
    ori_block_size = 128

    cu_seqlens_q = torch.tensor([0, s1], dtype=torch.int32).npu()
    seqused_kv = torch.tensor([s2], dtype=torch.int32).npu()

    metadata_op = MojoSparseAttnSharedkvMetadata()
    metadata = metadata_op.forward(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=0,
        cmp_ratio=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q='TND',
        layout_kv='PA_ND',
        has_ori_kv=True,
        has_cmp_kv=False,
        device='npu:13'
    )

    assert metadata is not None

def test_mojo_sparse_attn_sharedkv():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        pytest.skip("NPU not available!")

    torch.npu.set_device(13)

    b = 1
    s1 = 128
    s2 = 512
    n1 = 64
    n2 = 1
    dn = 512
    ori_block_size = 128

    cu_seqlens_q = torch.tensor([0, s1], dtype=torch.int32).npu()
    seqused_kv = torch.tensor([s2], dtype=torch.int32).npu()

    metadata_op = MojoSparseAttnSharedkvMetadata()
    metadata = metadata_op.forward(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=0,
        cmp_ratio=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q='TND',
        layout_kv='PA_ND',
        has_ori_kv=True,
        has_cmp_kv=False,
        device='npu:13'
    )

    sparse_op = MojoSparseAttnSharedkv()
    q = torch.randn(b*s1, n1, dn, dtype=torch.bfloat16).npu()
    
    ori_block_num = math.ceil(s2/ori_block_size)
    ori_block_table = torch.arange(ori_block_num, dtype=torch.int32).reshape(1, -1).npu()
    ori_kv = torch.randn(ori_block_num, ori_block_size, n2, dn, dtype=torch.bfloat16).npu()
    sinks = torch.randn(n1, dtype=torch.float32).npu()

    result = sparse_op.forward(
        q,
        ori_kv=ori_kv,
        cmp_kv=None,
        ori_sparse_indices=None,
        cmp_sparse_indices=None,
        ori_block_table=ori_block_table,
        cmp_block_table=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=0.0,
        cmp_ratio=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q='TND',
        layout_kv='PA_ND',
        return_softmax_lse=False
    )

    assert result is not None