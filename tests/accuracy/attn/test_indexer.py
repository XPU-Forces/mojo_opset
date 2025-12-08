import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLightningIndexer

def compare_result(triton_out: torch.Tensor, torch_out: torch.Tensor, topk = 2048):
    """
        比较两个topk索引输出结果,只需topk个索引元素集合相等(不考虑位置顺序）
    """
    triton_out = triton_out.cpu()
    torch_out = torch_out.cpu()
    if triton_out.shape != torch_out.shape:
        print(f"shape not equal: triton shape {triton_out.shape},torch shape: {torch_out.shape}")
        return False

    B, M, _ = triton_out.shape
    all_match = True

    for i in range(B):
        for j in range(M):
            pred_indices = triton_out[i, j]
            target_indices = torch_out[i, j]
            pred_sorted = torch.sort(pred_indices)[0]
            target_sorted = torch.sort(target_indices)[0]
            match = torch.equal(pred_sorted, target_sorted)
            if not match: 
                all_match = False

    if all_match:
        print(f"{B} batch、{M} seqlen topk={topk} index is same")
    else:
        print(f"{B} batch、{M} seqlen topk={topk} index is not same")

    return all_match

@pytest.mark.parametrize(
    "batch, q_seq_len, k_seq_len, q_head_num, k_head_num, head_dim, dummy_tensor",
    [
        (
            batch,
            q_seq_len,
            8192 if q_seq_len==1 else q_seq_len,  #
            q_head_num,
            1,
            128,
            torch.randn(1),
        )
        for batch in [1, 2, 8, 16,] #32, 128
        for q_seq_len in [1, 1024, 4096, 8192]
        for q_head_num in [128, 64]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_indexer(batch, q_seq_len, k_seq_len, q_head_num, k_head_num, head_dim, dummy_tensor):
    device = dummy_tensor.device
    query = torch.randn(
        batch,
        q_seq_len,
        q_head_num,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    query_scale = torch.randn(batch, q_seq_len, q_head_num, device=device, dtype=torch.float32)
    key = torch.randn(
        batch,
        k_seq_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    mask = torch.full((q_seq_len, q_seq_len), float("-inf"), device=device).triu_(1) if q_seq_len > 1 else None
    topk = 2048 if q_seq_len >= 4096 else q_seq_len // 2
    op = MojoLightningIndexer(top_k = topk)
    triton_topk_result,triton_index_score=op.forward_std(query, query_scale, key, None, mask)
    torch_topk_result,torch_index_score=op.forward_ref(query, query_scale, key, None, mask)
    torch.testing.assert_close(torch_index_score, triton_index_score, atol=1e-4, rtol=1e-4)
                           
    compare_result(triton_topk_result,torch_topk_result,topk)
      
    #atol, rtol = 1e-3, 1e-3
    #op.forward_diff(query, query_scale, key, None, mask,atol=atol, rtol=rtol)
