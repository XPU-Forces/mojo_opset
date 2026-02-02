import functools
import math

import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoSdpa

torch.manual_seed(42)

def print_shape(expression):
    import inspect
    frame = inspect.currentframe().f_back
    tensor = eval(expression, frame.f_globals, frame.f_locals)
    print(f"{expression}.shape: {tensor.shape}")

def generate_paged_decode_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)

    seqlens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)

    max_num_blocks_per_seq = (seqlens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = seqlens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, seqlens, block_tables


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, seqlens, block_tables, atol, rtol",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            3e-2 if dtype != torch.float32 else 1e-5,
            1e-3 if dtype != torch.float32 else 1e-6,
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    atol: float,
    rtol: float,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    paged_decode_attn = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )
    paged_decode_attn_ref = MojoPagedDecodeGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    paged_decode_attn.forward_diff_with(
        paged_decode_attn_ref,
        query,
        k_cache,
        v_cache,
        seqlens,
        block_tables,
        softmax_scale=sm_scale,
        atol=atol,
        rtol=rtol,
    )


def generate_paged_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)

    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)])
    total_tokens = cu_seqlens_q[-1].item()

    query = torch.randn(total_tokens, num_q_heads, head_dim, dtype=dtype)
    k_unpadded = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_unpadded = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = (q_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(q_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = q_lens[i].item()
        start_loc = cu_seqlens_q[i].item()

        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

        k_seq = k_unpadded[start_loc : start_loc + seq_len]
        v_seq = v_unpadded[start_loc : start_loc + seq_len]
        for j in range(num_blocks_for_seq):
            physical_block_id = assigned_blocks[j]
            start_pos_in_seq = j * block_size
            tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

            k_slice = k_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)
            v_slice = v_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)

            k_cache[physical_block_id, :, :tokens_in_block, :] = k_slice
            v_cache[physical_block_id, :, :tokens_in_block, :] = v_slice

    return query, k_cache, v_cache, cu_seqlens_q, block_tables


test_configs = [
    (2, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, block_tables, atol, rtol",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            2e-2 if dtype != torch.float32 else 1e-5,
            2e-3 if dtype != torch.float32 else 1e-6,
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, BLK_S, dtype, ID in test_configs
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    atol: float,
    rtol: float,
    gqa_layout: str,
):
    paged_prefill_attn = MojoPagedPrefillGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    paged_prefill_attn_ref = MojoPagedPrefillGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    paged_prefill_attn.forward_diff_with(
        paged_prefill_attn_ref,
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        block_tables,
        softmax_scale=sm_scale,
        atol=atol,
        rtol=rtol,
    )

def generate_paged_prefill_attention_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_len: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device = None,
):
    """
    生成MojoPagedPrefillAttention的测试数据。
    
    注意：这里假设query和KV的序列长度可以不同，模拟cross-attention场景。
    """
    if device is None:
        device = torch.device("cpu")
    
    # 为每个batch生成query和KV的长度
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32, device=device)
    kv_lens = torch.randint(max_kv_len // 2, max_kv_len, (batch_size,), dtype=torch.int32, device=device)
    
    # 确保至少有一个token
    q_lens = torch.clamp(q_lens, min=1)
    kv_lens = torch.clamp(kv_lens, min=1)
    
    # 计算累积序列长度
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), 
                              torch.cumsum(q_lens, 0)])
    cu_seqlens_kv = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), 
                               torch.cumsum(kv_lens, 0)])
    
    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_kv[-1].item()
    
    # 生成query张量
    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    
    # 生成unpadded的KV张量（用于参考实现验证）
    k_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    # 计算每个序列需要的最大块数
    max_num_blocks_per_seq = max(
        (kv_lens.max().item() + block_size - 1) // block_size,
        (q_lens.max().item() + block_size - 1) // block_size
    )
    
    # 计算总共需要的块数
    total_blocks_needed = int(
        torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item()
    )
    
    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq
    
    # 创建一些额外的块
    num_total_blocks = total_blocks_needed + 10
    
    # 初始化KV缓存
    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, 
                         dtype=dtype, device=device)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, 
                         dtype=dtype, device=device)
    
    # 初始化块表
    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device) # npu only support int32
    
    # 生成可用的块ID
    free_blocks = torch.randperm(num_total_blocks, device=device)
    
    current_block_offset = 0
    for i in range(batch_size):
        seq_kv_len = kv_lens[i].item()
        seq_q_len = q_lens[i].item()
        kv_start_loc = cu_seqlens_kv[i].item()
        
        num_blocks_for_seq = (seq_kv_len + block_size - 1) // block_size
        
        # 分配块给这个序列
        assigned_blocks = free_blocks[current_block_offset:current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq
        
        # 将KV数据填充到缓存块中
        kv_seq = k_unpadded[kv_start_loc:kv_start_loc + seq_kv_len]
        v_seq = v_unpadded[kv_start_loc:kv_start_loc + seq_kv_len]
        
        for j in range(num_blocks_for_seq):
            physical_block_id = assigned_blocks[j]
            start_pos_in_seq = j * block_size
            tokens_in_block = min(block_size, seq_kv_len - start_pos_in_seq)
            
            if tokens_in_block <= 0:
                continue
            
            # 获取KV切片并调整维度
            k_slice = kv_seq[start_pos_in_seq:start_pos_in_seq + tokens_in_block].permute(1, 0, 2)
            v_slice = v_seq[start_pos_in_seq:start_pos_in_seq + tokens_in_block].permute(1, 0, 2)
            
            # 填充到缓存中
            k_cache[physical_block_id, :, :tokens_in_block, :] = k_slice
            v_cache[physical_block_id, :, :tokens_in_block, :] = v_slice
    
    return query, k_cache, v_cache, cu_seqlens_q, cu_seqlens_kv, block_tables


# 测试配置
test_configs_prefill_attention = [
    # (batch_size, num_q_heads, num_kv_heads, head_dim, max_q_len, max_kv_len, block_size, dtype, test_id)
    # (2, 16, 4, 128, 512, 512, 32, torch.bfloat16, "M_BF16_SAME_LEN"),
    (2, 16, 4, 128, 1024, 1024, 128, torch.bfloat16, "M_BF16_SAME_LEN"),
    # (2, 16, 4, 128, 512, 256, 32, torch.bfloat16, "M_BF16_KV_SHORTER"),
    # (2, 16, 4, 128, 256, 512, 32, torch.bfloat16, "M_BF16_Q_SHORTER"),
    # (2, 8, 8, 64, 256, 256, 16, torch.float32, "M_F32_SAME_HEAD"),
    # (2, 32, 8, 64, 128, 128, 32, torch.float16, "M_F16_GQA"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, cu_seqlens_kv, block_tables, atol, rtol",
    [
        pytest.param(
            *generate_paged_prefill_attention_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_len=KV_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            2e-2 if dtype != torch.float32 else 1e-5,
            2e-3 if dtype != torch.float32 else 1e-6,
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_LEN, BLK_S, dtype, ID in test_configs_prefill_attention
    ],
)
# @pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_causal", [False,])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    atol: float,
    rtol: float,
    is_causal: bool,
):
    """
    测试MojoPagedPrefillAttention算子。
    注意：window_size参数使用默认值-1（全注意力）
    """
    # 创建算子实例 - 使用默认window_size=-1
    paged_prefill_attn = MojoPagedPrefillAttention(
        is_causal=is_causal,
        window_size=-1,  # 全注意力
    )
    
    # 创建参考实现（使用torch后端）
    paged_prefill_attn_ref = MojoPagedPrefillAttention._registry.get("torch")(
        is_causal=is_causal,
        window_size=-1,  # 全注意力
    )
    
    # 计算softmax scale
    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    out_ref = paged_prefill_attn_ref.forward(query, k_cache, v_cache, cu_seqlens_q, cu_seqlens_kv,
                                             block_tables, sm_scale)
    print_shape("query")
    print_shape("k_cache")
    print_shape("v_cache")
    print_shape("cu_seqlens_q")
    print("cu_seqlens_q: ", cu_seqlens_q)
    print_shape("cu_seqlens_kv")
    print("cu_seqlens_kv: ", cu_seqlens_kv)
    print_shape("block_tables")
    print("block_table.dtype: ", block_tables.dtype)
    print_shape("out_ref")
    print("out_ref: ", out_ref)
    import torch_npu
    actual_seq_qlen = cu_seqlens_q[1:]
    actual_seq_kvlen = cu_seqlens_kv[1:]
    _, num_query_heads, _ = query.shape
    _, num_key_value_heads, block_size, _ = k_cache.shape
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(query, k_cache, v_cache, 
      actual_seq_qlen = actual_seq_qlen, actual_seq_kvlen = actual_seq_kvlen, block_table=block_tables,
      num_query_heads = num_query_heads, num_key_value_heads=num_key_value_heads, softmax_scale = sm_scale,
      input_layout="TND", block_size=block_size)
    print("out.shape: ", out.shape)
    print("out: ", out)

    # 执行测试
    # paged_prefill_attn.forward_diff_with(
    #     paged_prefill_attn_ref,
    #     query,
    #     k_cache,
    #     v_cache,
    #     cu_seqlens_q,
    #     cu_seqlens_kv,
    #     block_tables,
    #     softmax_scale=sm_scale,
    #     atol=atol,
    #     rtol=rtol,
    # )

@functools.lru_cache()
def generate_diffusion_attention_mask(
    seq_length: int,
    block_size: int,
) -> torch.Tensor:
    total_length = seq_length * 2
    attn_mask = torch.zeros(total_length, total_length, dtype=torch.int8)

    for i in range(total_length):
        for j in range(total_length):
            block_i = i // block_size
            block_j = j // block_size
            if block_i == block_j:
                attn_mask[i, j] = 1

            if j >= seq_length and i < seq_length and ((j - seq_length) // block_size) < block_i:
                attn_mask[i, j] = 1

            if i >= seq_length and j >= seq_length and block_j < block_i:
                attn_mask[i, j] = 1

    return attn_mask.to(torch.bool)


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    blockwise_diffusion_attn_mask = generate_diffusion_attention_mask(seq_length, block_size)
    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num


@pytest.mark.parametrize(
    "query, key, value, blockwise_diffusion_attn_mask, enable_gqa",
    [
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=5,
                kv_head_num=1,
                head_dim=128,
                seq_length=2048,
                block_size=32,
            )
        ),
    ],
)
@auto_switch_platform()
def test_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blockwise_diffusion_attn_mask: torch.Tensor,
    enable_gqa: bool,
):
    diffusion_attn_ref = MojoSdpa._registry.get("torch")(
        mask=blockwise_diffusion_attn_mask, scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn = MojoSdpa(
        mask=blockwise_diffusion_attn_mask, scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn_ref.forward_diff_with(diffusion_attn, query, key, value)
