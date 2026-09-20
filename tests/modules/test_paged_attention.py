from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close

test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 1024, torch.bfloat16, "M_BF16_LONG"),
    (8, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (8, 8, 1, 128, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ"),
]


test_configs_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 4096, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 1, 128, 0, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ"),
]


test_configs_swa_decode = [
    (4, -1, 16, 4, 128, 1024, 512, torch.bfloat16, "M_BF16"),
    (8, -1, 16, 4, 96, 2048, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, -1, 8, 1, 128, 4096, 128, torch.bfloat16, "M_BF16_LONG"),
    (2, -1, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, -1, 8, 1, 128, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ"),
    (2, -1, 8, 2, 128, 2048, 1024, torch.bfloat16, "M_BF16_GROUP1"),
    (2, -1, 24, 8, 128, 2048, 1024, torch.bfloat16, "M_BF16_GROUP2"),
]


test_configs_swa_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 128, 2048, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 256, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 1, 128, 0, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ"),
    (2, 8, 2, 128, 2048, 0, 1024, torch.bfloat16, "M_BF16_GROUP1"),
    (2, 24, 8, 128, 1024, 1024, 1024, torch.bfloat16, "M_BF16_GROUP2"),
]


PAGED_CASES = (
    [
        ("paged_prefill_gqa_infer", "PagedPrefillGQAInfer", case, layout, None, None)
        for case in test_configs_prefill + [(2, 16, 4, 128, 256, 0, 128, torch.bfloat16, "NPU_SUPPORTED")]
        for layout in ("ABAB", "AABB")
    ]
    + [
        ("paged_decode_gqa_infer", "PagedDecodeGQAInfer", case, layout, None, None)
        for case in test_configs_decode + [(4, 16, 4, 128, 1024, 128, torch.bfloat16, "NPU_SUPPORTED")]
        for layout in ("ABAB", "AABB")
    ]
    + [
        ("paged_prefill_swa_infer", "PagedPrefillSWAInfer", case, layout, 4, window)
        for case in test_configs_swa_prefill
        for layout, window in (("ABAB", 255), ("AABB", 1023))
    ]
    + [
        ("paged_decode_swa_infer", "PagedDecodeSWAInfer", case, layout, 4, window)
        for case in test_configs_swa_decode
        for layout, window in (("ABAB", 255), ("AABB", 1023))
    ]
)


PREFILL_CASES = [(2, 16, 4, 128, 64), (1, 8, 1, 64, 128)]


def paged_bucket_case(layout, device, impl):
    """Original varlen bucket: 4 real tokens in an 8-token, 6-batch allocation."""
    if impl not in (None, "triton", "torch_reference"):
        pytest.skip("Original 16-token page bucket case has no optimized torch_npu implementation")
    query = torch.randn(8, 4, 128, dtype=torch.bfloat16, device=device)
    cu_q_lens = torch.tensor([0, 1, 2, 3, 4, 4, 4], dtype=torch.int32, device=device)
    key_cache = torch.zeros(6, 2, 16, 128, dtype=query.dtype, device=device)
    value_cache = torch.zeros_like(key_cache)
    key_cache[:4, :, 0] = torch.randn(4, 2, 128, dtype=query.dtype, device=device)
    value_cache[:4, :, 0] = torch.randn(4, 2, 128, dtype=query.dtype, device=device)
    block_tables = torch.full((6, 1), -1, dtype=torch.int32, device=device)
    block_tables[:4, 0] = torch.arange(4, dtype=torch.int32, device=device)
    inputs = (query, key_cache, value_cache, cu_q_lens, block_tables)
    options = dict(cu_total_seq_lens=cu_q_lens.clone(), softmax_scale=128 ** (-0.5), max_q_len=1, max_total_seq_len=1)
    op = M.PagedPrefillGQAInfer(gqa_layout=layout, implementation=impl)
    op.prepare_metadata(cu_q_lens, options["cu_total_seq_lens"], 4, 2, 16)
    reference = partial(F.paged_prefill_gqa_infer, gqa_layout=layout, implementation="torch_reference", **options)
    return (partial(op, **options), reference, inputs)


def generate_paged_decode_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
    seq_len: int = -1,
):
    if seq_len != -1:
        query = torch.randn(batch_size, seq_len, num_q_heads, head_dim, dtype=dtype)
    else:
        query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)
    min_seq_len = seq_len if seq_len != -1 else 1
    if max_seq_len > 0:
        total_seq_lens = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32)
        total_seq_lens = torch.clamp(total_seq_lens, min=min_seq_len)
    else:
        total_seq_lens = torch.randperm(batch_size, dtype=torch.int32)
        total_seq_lens = torch.where(total_seq_lens == 0, 0, total_seq_lens + min_seq_len - 1)
    max_total_seq_len = total_seq_lens.max().item()
    max_num_blocks_per_seq = (max_total_seq_len + block_size - 1) // block_size
    total_blocks_needed = int(
        torch.div(total_seq_lens + block_size - 1, block_size, rounding_mode="floor").sum().item()
    )
    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq
    num_total_blocks = total_blocks_needed + 10
    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)
    current_block_offset = 0
    for i in range(batch_size):
        seq_len = total_seq_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq
    return (query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len)


def generate_paged_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    if max_q_len > 0:
        q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
        q_lens = torch.clamp(q_lens, min=1)
    else:
        q_lens = torch.randperm(batch_size, dtype=torch.int32)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])
    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
        kv_lens = torch.where(q_lens > 0, kv_lens, torch.zeros_like(kv_lens))
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)])
    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()
    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    k_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    max_num_blocks_per_seq = (kv_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())
    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq
    num_total_blocks = total_blocks_needed + 10
    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)
    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        start_loc = cu_total_seq_lens[i].item()
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
    cu_total_seq_lens = None if kv_cache_lens is None else cu_total_seq_lens
    max_q_len = int((cu_q_lens[1:] - cu_q_lens[:-1]).max().item()) if cu_q_lens.numel() > 1 else 0
    max_total_seq_len = int(kv_lens.max().item()) if kv_lens.numel() > 0 else 0
    return (query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len)


def move_tensors(values, device):
    return tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in values)


def paged_case(spec, device, impl):
    name, module_name, case, layout, global_window, local_window = spec
    prefill, swa = ("prefill" in name, "swa" in name)
    dims = case[:-1]
    if prefill:
        batch, qheads, kheads, dim, qmax, prefix, page, dtype = dims
    elif swa:
        batch, steps, qheads, kheads, dim, kmax, page, dtype = dims
    else:
        batch, qheads, kheads, dim, kmax, page, dtype = dims
    if impl == "torch_npu":
        if dim % 128:
            pytest.skip("Original torch_npu fused attention requires head_dim divisible by 128")
        if not swa and (page % 128 or page > 512):
            pytest.skip("Original torch_npu paged GQA requires page_size 128..512 aligned to 128")
        if not swa and prefill and prefix:
            pytest.skip("Original torch_npu TND paged prefill does not support cached prefixes")
    config = dict(gqa_layout=layout)
    if swa:
        config.update(global_window_size=global_window, local_window_size=local_window)
    if prefill:
        data = move_tensors(generate_paged_prefill_data(*dims), device)
        inputs = data[:5]
        kwargs = dict(cu_total_seq_lens=data[5], max_q_len=data[6], max_total_seq_len=data[7])
    else:
        data = move_tensors(generate_paged_decode_data(batch, qheads, kheads, dim, kmax, page, dtype), device)
        inputs = data[:5]
        kwargs = dict(max_total_seq_len=data[5])
    op = getattr(M, module_name)(**config, implementation=impl)
    if name == "paged_prefill_gqa_infer":
        op.prepare_metadata(inputs[3], kwargs["cu_total_seq_lens"], qheads, kheads, page)
    reference = partial(getattr(F, name), **config, implementation="torch_reference", **kwargs)
    return (partial(op, **kwargs), reference, inputs, dtype)


def prefill_case(case, layout, device, impl):
    batch, qheads, kheads, dim, length = case
    if impl != "torch_reference" and dim % 128:
        pytest.skip("Original torch_npu fused attention requires head_dim divisible by 128")
    inputs = (
        torch.randn(batch, qheads, length, dim, device=device, dtype=torch.bfloat16),
        torch.randn(batch, kheads, length, dim, device=device, dtype=torch.bfloat16),
        torch.randn(batch, kheads, length, dim, device=device, dtype=torch.bfloat16),
        torch.arange(batch + 1, device=device, dtype=torch.int32) * length,
    )
    op = M.PrefillGQAInfer(gqa_layout=layout, implementation=impl)
    reference = partial(F.prefill_gqa_infer, gqa_layout=layout, implementation="torch_reference")
    return (op, reference, inputs)


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "spec",
    [pytest.param(spec, marks=pytest.mark.api("modules." + spec[1], ops=[spec[0]])) for spec in PAGED_CASES],
    ids=[f"{x[0]}-{x[2][-1]}-{x[3]}" for x in PAGED_CASES],
)
def test_paged(accuracy_backend, spec):
    impl, _, device = accuracy_backend
    call, ref, inputs, dtype = paged_case(spec, device, impl)
    assert_close(call(*inputs), ref(*inputs), dtype, rtol=2e-2, atol=2e-2)


@pytest.mark.api("modules.PagedPrefillGQAInfer", ops=["paged_prefill_gqa_infer"])
@pytest.mark.accuracy
@pytest.mark.parametrize("layout", ["ABAB", "AABB"])
def test_paged_bucket(accuracy_backend, layout):
    impl, _, device = accuracy_backend
    call, ref, inputs = paged_bucket_case(layout, device, impl)
    assert_close(call(*inputs)[:4].float(), ref(*inputs)[:4].float(), rtol=2e-2, atol=2e-2)


@pytest.mark.api("modules.PrefillGQAInfer", ops=["prefill_gqa_infer"])
@pytest.mark.accuracy
@pytest.mark.parametrize("case", PREFILL_CASES)
@pytest.mark.parametrize("layout", ["ABAB", "AABB"])
def test_prefill(accuracy_backend, case, layout):
    impl, _, device = accuracy_backend
    call, ref, inputs = prefill_case(case, layout, device, impl)
    assert_close(call(*inputs), ref(*inputs), torch.bfloat16, rtol=2e-2, atol=2e-2)
