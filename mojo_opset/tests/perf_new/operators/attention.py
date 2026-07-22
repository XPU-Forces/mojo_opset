"""Attention performance cases."""

from __future__ import annotations

import math
from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset import MojoSdpa
from mojo_opset import MojoSWA
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import perf_provider
from mojo_opset.benchmark import tensor



def _decode_cases(configs: tuple[tuple[Any, ...], ...]):
    return tuple(
        perf_case(
            f"{case_id}_{layout.lower()}",
            tags=(("smoke", "full") if index == 0 else ("full",)),
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            block_size=block_size,
            dtype=value_dtype,
            gqa_layout=layout,
        )
        for index, (
            batch,
            q_heads,
            kv_heads,
            head_dim,
            max_seq_len,
            block_size,
            value_dtype,
            case_id,
            layout,
        ) in enumerate(configs)
    )


_DECODE_CONFIGS = tuple(
    (*config, layout)
    for config in (
        (8, 16, 4, 128, 1024, 32, torch.bfloat16, "bf16"),
        (8, 16, 4, 96, 1024, 128, torch.bfloat16, "bf16_paddim"),
        (8, 8, 1, 128, 8192, 128, torch.bfloat16, "bf16_long"),
    )
    for layout in ("ABAB", "AABB")
)

PAGED_DECODE_CASES = _decode_cases(_DECODE_CONFIGS)


def _decode_workload(case: Mapping[str, Any], *, sliding_window: bool) -> PerfWorkload:
    batch = int(case["batch"])
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    max_seq_len = int(case["max_seq_len"])
    block_size = int(case["block_size"])
    value_dtype = case["dtype"]
    generator = torch.Generator().manual_seed(20260716 + batch + max_seq_len + head_dim)
    total_seq_lens_cpu = torch.randint(
        1,
        max_seq_len,
        (batch,),
        dtype=torch.int32,
        generator=generator,
    )
    total_seq_lens_values = total_seq_lens_cpu.tolist()
    max_blocks_per_seq = (max(total_seq_lens_values) + block_size - 1) // block_size
    total_blocks = sum((length + block_size - 1) // block_size for length in total_seq_lens_values) + 10
    query_shape = (batch, q_heads, head_dim)
    cache_shape = (total_blocks, kv_heads, block_size, head_dim)
    table_shape = (batch, max_blocks_per_seq)

    def tensor_factory(device: str):
        block_tables = torch.full(table_shape, -1, dtype=torch.int32, device=device)
        current = 0
        for index, seq_len in enumerate(total_seq_lens_values):
            needed = (seq_len + block_size - 1) // block_size
            block_tables[index, :needed] = torch.arange(
                current,
                current + needed,
                dtype=torch.int32,
                device=device,
            )
            current += needed
        return {
            "query": torch.randn(query_shape, dtype=value_dtype, device=device),
            "key_cache": torch.randn(cache_shape, dtype=value_dtype, device=device),
            "value_cache": torch.randn(cache_shape, dtype=value_dtype, device=device),
            "total_seq_lens": torch.tensor(total_seq_lens_values, dtype=torch.int32, device=device),
            "block_tables": block_tables,
        }

    op_kwargs = {"is_causal": True, "gqa_layout": case["gqa_layout"]}
    if sliding_window:
        op_kwargs.update(global_window_size=4, local_window_size=1023)
    return PerfWorkload(
        op_kwargs=op_kwargs,
        inputs={
            "query": tensor(query_shape, value_dtype),
            "key_cache": tensor(cache_shape, value_dtype),
            "value_cache": tensor(cache_shape, value_dtype),
            "total_seq_lens": tensor((batch,), torch.int32),
            "block_tables": tensor(table_shape, torch.int32),
        },
        outputs={"output": tensor(query_shape, value_dtype)},
        kwargs={"softmax_scale": 1.0 / math.sqrt(head_dim)},
        tensor_factory=tensor_factory,
        flops=2 * batch * q_heads * max(total_seq_lens_values) * head_dim,
    )


@mojo_perf(name="mojo_paged_decode_gqa", target=MojoPagedDecodeGQA, cases=PAGED_DECODE_CASES, providers=("ttx",))
def paged_decode_gqa_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _decode_workload(case, sliding_window=False)


_SWA_DECODE_CONFIGS = tuple(
    (*config, "AABB")
    for config in (
        (8, 16, 4, 128, 1024, 32, torch.bfloat16, "bf16"),
        (8, 16, 4, 96, 1024, 128, torch.bfloat16, "bf16_paddim"),
        (8, 16, 4, 128, 8192, 128, torch.bfloat16, "bf16_long"),
    )
)

PAGED_DECODE_SWA_CASES = _decode_cases(_SWA_DECODE_CONFIGS)


@mojo_perf(name="mojo_paged_decode_swa", target=MojoPagedDecodeSWA, cases=PAGED_DECODE_SWA_CASES, providers=("ttx",))
def paged_decode_swa_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _decode_workload(case, sliding_window=True)


def _prefill_cases(configs: tuple[tuple[Any, ...], ...]):
    return tuple(
        perf_case(
            f"{case_id}_{layout.lower()}",
            tags=(("smoke", "full") if index == 0 else ("full",)),
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            max_q_len=max_q_len,
            max_kv_computed_len=max_kv_computed_len,
            block_size=block_size,
            dtype=value_dtype,
            gqa_layout=layout,
        )
        for index, (
            batch,
            q_heads,
            kv_heads,
            head_dim,
            max_q_len,
            max_kv_computed_len,
            block_size,
            value_dtype,
            case_id,
            layout,
        ) in enumerate(configs)
    )


_PREFILL_CONFIGS = tuple(
    (*config, layout)
    for config in (
        (2, 16, 4, 128, 1024, 0, 128, torch.bfloat16, "bf16"),
        (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "bf16_paddim"),
        (2, 8, 1, 128, 4096, 8192, 128, torch.bfloat16, "bf16_with_cache"),
    )
    for layout in ("ABAB", "AABB")
)

PAGED_PREFILL_CASES = _prefill_cases(_PREFILL_CONFIGS)


def _prefill_lengths(case: Mapping[str, Any]):
    """Build query and total-KV lengths.

    ``max_kv_computed_len`` is the existing cache-prefix length, so the total
    visible KV length is ``query length + cache-prefix length``.
    """

    batch = int(case["batch"])
    max_q_len = int(case["max_q_len"])
    max_kv_computed_len = int(case["max_kv_computed_len"])
    generator = torch.Generator().manual_seed(20260716 + batch + max_q_len + max_kv_computed_len)
    q_lens = torch.randint(
        max(max_q_len // 2, 1),
        max_q_len,
        (batch,),
        dtype=torch.int32,
        generator=generator,
    ).clamp(min=1)
    if max_kv_computed_len <= 0:
        cache_lens = None
        kv_lens = q_lens
    else:
        cache_lens = torch.randint(
            max(max_kv_computed_len // 2, 1),
            max_kv_computed_len,
            (batch,),
            dtype=torch.int32,
            generator=generator,
        )
        kv_lens = q_lens + cache_lens
    return q_lens.tolist(), kv_lens.tolist(), cache_lens is not None


def _prefill_workload(case: Mapping[str, Any], *, sliding_window: bool) -> PerfWorkload:
    batch = int(case["batch"])
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    block_size = int(case["block_size"])
    value_dtype = case["dtype"]
    q_lens, kv_lens, has_cache = _prefill_lengths(case)
    total_q_tokens = sum(q_lens)
    max_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    total_blocks = sum((length + block_size - 1) // block_size for length in kv_lens) + 10
    query_shape = (total_q_tokens, q_heads, head_dim)
    cache_shape = (total_blocks, kv_heads, block_size, head_dim)
    table_shape = (batch, max_blocks_per_seq)

    def tensor_factory(device: str):
        cu_q_lens = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        block_tables = torch.full(table_shape, -1, dtype=torch.int32, device=device)
        current = 0
        for index, seq_len in enumerate(kv_lens):
            needed = (seq_len + block_size - 1) // block_size
            block_tables[index, :needed] = torch.arange(
                current,
                current + needed,
                dtype=torch.int32,
                device=device,
            )
            current += needed
        mapping = {
            "query": torch.randn(query_shape, dtype=value_dtype, device=device),
            "key_cache": torch.randn(cache_shape, dtype=value_dtype, device=device),
            "value_cache": torch.randn(cache_shape, dtype=value_dtype, device=device),
            "cu_q_lens": cu_q_lens,
            "block_tables": block_tables,
        }
        if has_cache:
            mapping["cu_total_seq_lens"] = torch.tensor(
                [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
                dtype=torch.int32,
                device=device,
            )
        return mapping

    inputs = {
        "query": tensor(query_shape, value_dtype),
        "key_cache": tensor(cache_shape, value_dtype),
        "value_cache": tensor(cache_shape, value_dtype),
        "cu_q_lens": tensor((batch + 1,), torch.int32),
        "block_tables": tensor(table_shape, torch.int32),
    }
    cu_total_arg: str | None = None
    if has_cache:
        inputs["cu_total_seq_lens"] = tensor((batch + 1,), torch.int32)
        cu_total_arg = "cu_total_seq_lens"
    op_kwargs = {"is_causal": True, "gqa_layout": case["gqa_layout"]}
    if sliding_window:
        op_kwargs.update(global_window_size=4, local_window_size=1023)
    return PerfWorkload(
        op_kwargs=op_kwargs,
        inputs=inputs,
        outputs={"output": tensor(query_shape, value_dtype)},
        kwargs={
            "softmax_scale": 1.0 / math.sqrt(head_dim),
            "cu_total_seq_lens": cu_total_arg,
        },
        tensor_factory=tensor_factory,
    )


def _torch_npu_paged_prefill_supported(case: Mapping[str, Any]) -> bool:
    head_dim = int(case["head_dim"])
    block_size = int(case["block_size"])
    max_kv_computed_len = int(case["max_kv_computed_len"])
    return (
        max_kv_computed_len == 0
        and head_dim % 128 == 0
        and block_size % 128 == 0
        and block_size <= 512
    )


@mojo_perf(
    name="mojo_paged_prefill_gqa",
    target=MojoPagedPrefillGQA,
    cases=PAGED_PREFILL_CASES,
    providers=(
        "ttx",
        perf_provider(
            "torch_npu",
            supports=_torch_npu_paged_prefill_supported,
            unsupported_reason=(
                "requires no pre-existing KV cache, head_dim divisible by 128, "
                "and block_size divisible by 128 with block_size <= 512"
            ),
        ),
    ),
)
def paged_prefill_gqa_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _prefill_workload(case, sliding_window=False)


_SWA_PREFILL_CONFIGS = tuple(
    (*config, "AABB")
    for config in (
        (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "bf16"),
        (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "bf16_paddim"),
        (2, 16, 4, 128, 1024, 8192, 128, torch.bfloat16, "bf16_with_cache"),
    )
)

PAGED_PREFILL_SWA_CASES = _prefill_cases(_SWA_PREFILL_CONFIGS)


@mojo_perf(
    name="mojo_paged_prefill_swa",
    target=MojoPagedPrefillSWA,
    cases=PAGED_PREFILL_SWA_CASES,
    providers=("ttx",),
)
def paged_prefill_swa_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _prefill_workload(case, sliding_window=True)


SDPA_CASES = (
    perf_case(
        "smoke_b1_qh8_kh2_d128_s64",
        tags=("smoke",),
        batch=1,
        q_heads=8,
        kv_heads=2,
        head_dim=128,
        seq_length=64,
    ),
    perf_case(
        "b1_qh8_kh2_d128_s8192",
        tags=("full",),
        batch=1,
        q_heads=8,
        kv_heads=2,
        head_dim=128,
        seq_length=8192,
    ),
)


@mojo_perf(name="mojo_sdpa", target=MojoSdpa, cases=SDPA_CASES, providers=("ttx",))
def sdpa_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    tokens = 2 * int(case["seq_length"])
    q_shape = (batch, q_heads, tokens, head_dim)
    kv_shape = (batch, kv_heads, tokens, head_dim)
    return PerfWorkload(
        op_kwargs={"scale": 1.0 / math.sqrt(head_dim), "enable_gqa": q_heads != kv_heads},
        inputs={
            "query": tensor(q_shape, torch.bfloat16, creator=torch.randn),
            "key": tensor(kv_shape, torch.bfloat16, creator=torch.randn),
            "value": tensor(kv_shape, torch.bfloat16, creator=torch.randn),
            "mask": tensor((tokens, tokens), torch.bool, creator=torch.ones),
        },
        outputs={"output": tensor(q_shape, torch.bfloat16)},
        flops=4 * batch * q_heads * tokens * tokens * head_dim,
    )


SWA_CASES = tuple(
    perf_case(
        case_id,
        tags=(("smoke", "full") if index == 0 else ("full",)),
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_q_len=max_q_len,
        max_kv_computed_len=max_kv_computed_len,
        dtype=value_dtype,
    )
    for index, (batch, q_heads, kv_heads, head_dim, max_q_len, max_kv_computed_len, value_dtype, case_id) in enumerate(
        (
            (2, 16, 4, 128, 1024, 0, torch.bfloat16, "bf16"),
            (2, 16, 4, 96, 1024, 0, torch.bfloat16, "bf16_paddim"),
            (2, 16, 4, 128, 1024, 8192, torch.bfloat16, "bf16_with_cache"),
        )
    )
)


@mojo_perf(name="mojo_swa", target=MojoSWA, cases=SWA_CASES, providers=("ttx",))
def swa_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    value_dtype = case["dtype"]
    q_lens, kv_lens, _ = _prefill_lengths(case)
    total_q_tokens = sum(q_lens)
    total_kv_tokens = sum(kv_lens)
    q_shape = (total_q_tokens, q_heads, head_dim)
    kv_shape = (total_kv_tokens, kv_heads, head_dim)

    def tensor_factory(device: str):
        return {
            "query": torch.randn(q_shape, dtype=value_dtype, device=device),
            "key": torch.randn(kv_shape, dtype=value_dtype, device=device),
            "value": torch.randn(kv_shape, dtype=value_dtype, device=device),
            "cu_q_lens": torch.tensor(
                [0, *torch.tensor(q_lens).cumsum(0).tolist()],
                dtype=torch.int32,
                device=device,
            ),
            "cu_total_seq_lens": torch.tensor(
                [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
                dtype=torch.int32,
                device=device,
            ),
        }

    return PerfWorkload(
        op_kwargs={
            "is_causal": True,
            "gqa_layout": "AABB",
            "global_window_size": 4,
            "local_window_size": 1023,
        },
        inputs={
            "query": tensor(q_shape, value_dtype),
            "key": tensor(kv_shape, value_dtype),
            "value": tensor(kv_shape, value_dtype),
            "cu_q_lens": tensor((batch + 1,), torch.int32),
            "cu_total_seq_lens": tensor((batch + 1,), torch.int32),
        },
        outputs={"output": tensor(q_shape, value_dtype)},
        kwargs={"softmax_scale": 1.0 / math.sqrt(head_dim)},
        tensor_factory=tensor_factory,
    )
