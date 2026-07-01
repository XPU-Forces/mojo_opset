import pytest
import torch
import torch.nn as nn

from mojo_opset.experimental import MojoFusedNormRoPESageQuantStore
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import assert_deterministic
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import get_torch_device

torch.manual_seed(42)

CONFIGS = [
    # (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim)
    (8, 2, 32, 4, 128, 128),
    (8, 2, 32, 4, 128, 64),
    (16, 4, 48, 8, 96, 96),
    (4, 1, 16, 2, 128, 128),
    (8, 2, 32, 4, 64, 64),
    (32, 8, 64, 16, 128, 128),
    (4, 2, 8, 4, 96, 48),
]

SEQ_CONFIGS = [
    # (batch_size, q_lens_list, context_kv_lens_list)
    (1, [1], [0]),
    (1, [1], [15]),
    (1, [1], [127]),
    (1, [1], [1023]),
    (2, [1, 1], [0, 7]),
    (4, [1, 1, 1, 1], [0, 15, 31, 63]),
    (1, [32], [0]),
    (1, [64], [0]),
    (1, [128], [0]),
    (1, [256], [0]),
    (2, [16, 8], [5, 10]),
    (2, [64, 32], [0, 128]),
    (3, [1, 1, 1], [10, 20, 30]),
    (1, [512], [0]),
    (2, [128, 128], [64, 64]),
]

BLOCK_SIZE = 128


def _build_kv_case(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val, device):
    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32, device=device)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32, device=device)

    is_decode = all(q == 1 for q in q_lens_val)
    cu_q_lens = (
        torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ])
        if not is_decode
        else None
    )

    total_tokens = int(q_lens.sum().item()) if not is_decode else batch_size

    max_kv_len = int(torch.clamp(context_kv_lens + q_lens, min=0).max().item())
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        max(0, ckv + ql + block_size - 1) // block_size
        for ckv, ql in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)
    v_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    next_block = 0
    for b in range(batch_size):
        needed = max(0, context_kv_lens_val[b] + q_lens_val[b] + block_size - 1) // block_size
        if needed > 0:
            block_table[b, :needed] = torch.arange(next_block, next_block + needed, dtype=torch.int32, device=device)
        next_block += needed

    return {
        "total_tokens": total_tokens,
        "cu_q_lens": cu_q_lens,
        "context_kv_lens": context_kv_lens,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
    }


def _build_inputs(cfg, seq_cfg, device):
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim = cfg
    batch_size, q_lens_val, context_kv_lens_val = seq_cfg

    full_kv_case = _build_kv_case(batch_size, num_heads_full_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val, device)
    swa_kv_case = _build_kv_case(batch_size, num_heads_swa_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val, device)

    T = full_kv_case["total_tokens"]
    inputs = {
        "swa_query": torch.randn(T, num_heads_swa_q, head_dim, dtype=torch.bfloat16, device=device),
        "swa_key": torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device),
        "swa_value": torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device),
        "full_query": torch.randn(T, num_heads_full_q, head_dim, dtype=torch.bfloat16, device=device),
        "full_key": torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device),
        "full_value": torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device),
        "cos": torch.randn(T, rope_dim, dtype=torch.bfloat16, device=device),
        "sin": torch.randn(T, rope_dim, dtype=torch.bfloat16, device=device),
    }
    return inputs, full_kv_case, swa_kv_case, T


def _forward_args(inputs, full_kv_case, swa_kv_case):
    return (
        inputs["swa_query"], inputs["swa_key"], inputs["swa_value"],
        inputs["full_query"], inputs["full_key"], inputs["full_value"],
        inputs["cos"], inputs["sin"],
        full_kv_case["k_cache"].clone(), full_kv_case["v_cache"].clone(),
        swa_kv_case["k_cache"].clone(), swa_kv_case["v_cache"].clone(),
        full_kv_case["block_table"], full_kv_case["cu_q_lens"], full_kv_case["context_kv_lens"],
        swa_kv_case["block_table"], swa_kv_case["cu_q_lens"], swa_kv_case["context_kv_lens"],
    )


def _build_sage_caches(full_kv_case):
    key_cache = full_kv_case["k_cache"]
    key_pt_cache = torch.zeros_like(key_cache)
    key_pt_scale_cache = torch.zeros(
        key_cache.shape[:-1] + (1,), dtype=torch.float32, device=key_cache.device
    )
    return key_pt_cache, key_pt_scale_cache


SAGE_CONFIGS = [cfg for cfg in CONFIGS if cfg[4] == 128]


@pytest.mark.parametrize("num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim", CONFIGS)
@pytest.mark.parametrize("batch_size, q_lens_val, context_kv_lens_val", SEQ_CONFIGS)
@pytest.mark.parametrize("update_kv", [True, False])
@bypass_not_implemented
def test_diff_vs_torch_no_sage(
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim,
    batch_size, q_lens_val, context_kv_lens_val,
    update_kv,
):
    """forward_diff_with: dedicated backend vs torch reference (enable_sage=False)."""
    torch.manual_seed(42)
    device = get_torch_device()
    cfg = (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim)
    seq_cfg = (batch_size, q_lens_val, context_kv_lens_val)

    op = MojoFusedNormRoPESageQuantStore(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
        enable_sage=False,
    ).to(device)

    op_ref = MojoFusedNormRoPESageQuantStore._registry.get("torch")(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
        enable_sage=False,
    ).to(device)

    for p in op_ref.parameters():
        nn.init.normal_(p, mean=1.0, std=0.1)
    op.load_state_dict(op_ref.state_dict())

    full_kv_case = _build_kv_case(batch_size, num_heads_full_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val, device)
    swa_kv_case = _build_kv_case(batch_size, num_heads_swa_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val, device)

    T = full_kv_case["total_tokens"]
    swa_query = torch.randn(T, num_heads_swa_q, head_dim, dtype=torch.bfloat16, device=device)
    swa_key = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device)
    swa_value = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device)
    full_query = torch.randn(T, num_heads_full_q, head_dim, dtype=torch.bfloat16, device=device)
    full_key = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device)
    full_value = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device)

    cos = torch.randn(T, rope_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(T, rope_dim, dtype=torch.bfloat16, device=device)

    op.forward_diff_with(
        op_ref,
        swa_query, swa_key, swa_value,
        full_query, full_key, full_value,
        cos, sin,
        full_kv_case["k_cache"].clone(), full_kv_case["v_cache"].clone(),
        swa_kv_case["k_cache"].clone(), swa_kv_case["v_cache"].clone(),
        full_kv_case["block_table"], full_kv_case["cu_q_lens"], full_kv_case["context_kv_lens"],
        swa_kv_case["block_table"], swa_kv_case["cu_q_lens"], swa_kv_case["context_kv_lens"],
        update_kv=update_kv,
        atol=1, rtol=0.05, ptol=0.999,
    )


@pytest.mark.parametrize("num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim", SAGE_CONFIGS)
@pytest.mark.parametrize("batch_size, q_lens_val, context_kv_lens_val", SEQ_CONFIGS)
@pytest.mark.parametrize("update_kv", [True, False])
@bypass_not_implemented
def test_diff_vs_torch_with_sage(
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim,
    batch_size, q_lens_val, context_kv_lens_val,
    update_kv,
):
    """Dedicated ixformer SAGE path matches torch reference, including SAGE caches."""
    torch.manual_seed(42)
    device = get_torch_device()
    cfg = (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim)
    seq_cfg = (batch_size, q_lens_val, context_kv_lens_val)

    op = MojoFusedNormRoPESageQuantStore(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
        enable_sage=True,
    ).to(device)
    op_ref = MojoFusedNormRoPESageQuantStore._registry.get("torch")(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
        enable_sage=True,
    ).to(device)

    for p in op_ref.parameters():
        nn.init.normal_(p, mean=1.0, std=0.1)
    op.load_state_dict(op_ref.state_dict())

    inputs, full_kv_case, swa_kv_case, T = _build_inputs(cfg, seq_cfg, device)
    fused_kpt_cache, fused_scale_cache = _build_sage_caches(full_kv_case)
    ref_kpt_cache, ref_scale_cache = _build_sage_caches(full_kv_case)

    fused_out = op(
        *_forward_args(inputs, full_kv_case, swa_kv_case),
        sage_full_k_pt_cache=fused_kpt_cache,
        sage_full_k_pt_scale_cache=fused_scale_cache,
        update_kv=update_kv,
    )
    ref_out = op_ref(
        *_forward_args(inputs, full_kv_case, swa_kv_case),
        sage_full_k_pt_cache=ref_kpt_cache,
        sage_full_k_pt_scale_cache=ref_scale_cache,
        update_kv=update_kv,
    )

    assert len(fused_out) == len(ref_out) == 12
    output_names = (
        "swa_q", "full_q", "full_key", "full_k_scale",
        "swa_key", "swa_k_scale", "full_value", "full_v_scale",
        "swa_value", "swa_v_scale", "full_key_pt", "full_key_pt_scale",
    )
    for name, fused, ref in zip(output_names, fused_out, ref_out):
        if fused is None or ref is None:
            assert fused is ref, name
        elif fused.dtype == torch.int8:
            diff = (fused.to(torch.int32) - ref.to(torch.int32)).abs().float()
            assert (diff <= 1).float().mean().item() >= 0.99, name
        elif name.endswith("scale"):
            torch.testing.assert_close(
                fused.float(), ref.float(), atol=1e-3, rtol=1e-2
            )
        else:
            assert_close(fused, ref)

    full_key_pt_int8, full_key_pt_scale = fused_out[-2], fused_out[-1]
    if not update_kv:
        assert full_key_pt_int8 is None
        assert full_key_pt_scale is None
        assert not bool((fused_kpt_cache != 0).any().item())
        assert not bool((fused_scale_cache != 0).any().item())
        return

    assert full_key_pt_int8.shape == (T, num_heads_full_k, head_dim)
    assert full_key_pt_int8.dtype == torch.int8
    assert full_key_pt_scale.shape == (T, num_heads_full_k, 1)
    assert full_key_pt_scale.dtype == torch.float32

    kpt_written = ref_kpt_cache != 0
    if bool(kpt_written.any().item()):
        diff = (fused_kpt_cache.to(torch.int32) - ref_kpt_cache.to(torch.int32)).abs().float()[kpt_written]
        assert (diff <= 1).float().mean().item() >= 0.99

    scale_written = ref_scale_cache != 0
    if bool(scale_written.any().item()):
        torch.testing.assert_close(
            fused_scale_cache[scale_written],
            ref_scale_cache[scale_written],
            atol=1e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim", CONFIGS)
@pytest.mark.parametrize("batch_size, q_lens_val, context_kv_lens_val", SEQ_CONFIGS)
@pytest.mark.parametrize("enable_sage", [True, False])
@pytest.mark.parametrize("update_kv", [True, False])
def test_torch_reference_logic(
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim,
    batch_size, q_lens_val, context_kv_lens_val,
    enable_sage, update_kv,
):
    """Validate the torch reference: tuple shape/dtype + SAGE/static quant math."""
    torch.manual_seed(42)
    device = "cpu" if get_torch_device() == "meta" else get_torch_device()
    cfg = (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim)
    seq_cfg = (batch_size, q_lens_val, context_kv_lens_val)

    # The session fixture sets the default device to ``meta`` on hosts without
    # an accelerator; scope a concrete device so the pure-torch math runs.
    with torch.device(device):
        op = MojoFusedNormRoPESageQuantStore._registry.get("torch")(
            num_heads_swa_q=num_heads_swa_q,
            num_heads_swa_k=num_heads_swa_k,
            num_heads_full_q=num_heads_full_q,
            num_heads_full_k=num_heads_full_k,
            head_dim=head_dim,
            norm_eps=1e-5,
            use_query_norm=True,
            use_key_norm=True,
            quant_dtype=torch.int8,
            enable_sage=enable_sage,
        )
        for p in op.parameters():
            nn.init.normal_(p, mean=1.0, std=0.1)

        inputs, full_kv_case, swa_kv_case, T = _build_inputs(cfg, seq_cfg, device)
        out = op(*_forward_args(inputs, full_kv_case, swa_kv_case), update_kv=update_kv)

    # --- contract: always a 12-tuple, query outputs present ---
    assert len(out) == 12
    (swa_q_out, full_q_out,
     full_key_q, full_k_scale,
     swa_key_q, swa_k_scale,
     full_val_q, full_v_scale,
     swa_val_q, swa_v_scale,
     full_key_pt_int8, full_key_pt_scale) = out

    assert swa_q_out.shape == (T, num_heads_swa_q, head_dim)
    assert full_q_out.shape == (T, num_heads_full_q, head_dim)
    assert swa_q_out.dtype == torch.bfloat16
    assert full_q_out.dtype == torch.bfloat16

    if not update_kv:
        # YOCO reuse layer: everything except the queries is None.
        for t in out[2:]:
            assert t is None
        return

    # --- static per-channel int8 K/V quant ---
    assert full_key_q.shape == (T, num_heads_full_k, head_dim)
    assert full_key_q.dtype == torch.int8
    assert full_k_scale.shape == (num_heads_full_k, head_dim)
    assert full_val_q.shape == (T, num_heads_full_k, head_dim)
    assert full_val_q.dtype == torch.int8
    assert swa_key_q.shape == (T, num_heads_swa_k, head_dim)
    assert swa_key_q.dtype == torch.int8
    assert swa_k_scale.shape == (num_heads_swa_k, head_dim)
    assert swa_val_q.shape == (T, num_heads_swa_k, head_dim)
    assert swa_val_q.dtype == torch.int8

    # --- recompute the post-norm+rope full key (the SAGE/static quant source) ---
    swa_q_n, swa_k_n, full_q_n, full_k_n = op.qk_norm([
        inputs["swa_query"], inputs["swa_key"], inputs["full_query"], inputs["full_key"],
    ])
    _, ref_full_key = op.apply_rope(full_q_n, full_k_n, inputs["cos"], inputs["sin"], head_first=False)
    ref_full_key = ref_full_key.float()

    # static dequant should reconstruct the key within one quant step (per channel).
    static_deq = full_key_q.float() * full_k_scale.float()
    allowed_static = full_k_scale.float().abs().unsqueeze(0)  # (1, kh, hd)
    assert torch.all((static_deq - ref_full_key).abs() <= allowed_static + 1e-2)

    # --- SAGE per-token int8 key + per-token scale ---
    if enable_sage:
        assert full_key_pt_int8.shape == (T, num_heads_full_k, head_dim)
        assert full_key_pt_int8.dtype == torch.int8
        assert full_key_pt_scale.shape == (T, num_heads_full_k, 1)

        # per-token scale = amax over head_dim / 127 (matching MojoDynamicQuant).
        expected_scale = ref_full_key.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
        expected_scale = torch.where(expected_scale < 1e-6, torch.ones_like(expected_scale), expected_scale)
        torch.testing.assert_close(full_key_pt_scale.float(), expected_scale, atol=1e-3, rtol=1e-2)

        # per-token dequant should reconstruct the key within one quant step.
        pt_deq = full_key_pt_int8.float() * full_key_pt_scale.float()
        allowed_pt = full_key_pt_scale.float().abs()  # (T, kh, 1) -> broadcast over hd
        assert torch.all((pt_deq - ref_full_key).abs() <= allowed_pt + 1e-2)
    else:
        assert full_key_pt_int8 is None
        assert full_key_pt_scale is None


# ---------------------------------------------------------------------------
# 3. Determinism of the torch reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_sage", [True, False])
def test_torch_reference_deterministic(enable_sage):
    device = "cpu" if get_torch_device() == "meta" else get_torch_device()
    cfg = (8, 2, 32, 4, 128, 128)
    seq_cfg = (2, [16, 8], [5, 10])

    with torch.device(device):
        op = MojoFusedNormRoPESageQuantStore._registry.get("torch")(
            num_heads_swa_q=8, num_heads_swa_k=2, num_heads_full_q=32, num_heads_full_k=4,
            head_dim=128, norm_eps=1e-5, use_query_norm=True, use_key_norm=True,
            quant_dtype=torch.int8, enable_sage=enable_sage,
        )
        for p in op.parameters():
            nn.init.normal_(p, mean=1.0, std=0.1)

        inputs, full_kv_case, swa_kv_case, _ = _build_inputs(cfg, seq_cfg, device)
        args = _forward_args(inputs, full_kv_case, swa_kv_case)

        assert_deterministic(lambda: op(*args, update_kv=True))
