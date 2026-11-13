"""A5 synthetic 8:1 KV cases, not a captured Omni multimodal layout.

Select with e.g. ``-k 'q32k and causal and packed and p0.1'``. Metadata and
payload allocation stay outside timing. ``with_rng`` refills the payload in
each forward/step; backward reuses its saved forward and immutable mask.
The separate RNG phase measures refill only. Model FLOPs count visible pairs
and exclude softmax, dropout and backward recomputation; no peak rate is assumed.
"""

import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions
from mojo_opset.utils.attention_dropout import create_flex_dropout_mask
from mojo_opset.utils.attention_dropout import refill_flex_dropout_mask
from mojo_opset.utils.flex_attention_mask import create_flex_block_mask
from tests._fixtures import check_implementation
from tests.performance._workload import training

_PHASES = ("forward", "backward", "forward_backward")
_WORKLOADS = [pytest.param(0.0, None, False, phase, id=f"p0-{phase}") for phase in _PHASES]
_WORKLOADS += [
    pytest.param(
        probability,
        packed,
        include_rng,
        phase,
        id=f"p{probability}-{'packed' if packed else 'uint8'}-{phase}-{'with_rng' if include_rng else 'precomputed'}",
    )
    for probability in (0.1, 0.01)
    for packed in (False, True)
    for phase, include_rng in (
        *((phase, False) for phase in _PHASES),
        ("forward", True),
        ("forward_backward", True),
        ("rng", True),
    )
]


def _mask_case(q_len, pattern):
    kv_len = q_len // 8
    prefix = (8 * kv_len + 136) // 137
    gap = window = (4 * kv_len + 136) // 137

    def mask_mod(b, h, q, k):
        end = (q + 1) // 8
        if pattern == "disjoint":
            return ((k < end) & (k < prefix)) | ((k >= end + gap) & (k < end + gap + window))
        if pattern == "empty_rows":
            return (k < end) & (q % 7 != 0)
        return k < end

    # Count visible pairs on the CPU without materializing a dense Q x KV mask.
    qi = torch.arange(q_len)
    end = (qi + 1) // 8
    counts = end
    if pattern == "disjoint":
        counts = end.clamp(max=prefix) + (kv_len - end - gap).clamp(min=0, max=window)
    elif pattern == "empty_rows":
        counts = end * (qi % 7 != 0)
    return mask_mod, int(counts.sum())


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.parametrize("q_len", [pytest.param(8192, id="q8k"), pytest.param(32768, id="q32k")])
@pytest.mark.parametrize("pattern", ["causal", "disjoint", "empty_rows"])
@pytest.mark.parametrize("probability,packed,include_rng,phase", _WORKLOADS)
def test_flex_attention_v2(benchmark, perf_environment, q_len, pattern, probability, packed, include_rng, phase):
    _, device, target, implementation = perf_environment
    selected = check_implementation(("flex_attention_v2",), implementation, target)["flex_attention_v2"]
    if Target.parse(target).arch_key != "npu.a5" or selected != "triton":
        pytest.skip("These workloads measure A5 Triton attention and its Triton dropout RNG")

    batch, q_heads, kv_heads, head_dim = 1, 6, 2, 128
    kv_len = q_len // 8
    mask_mod, visible_pairs_per_head = _mask_case(q_len, pattern)
    mask = create_flex_block_mask(mask_mod, Q_LEN=q_len, KV_LEN=kv_len, device=device, BLOCK_SIZE=128)
    active_blocks = int(mask.kv_num_blocks.sum())
    if mask.full_kv_num_blocks is not None:
        active_blocks += int(mask.full_kv_num_blocks.sum())
    payload_bytes = 0 if probability == 0 else batch * q_heads * active_blocks * 128 * (16 if packed else 128)

    def factory():
        requires_grad = phase in ("backward", "forward_backward")
        q = torch.randn(
            batch, q_heads, q_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=requires_grad
        )
        k = torch.randn(
            batch, kv_heads, kv_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=requires_grad
        )
        v = torch.randn_like(k, requires_grad=requires_grad)
        state = (
            None if probability == 0 else create_flex_dropout_mask(q, k, mask, probability, bitpacked=packed, seed=53)
        )
        next_seed = 54

        def refill():
            nonlocal next_seed
            refill_flex_dropout_mask(state, seed=next_seed)
            next_seed += 1
            return state.tiles

        if phase == "rng":
            return refill

        def call():
            if include_rng:
                refill()
            return functions.flex_attention_v2(
                q, k, v, block_mask=mask, enable_gqa=True, dropout=state, implementation=implementation
            )

        return training(call, (q, k, v), phase)

    visible_pairs = batch * q_heads * visible_pairs_per_head
    benchmark(
        factory=factory,
        op="flex_attention_v2",
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        q_len=q_len,
        kv_len=kv_len,
        head_dim=head_dim,
        dtype="bfloat16",
        mask=pattern,
        dropout_p=probability,
        dropout_storage="none" if probability == 0 else "packed" if packed else "uint8",
        include_rng=include_rng,
        rng="none" if probability == 0 else "triton_philox4x32_10",
        dropout_payload_bytes=payload_bytes,
        visible_pairs=visible_pairs,
        model_flops={"forward": 4, "backward": 8, "forward_backward": 12, "rng": 0}[phase] * head_dim * visible_pairs,
        phase=phase,
    )
