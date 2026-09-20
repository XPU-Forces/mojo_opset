import pytest
import torch

from mojo_opset import functions
from mojo_opset.utils.flex_attention_mask import create_flex_block_mask
from tests.functions.test_flex_attention import BUSINESS_FLEX_CASES
from tests.functions.test_flex_attention import build_problem
from tests.performance._workload import training


@pytest.mark.api("functions.flex_attention")
@pytest.mark.parametrize(
    "batch_size,q_head,kv_head,head_dim,data_lens,data_types,sliding_windows,global_windows,dtype,mask_func",
    BUSINESS_FLEX_CASES,
)
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_flex_attention(
    benchmark,
    perf_environment,
    batch_size,
    q_head,
    kv_head,
    head_dim,
    data_lens,
    data_types,
    sliding_windows,
    global_windows,
    dtype,
    mask_func,
    phase,
):
    _, device, _, implementation = perf_environment

    def factory():
        problem = build_problem(
            batch_size,
            q_head,
            kv_head,
            head_dim,
            data_lens,
            data_types,
            sliding_windows,
            global_windows,
            dtype,
            mask_func,
            device,
        )
        mask = create_flex_block_mask(
            mask_func(problem),
            B=1,
            H=1,
            Q_LEN=problem["total_s"],
            KV_LEN=problem["total_s"],
            device=device,
            BLOCK_SIZE=128,
        )
        inputs = tuple(problem[name].requires_grad_(True) for name in ("q", "k", "v"))
        return training(
            lambda: functions.flex_attention(*inputs, block_mask=mask, implementation=implementation),
            inputs,
            phase,
            grad_factory=lambda output: torch.full_like(output, 520000 / output.numel()),
        )

    benchmark(
        factory=factory,
        op="flex_attention",
        batch=batch_size,
        q_heads=q_head,
        kv_heads=kv_head,
        head_dim=head_dim,
        data_lens=data_lens,
        mask=mask_func.__name__,
        dtype=str(dtype),
        phase=phase,
    )
