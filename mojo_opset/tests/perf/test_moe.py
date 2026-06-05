import pytest
import torch

from mojo_opset import MojoMoE
from mojo_opset.core.operators.moe import MojoExperts
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


def _make_tokens_per_expert(num_tokens: int, num_experts: int) -> torch.Tensor:
    """Even-ish distribution; deterministic for perf reproducibility."""
    base = num_tokens // num_experts
    rem = num_tokens - base * num_experts
    counts = [base + (1 if i < rem else 0) for i in range(num_experts)]
    return torch.tensor(counts, dtype=torch.int32)


@pytest.mark.parametrize(
    "sorted_hidden_states, tokens_per_expert, hidden_size, intermediate_size",
    [
        (
            torch.randn(num_tokens, hidden_size, dtype=dtype),
            _make_tokens_per_expert(num_tokens, num_experts),
            hidden_size,
            intermediate_size,
        )
        for (num_experts, hidden_size, intermediate_size, num_tokens) in [
            (16, 1024, 2048, 256),
            (32, 1024, 4096, 1024),
            (64, 1024, 4096, 2048),
        ]
        for dtype in [torch.bfloat16]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_moe_experts(sorted_hidden_states, tokens_per_expert, hidden_size, intermediate_size):
    """Bench MojoExperts (stacked grouped GEMM with SwiGLU) on bf16."""
    torch.manual_seed(0)
    num_experts = tokens_per_expert.numel()
    dtype = sorted_hidden_states.dtype
    device = sorted_hidden_states.device

    experts = MojoExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu",
        device=device,
        dtype=dtype,
    )

    perf(lambda: experts(sorted_hidden_states, tokens_per_expert))  # noqa: F821


@pytest.mark.parametrize(
    "x, top_k, num_experts, hidden_size, intermediate_size",
    [
        (
            torch.randn(num_tokens, hidden_size, dtype=dtype),
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
        )
        for (num_experts, top_k, hidden_size, intermediate_size, num_tokens) in [
            (16, 4, 1024, 2048, 64),
            (32, 8, 1024, 4096, 128),
            (64, 8, 1024, 4096, 256),
        ]
        for dtype in [torch.bfloat16]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_moe_full(x, top_k, num_experts, hidden_size, intermediate_size):
    """Bench end-to-end MojoMoE for context (Experts is the dominant cost)."""
    torch.manual_seed(0)
    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=x.device,
        dtype=x.dtype,
    )
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()

    perf(lambda: moe(x))  # noqa: F821
