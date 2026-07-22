"""Sampling performance cases."""

from functools import partial
from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoJoinProbRejectSampling
from mojo_opset import MojoRejectSampling
from mojo_opset import MojoTopKSampling
from mojo_opset import MojoTopPFilter
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor



TOPK_CASES = tuple(
    perf_case(
        f"b{batch}_v{vocab}_k{top_k}",
        tags=(("smoke", "full") if batch == 15 else ("full",)),
        batch=batch,
        vocab=vocab,
        top_k=top_k,
        min_tokens_to_keep=1,
    )
    for batch, vocab, top_k in ((120, 151936, 20), (15, 155136, 50), (18, 155136, 100))
)


@mojo_perf(name="mojo_topk_sampling", target=MojoTopKSampling, cases=TOPK_CASES)
def topk_sampling_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["batch"]), int(case["vocab"]))
    return PerfWorkload(
        op_kwargs={
            "top_k": int(case["top_k"]),
            "min_tokens_to_keep": int(case["min_tokens_to_keep"]),
        },
        inputs={"logits": tensor(shape, torch.float32, creator=torch.randn)},
        outputs={"filtered": tensor(shape, torch.float32)},
    )


TOPP_CASES = tuple(
    perf_case(
        f"b{batch}_v{vocab}_randk{rand_top_k}",
        tags=(("smoke", "full") if batch == 15 else ("full",)),
        batch=batch,
        vocab=vocab,
        rand_top_k=rand_top_k,
        top_p=0.7,
        min_tokens_to_keep=1,
    )
    for batch, vocab, rand_top_k in ((120, 151936, 1000), (15, 155136, 100), (18, 155136, 100))
)


@mojo_perf(name="mojo_topp_filter", target=MojoTopPFilter, cases=TOPP_CASES)
def topp_filter_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["batch"]), int(case["vocab"]))
    return PerfWorkload(
        inputs={"logits": tensor(shape, torch.float32, creator=torch.randn)},
        outputs={"filtered": tensor(shape, torch.float32)},
        args=(
            "logits",
            float(case["top_p"]),
            int(case["min_tokens_to_keep"]),
            int(case["rand_top_k"]),
        ),
    )


REJECT_CASES = (
    perf_case("b15_s3_v155136", tags=("smoke", "full"), batch=15, spec_step=3, vocab=155136),
)


def _reject_workload(case: Mapping[str, Any], *, joined: bool) -> PerfWorkload:
    batch = int(case["batch"])
    spec_step = int(case["spec_step"])
    vocab = int(case["vocab"])
    logits_creator = torch.rand if joined else torch.randn
    draft_prob_creator = torch.rand if joined else torch.ones
    return PerfWorkload(
        inputs={
            "target_logits": tensor((batch, spec_step + 1, vocab), torch.float32, creator=logits_creator),
            "draft_tokens": tensor(
                (batch, spec_step),
                torch.int64,
                creator=partial(torch.randint, low=0, high=vocab),
            ),
            "draft_probs": tensor((batch, spec_step), torch.float32, creator=draft_prob_creator),
        },
        outputs={"accepted_tokens": tensor((batch, spec_step + 1), torch.int64)},
    )


@mojo_perf(name="mojo_reject_sampling", target=MojoRejectSampling, cases=REJECT_CASES)
def reject_sampling_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _reject_workload(case, joined=False)


@mojo_perf(
    name="mojo_join_prob_reject_sampling",
    target=MojoJoinProbRejectSampling,
    cases=REJECT_CASES,
)
def join_prob_reject_sampling_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _reject_workload(case, joined=True)
