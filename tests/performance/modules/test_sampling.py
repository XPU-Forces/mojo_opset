import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.TopKSampling", ops=["top_k_sampling"])
@pytest.mark.parametrize("batch,vocab,top_k", [(120, 151936, 20), (15, 155136, 50), (18, 155136, 100)])
def test_top_k(benchmark, perf_environment, batch, vocab, top_k):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.TopKSampling(top_k=top_k, min_tokens_to_keep=1, implementation=implementation)
        logits = torch.randn(batch, vocab, device=device)
        return lambda: module(logits)

    benchmark(
        factory=factory, op="top_k_sampling", batch=batch, vocab=vocab, top_k=top_k, dtype="float32", phase="forward"
    )


@pytest.mark.api("modules.TopPFilter", ops=["top_p_filter"])
@pytest.mark.parametrize("batch,vocab,rand_top_k", [(120, 151936, 1000), (15, 155136, 100), (18, 155136, 100)])
def test_top_p_filter(benchmark, perf_environment, batch, vocab, rand_top_k):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.TopPFilter(implementation=implementation)
        logits = torch.randn(batch, vocab, device=device)
        return lambda: module(logits, top_p=0.7, min_tokens_to_keep=1, rand_top_k=rand_top_k)

    benchmark(
        factory=factory,
        op="top_p_filter",
        batch=batch,
        vocab=vocab,
        rand_top_k=rand_top_k,
        top_p=0.7,
        dtype="float32",
        phase="forward",
    )


@pytest.mark.parametrize(
    "joined",
    [
        pytest.param(
            joined,
            marks=pytest.mark.api(
                "modules.JoinProbRejectSampling" if joined else "modules.RejectSampling",
                ops=["join_prob_reject_sampling" if joined else "reject_sampling"],
            ),
        )
        for joined in [False, True]
    ],
)
def test_reject(benchmark, perf_environment, joined):
    _, device, _, implementation = perf_environment
    op = "join_prob_reject_sampling" if joined else "reject_sampling"

    def factory():
        module = (modules.JoinProbRejectSampling if joined else modules.RejectSampling)(implementation=implementation)
        logits = (torch.rand if joined else torch.randn)(15, 4, 155136, device=device)
        draft = torch.randint(155136, (15, 3), device=device)
        probs = (torch.rand if joined else torch.ones)(15, 3, device=device)
        return lambda: module(logits, draft, probs)

    benchmark(factory=factory, op=op, batch=15, spec_step=3, vocab=155136, dtype="float32", phase="forward")
