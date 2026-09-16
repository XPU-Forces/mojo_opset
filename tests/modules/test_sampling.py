import random

from functools import partial

import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

PENALTY_SHAPES = [(20, 151936)]


REJECTION_CASES = [(15, 155136, 3)]


SAMPLING_CASES = [
    ("top_k_sampling", (20, 151936), dict(top_k=10, min_tokens_to_keep=1)),
    ("top_p_sampling", (20, 151936), dict(top_p=0.75, min_tokens_to_keep=1, rand_top_k=1000)),
    ("top_p_filter", (20, 151936), dict(top_p=0.75, min_tokens_to_keep=1, rand_top_k=1000)),
    ("top_p_filter", (60, 155136), dict(top_p=0.7, min_tokens_to_keep=1, rand_top_k=100)),
]


SAMPLING_MODULES = {
    "top_k_sampling": "TopKSampling",
    "top_p_sampling": "TopPSampling",
    "top_p_filter": "TopPFilter",
    "reject_sampling": "RejectSampling",
    "join_prob_reject_sampling": "JoinProbRejectSampling",
    "apply_penalties_temperature": "ApplyPenaltiesTemperature",
}


def accepted_prefix(result):
    tokens, lengths = result
    valid = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
    return tokens.masked_fill(~valid, 0).long(), lengths.long()


def check_sampling(name, actual_fn, reference_fn, logits):
    if name == "top_p_filter":
        actual_probs, actual_ids = actual_fn(logits.clone())
        expected_probs, expected_ids = reference_fn(logits.clone())
        assert_close(actual_probs.float(), expected_probs.float(), torch.float32, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(actual_ids.sort(-1).values, expected_ids.sort(-1).values, rtol=0, atol=0)
        return
    # Different sampling algorithms need not draw the same token under one seed.
    # Retain master's 200-draw nucleus statistics and apply the same check to top-k.
    actual_probs, expected_probs = [], []
    for _ in range(200):
        expected_p, expected_ids = reference_fn(logits.clone())
        actual_p, actual_ids = actual_fn(logits.clone())
        assert actual_ids.shape == expected_ids.shape
        assert bool(((actual_ids >= 0) & (actual_ids < logits.shape[-1])).all())
        actual_probs.append(actual_p.float())
        expected_probs.append(expected_p.float())
    actual, expected = torch.cat(actual_probs, 1), torch.cat(expected_probs, 1)
    for reduction in (torch.mean, torch.std):
        assert_close(reduction(actual, dim=1), reduction(expected, dim=1), torch.float32, rtol=1e-2, atol=1e-2)


def make_penalty_case(shape, device):
    batch, vocab = shape
    logits = torch.randn(shape, device=device)
    frequencies = torch.randint(0, 5, shape, device=device, dtype=torch.int32)
    frequencies[torch.rand(shape, device=device) > 0.05] = 0
    token_freqs = [row if bool(row.any()) else None for row in frequencies]
    rng = random.Random(42)
    frequency = [rng.uniform(-0.5, 0.5) for _ in range(batch)]
    presence = [rng.uniform(-0.5, 0.5) for _ in range(batch)]
    repetition = [rng.uniform(0.5, 3) for _ in range(batch)]
    temperatures = [rng.uniform(0.1, 2) for _ in range(batch)]
    return logits, (token_freqs, presence, frequency, repetition, temperatures)


def make_rejection_case(name, case, device):
    batch, vocab, steps = case
    target = torch.randn(batch, steps + 1, vocab, device=device)
    if name == "join_prob_reject_sampling":
        target = target.softmax(-1)
    return target, torch.randint(vocab, (batch, steps), device=device), torch.ones(batch, steps, device=device)


def _sampler(name, options, implementation):
    module = getattr(modules, SAMPLING_MODULES[name])
    if name == "top_p_filter":
        return partial(module(implementation=implementation), **options)
    return module(**options, implementation=implementation)


@pytest.mark.api("modules.TopKSampling", "modules.TopPSampling", "modules.TopPFilter")
@pytest.mark.parametrize("name,shape,options", SAMPLING_CASES)
@pytest.mark.accuracy
def test_sampling(accuracy_backend, name, shape, options):
    implementation, _, device = accuracy_backend
    check_sampling(
        name,
        _sampler(name, options, implementation),
        _sampler(name, options, "torch_reference"),
        torch.randn(shape, device=device),
    )


@pytest.mark.api("modules.RejectSampling", "modules.JoinProbRejectSampling")
@pytest.mark.parametrize("name", ["reject_sampling", "join_prob_reject_sampling"])
@pytest.mark.parametrize("case", REJECTION_CASES)
@pytest.mark.accuracy
def test_rejection(accuracy_backend, name, case):
    implementation, _, device = accuracy_backend
    inputs = make_rejection_case(name, case, device)
    actual = accepted_prefix(_sampler(name, {}, implementation)(*inputs, random_seed=42))
    expected = accepted_prefix(_sampler(name, {}, "torch_reference")(*inputs, random_seed=42))
    for a, e in zip(actual, expected):
        assert_close(a, e, rtol=0, atol=0)


@pytest.mark.api("modules.ApplyPenaltiesTemperature")
@pytest.mark.parametrize("shape", PENALTY_SHAPES)
@pytest.mark.accuracy
def test_penalties(accuracy_backend, shape):
    implementation, _, device = accuracy_backend
    logits, options = make_penalty_case(shape, device)
    actual = modules.ApplyPenaltiesTemperature(implementation=implementation)(logits.clone(), *options)
    expected = modules.ApplyPenaltiesTemperature(implementation="torch_reference")(logits.clone(), *options)
    assert_close(actual, expected, logits.dtype)
