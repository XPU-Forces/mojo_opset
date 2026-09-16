import pytest
import torch

from mojo_opset import functions
from tests._checks import assert_close

COMPOSITE_CASES = [(batch, tokens) for batch in (1, 16, 32) for tokens in (1, 1024, 4096)]


INDEXER_SHAPES = [
    (8, 1024, 1024, 64, 64),
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]


def assert_indexer_scores(actual, expected):
    actual_ids, actual_scores = actual
    reference_ids, reference_scores = expected

    def ratio(actual, expected, atol, rtol, minimum):
        matches = torch.isclose(actual, expected, atol=atol, rtol=rtol)
        fraction = float(matches.float().mean())
        assert fraction >= minimum, f"{fraction:.5%} of scores match; expected at least {minimum:.5%}"

    # Preserve master's int8-quantization-step tolerances, including order-invariant top-k comparison.
    ratio(actual_scores, reference_scores, 1e-2, 2e-2, 0.98)
    if reference_ids.numel():
        expected = reference_scores.gather(-1, reference_ids).sort(-1, descending=True).values
        actual = reference_scores.gather(-1, actual_ids).sort(-1, descending=True).values
        ratio(actual, expected, 1.0, 1e-2, 0.999)


def assert_lightning_scores(actual, expected):
    # Original LightningIndexer uses forward_diff_with defaults for every dtype:
    # FP32 comparison, rtol=atol=1e-2, and all elements must pass (ptol=1).
    assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2, name="index_score")


def make_composite_case(case, dtype, device, implementation):
    from mojo_opset import modules

    batch, tokens = case
    options = dict(
        n_heads=64,
        head_dim=64,
        qk_rope_head_dim=32,
        topk=2048 if tokens >= 4096 else tokens // 2,
        max_batch_size=batch,
        max_seq_len=tokens,
    )
    reference = modules.Indexer(**options, implementation="torch_reference").to(device=device, dtype=dtype)
    actual = modules.Indexer(**options, implementation=implementation).to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn_like(parameter))
        actual.load_state_dict(reference.state_dict())
    x = torch.randn(batch, tokens, 7168, device=device, dtype=dtype)
    qr = torch.randn(batch, tokens, 1536, device=device, dtype=dtype)
    inv_freq = 1 / (10000 ** (torch.arange(0, 32, 2, device=device).float() / 32))
    angles = torch.outer(torch.arange(tokens, device=device).float(), inv_freq)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return actual, reference, (x, qr, 0, freqs, None)


def make_indexer_case(shape, dtype, device):
    batch, q_len, k_len, heads, dim = shape
    return (
        torch.randn(batch, q_len, heads, dim, device=device, dtype=dtype),
        torch.randn(batch, q_len, heads, device=device, dtype=torch.float32),
        torch.randn(batch, k_len, dim, device=device, dtype=dtype),
        torch.randn(batch, k_len, device=device, dtype=torch.float32),
    )


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.parametrize("shape", INDEXER_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_indexer(accuracy_backend, shape, dtype):
    implementation, _, device = accuracy_backend
    inputs = make_indexer_case(shape, dtype, device)
    actual = functions.lightning_indexer(*inputs, implementation=implementation)
    expected = functions.lightning_indexer(*inputs, implementation="torch_reference")
    assert_lightning_scores(actual, expected)


def _run_composite(module, inputs):
    return functions.indexer(
        inputs[0],
        inputs[1],
        module.wq_b.weight,
        module.wk.weight,
        module.k_norm.weight,
        module.k_norm.bias,
        module.weights_proj.weight,
        module.k_cache,
        module.k_scale_cache,
        *inputs[2:],
        n_heads=module.n_heads,
        head_dim=module.head_dim,
        topk=module.topk,
        norm_eps=module.k_norm.eps,
        implementation=module.implementation,
    )


@pytest.mark.api("functions.indexer")
@pytest.mark.parametrize("case", COMPOSITE_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_composite(accuracy_backend, case, dtype):
    implementation, _, device = accuracy_backend
    actual, reference, inputs = make_composite_case(case, dtype, device, implementation)
    with torch.no_grad():
        result = _run_composite(actual, inputs)
        expected = _run_composite(reference, inputs)
        assert_indexer_scores(result, expected)
