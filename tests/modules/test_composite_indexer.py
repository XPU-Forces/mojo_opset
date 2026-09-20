import pytest
import torch

from mojo_opset import modules

COMPOSITE_CASES = [(batch, tokens) for batch in (1, 16, 32) for tokens in (1, 1024, 4096)]


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


def make_composite_case(case, dtype, device, implementation):

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


@pytest.mark.api("modules.Indexer", ops=["layer_norm_infer", "apply_rope_infer", "dynamic_quant", "lightning_indexer"])
@pytest.mark.parametrize("case", COMPOSITE_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_composite(accuracy_backend, case, dtype):
    implementation, _, device = accuracy_backend
    actual, reference, inputs = make_composite_case(case, dtype, device, implementation)
    with torch.no_grad():
        result = actual(*inputs)
        expected = reference(*inputs)
        assert_indexer_scores(result, expected)
