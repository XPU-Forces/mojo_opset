import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

ENCODING_CASES = [
    "nf4-prefill",
    "nf4-decode",
]


NF4_DIMS = [128, 1024, 4096]


def make_nf4_lut(vocab, dim, device):
    indices = torch.randint(16, (vocab, dim), device="cpu", dtype=torch.uint8)
    packed = (indices[:, ::2] | (indices[:, 1::2] << 4)).to(device=device, dtype=torch.int8)
    return packed, torch.randn(vocab, dim // 64, device=device), torch.randn(vocab, dim // 64, device=device)


def make_encoding_case(case, device):
    mode = case.rsplit("-", 1)[-1]
    vocab, embed_dim, oe_dim = 128, 64, 64
    sizes, grams = [131, 133, 135, 139], [2, 2, 3, 3]
    if mode == "prefill":
        ids = torch.arange(1, 33, device=device).expand(32, 32).flatten()
        lens = torch.full((32,), 32, device=device)
        history = torch.zeros(32, 2, device=device, dtype=torch.int64)
    else:
        ids, lens = torch.arange(1, 17, device=device), None
        history = torch.zeros(16, 2, device=device, dtype=torch.int64)
    weight, scale, mean = make_nf4_lut(sum(sizes), oe_dim, device)
    options = dict(
        _ori_embedding_weight=torch.randn(vocab, embed_dim, device=device),
        _mega_embedding_weight=weight,
        _mega_embedding_scale=scale,
        _mega_embedding_mean=mean,
        _mega_embedding_group_size=64,
    )
    if lens is None:
        ids = ids.view(-1, 1)
    return (vocab, embed_dim, oe_dim, sizes, grams), options, (ids, history, lens)


def make_nf4_case(dim, device):
    ids = torch.tensor([[0, 17, 32], [128, 256, 257]], device=device)
    return (ids, *make_nf4_lut(257, dim, device))


@pytest.mark.api("modules.NF4DequantEmbedding", ops=["embedding_nf4_dequant"])
@pytest.mark.parametrize("dim", NF4_DIMS)
@pytest.mark.accuracy
def test_nf4(accuracy_backend, dim):
    implementation, _, device = accuracy_backend
    ids, *weights = make_nf4_case(dim, device)
    actual = modules.NF4DequantEmbedding(
        *weights, group_size=64, output_dtype=torch.float32, implementation=implementation
    )(ids)
    expected = modules.NF4DequantEmbedding(
        *weights, group_size=64, output_dtype=torch.float32, implementation="torch_reference"
    )(ids)
    assert_close(actual, expected, rtol=0, atol=1e-5)


def _encoding_modules(case, device, implementation):
    config, options, inputs = make_encoding_case(case, device)
    reference_options = options.copy()
    if case.startswith("cpu"):
        reference_options["mega_embedding_cpu_only"] = False
        reference_options["_mega_embedding_weight"] = options["_mega_embedding_weight"].to(device)
    reference = modules.OverEncoding(*config, **reference_options, implementation="torch_reference").to(device)
    actual = modules.OverEncoding(*config, **options, implementation=implementation).to(device)
    with torch.no_grad():
        for part in reference.modules():
            if isinstance(part, (torch.nn.Embedding, torch.nn.Linear)):
                if case.startswith("basic"):
                    part.weight.copy_(torch.arange(part.weight.shape[0], device=device)[:, None].expand_as(part.weight))
                elif not case.startswith(("nf4", "cpu")):
                    part.weight.fill_(2)
                    part.weight.fill_diagonal_(1)
                elif case.startswith("cpu") and isinstance(part, torch.nn.Linear):
                    part.weight.fill_(2)
                    part.weight.fill_diagonal_(1)
        actual.load_state_dict(reference.state_dict(), strict=False)
    return actual, reference, inputs


@pytest.mark.api("modules.OverEncoding", ops=["over_encoding_decode"])
@pytest.mark.parametrize("case", ENCODING_CASES)
@pytest.mark.accuracy
def test_encoding(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    actual, reference, inputs = _encoding_modules(case, device, implementation)
    with torch.no_grad():
        assert_close(actual(*inputs), reference(*inputs), rtol=1e-5, atol=1e-5)
