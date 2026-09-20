import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

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


@pytest.mark.api("functions.embedding_nf4_dequant")
@pytest.mark.parametrize("dim", NF4_DIMS)
@pytest.mark.accuracy
def test_nf4(accuracy_backend, dim):
    implementation, _, device = accuracy_backend
    inputs = make_nf4_case(dim, device)
    actual = F.embedding_nf4_dequant(*inputs, group_size=64, output_dtype=torch.float32, implementation=implementation)
    expected = F.embedding_nf4_dequant(
        *inputs, group_size=64, output_dtype=torch.float32, implementation="torch_reference"
    )
    assert_close(actual, expected, rtol=0, atol=1e-5)


def _decode_case(device):
    config, weights, (ids, history, _) = make_encoding_case("nf4-decode", device)
    vocab, _, _, sizes, grams = config
    sizes, grams = (torch.tensor(x, device=device, dtype=torch.int64) for x in (sizes, grams))
    offsets = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
    inputs = (
        ids,
        history,
        sizes,
        offsets,
        grams,
        weights["_mega_embedding_weight"],
        weights["_mega_embedding_scale"],
        weights["_mega_embedding_mean"],
    )
    return inputs, dict(ori_vocab_size=vocab, group_size=64, output_dtype=torch.float32)


@pytest.mark.api("functions.over_encoding_decode")
@pytest.mark.accuracy
def test_decode(accuracy_backend):
    implementation, _, device = accuracy_backend
    inputs, options = _decode_case(device)
    actual = F.over_encoding_decode(*inputs, **options, implementation=implementation)
    expected = F.over_encoding_decode(*inputs, **options, implementation="torch_reference")
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
