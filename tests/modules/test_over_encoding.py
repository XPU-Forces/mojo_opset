import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

ENCODING_CASES = [
    "basic-prefill",
    "basic-decode",
    "wide-prefill",
    "wide-decode",
    "prime-prefill",
    "prime-decode",
    "cpu-prefill",
    "cpu-decode",
    "nf4-prefill",
    "nf4-decode",
]


NF4_DIMS = [128, 1024, 4096]


NGRAM_CASES = [
    "basic-prefill",
    "basic-decode",
    "history-decode",
    "prime-prefill",
    "varlen-97",
    "varlen-193",
    "varlen-257",
]


def make_nf4_lut(vocab, dim, device):
    indices = torch.randint(16, (vocab, dim), device="cpu", dtype=torch.uint8)
    packed = (indices[:, ::2] | (indices[:, 1::2] << 4)).to(device=device, dtype=torch.int8)
    return packed, torch.randn(vocab, dim // 64, device=device), torch.randn(vocab, dim // 64, device=device)


def make_ngram_case(case, device):
    if case.startswith("basic") or case == "history-decode":
        vocab, sizes, grams = 10, [10000] * 6, [2, 2, 3, 3, 4, 4]
        ids = torch.arange(1, 6, device=device)
        if case.endswith("prefill"):
            lens = [5]
            history = [[1] * 16 + [0] * 3]
        else:
            ids, lens = ids[:, None], None
            history = [([1] * 16 if case == "history-decode" else []) + [1, 2, 3]] * 5
    elif case == "prime-prefill":
        vocab, sizes, grams = 257, [263, 269, 271, 277, 281, 283, 293, 307], [2, 2, 3, 3, 4, 4, 5, 5]
        ids = torch.tensor(
            [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89], device=device
        )
        lens, history = [5, 7, 9], [[3, 5, 7, 9], [2, 4, 6, 8], [11, 13, 17, 19]]
    else:
        vocab, sizes, grams, lens, history = {
            "varlen-97": (97, [101, 103, 107, 109], [2, 2, 3, 3], [4, 6], [[5, 7], [11, 13]]),
            "varlen-193": (
                193,
                [197, 199, 211, 223, 227, 229],
                [2, 2, 3, 3, 4, 4],
                [3, 5, 7],
                [[2, 3, 5], [7, 11, 13], [17, 19, 23]],
            ),
            "varlen-257": (
                257,
                [263, 269, 271, 277, 281, 283, 293, 307],
                [2, 2, 3, 3, 4, 4, 5, 5],
                [8, 9, 11],
                [[3, 5, 7, 9], [2, 4, 6, 8], [11, 13, 17, 19]],
            ),
        }[case]
        ids = torch.arange(1, sum(lens) + 1, device=device) * 3 % vocab
    sizes, grams, history = (torch.tensor(x, dtype=torch.int64, device=device) for x in (sizes, grams, history))
    offsets = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
    lens = torch.tensor(lens, device=device, dtype=torch.int64) if lens is not None else None
    return ids, history, lens, sizes, offsets, grams, vocab


def make_encoding_case(case, device):
    mode = case.rsplit("-", 1)[-1]
    kind = case.split("-", 1)[0]
    options = {}
    if kind == "basic":
        vocab, embed_dim, oe_dim, sizes, grams = 10, 128, 21, [10] * 6, [2, 2, 3, 3, 4, 4]
        ids = torch.arange(1, 7, device=device)
        lens = torch.tensor([3, 3], device=device) if mode == "prefill" else None
        history = torch.zeros(
            2 if lens is not None else 6, 3 if lens is not None else 4, device=device, dtype=torch.int64
        )
    elif kind in ("wide", "cpu"):
        vocab, embed_dim, oe_dim = 10086, 1536, 192
        sizes, grams = [10086 + 2**i for i in range(12)], [i for i in range(2, 8) for _ in range(2)]
        ids = torch.arange(1, 129, device=device)
        lens = torch.tensor([64, 64], device=device) if mode == "prefill" else None
        history = torch.ones(2 if lens is not None else 128, 6, device=device, dtype=torch.int64)
        if kind == "cpu":
            # Explicit custom weights, not uninitialized storage as in the old test.
            options = dict(
                _ori_embedding_weight=torch.randn(vocab, embed_dim, device=device),
                _mega_embedding_weight=torch.randn(sum(sizes), oe_dim, device="cpu"),
                mega_embedding_cpu_only=True,
            )
    elif kind == "prime":
        vocab, sizes, grams = 257, [263, 269, 271, 277, 281, 283, 293, 307], [2, 2, 3, 3, 4, 4, 5, 5]
        if mode == "prefill":
            embed_dim, oe_dim = 640, 80
            ids, history, lens, *_ = make_ngram_case("prime-prefill", device)
        else:
            embed_dim, oe_dim = 320, 40
            ids, lens = torch.arange(1, 49, device=device) % vocab, None
            history = torch.arange(48 * 4, device=device).view(48, 4) % 17
    else:
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


@pytest.mark.api("modules.OverEncodingNGram")
@pytest.mark.parametrize("case", NGRAM_CASES)
@pytest.mark.accuracy
def test_ngram(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    ids, history, lens, sizes, _, grams, vocab = make_ngram_case(case, device)
    module = modules.OverEncodingNGram(vocab, sizes, grams, implementation=implementation).to(device)
    reference = modules.OverEncodingNGram(vocab, sizes, grams, implementation="torch_reference").to(device)
    assert_close(module(ids, history, lens), reference(ids, history, lens), rtol=0, atol=0)


@pytest.mark.api("modules.NF4DequantEmbedding")
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


@pytest.mark.api("modules.OverEncoding")
@pytest.mark.parametrize("case", ENCODING_CASES)
@pytest.mark.accuracy
def test_encoding(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    actual, reference, inputs = _encoding_modules(case, device, implementation)
    with torch.no_grad():
        assert_close(actual(*inputs), reference(*inputs), rtol=1e-5, atol=1e-5)
