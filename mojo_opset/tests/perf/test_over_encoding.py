import pytest
import torch

from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset import MojoOverEncoding
from mojo_opset import MojoOverEncodingNGram


@pytest.mark.parametrize(
    "input_ids,seq_lens,oe_history_inputs",
    (
        (
            torch.arange(1, 129, dtype=torch.long),
            torch.tensor([64, 64], dtype=torch.int),
            torch.ones(2, 6, dtype=torch.int),
        ),
        (
            torch.arange(1, 129, dtype=torch.long).view(
                128, 1
            ),
            None,
            torch.ones(128, 6, dtype=torch.int),
        ),
    ),
)
@pytest.mark.parametrize(
    "vocab_size,oe_vocab_sizes,n_grams",
    (
        (
            10086,
            torch.tensor(
                [10086 + 2**i for i in range(12)],
                dtype=torch.int,
            ),
            torch.tensor(
                [i for i in range(2, 8) for _ in range(2)],
                dtype=torch.int,
            ),
        ),
    ),
)
@pytest.mark.parametrize(
    "embed_dim, oe_embed_dims",
    ((1536, 192),),
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_over_encoding_parametrized(
    input_ids,
    seq_lens,
    oe_history_inputs,
    vocab_size,
    oe_vocab_sizes,
    n_grams,
    embed_dim,
    oe_embed_dims,
):

    ttx_oe_layer = MojoOverEncoding(vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams)
    ref_oe_layer = MojoOverEncoding._registry.get("torch")(
        vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams
    )
    if input_ids.device.type == "npu":
        ttx_oe_layer = ttx_oe_layer.npu()
        ref_oe_layer = ref_oe_layer.npu()
    elif input_ids.device.type == "cuda":
        ttx_oe_layer = ttx_oe_layer.cuda()
        ref_oe_layer = ref_oe_layer.cuda()
    perf(lambda: ttx_oe_layer(input_ids, oe_history_inputs, seq_lens))
    perf(lambda: ref_oe_layer(input_ids, oe_history_inputs, seq_lens))


@pytest.mark.parametrize(
    "input_ids,seq_lens,oe_history_inputs,vocab_size,oe_vocab_sizes,n_grams,embed_dim,oe_embed_dims",
    (
        (
            torch.tensor(
                [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
                dtype=torch.long,
            ),
            torch.tensor([5, 7, 9], dtype=torch.int),
            torch.tensor(
                [
                    [3, 5, 7, 9],
                    [2, 4, 6, 8],
                    [11, 13, 17, 19],
                ],
                dtype=torch.int,
            ),
            257,
            torch.tensor(
                [263, 269, 271, 277, 281, 283, 293, 307],
                dtype=torch.int,
            ),
            torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int),
            640,
            80,
        ),
        (
            (torch.arange(1, 49, dtype=torch.long) % 257).view(48, 1),
            None,
            (torch.arange(48 * 4, dtype=torch.int).view(48, 4) % 17),
            257,
            torch.tensor(
                [263, 269, 271, 277, 281, 283, 293, 307],
                dtype=torch.int,
            ),
            torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int),
            320,
            40,
        ),
    ),
    ids=("prefill-b3-len5-7-9", "decode-b48-s1"),
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_over_encoding_additional_shapes(
    input_ids,
    seq_lens,
    oe_history_inputs,
    vocab_size,
    oe_vocab_sizes,
    n_grams,
    embed_dim,
    oe_embed_dims,
):
    ttx_oe_layer = MojoOverEncoding(vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams)
    ref_oe_layer = MojoOverEncoding._registry.get("torch")(
        vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams
    )
    if input_ids.device.type == "npu":
        ttx_oe_layer = ttx_oe_layer.npu()
        ref_oe_layer = ref_oe_layer.npu()
    elif input_ids.device.type == "cuda":
        ttx_oe_layer = ttx_oe_layer.cuda()
        ref_oe_layer = ref_oe_layer.cuda()
    perf(lambda: ttx_oe_layer(input_ids, oe_history_inputs, seq_lens))
    perf(lambda: ref_oe_layer(input_ids, oe_history_inputs, seq_lens))

@pytest.mark.parametrize(
    "input_ids,seq_lens,oe_history_inputs",
    (
        (
            torch.arange(1, 129, dtype=torch.long),
            torch.tensor([64, 64], dtype=torch.int),
            torch.ones(2, 6, dtype=torch.int),
        ),
        (
            torch.arange(1, 129, dtype=torch.long).view(
                128, 1
            ),
            None,
            torch.ones(128, 6, dtype=torch.int),
        ),
    ),
)
@pytest.mark.parametrize(
    "vocab_size,oe_vocab_sizes,n_grams",
    (
        (
            10086,
            torch.tensor(
                [10086 + 2**i for i in range(12)],
                dtype=torch.int,
            ),
            torch.tensor(
                [i for i in range(2, 8) for _ in range(2)],
                dtype=torch.int,
            ),
        ),
    ),
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_n_gram_parametrized(
    input_ids,
    seq_lens,
    oe_history_inputs,
    vocab_size,
    oe_vocab_sizes,
    n_grams,
):
    ttx_ngram = MojoOverEncodingNGram(vocab_size, oe_vocab_sizes, n_grams)
    ref_ngram = MojoOverEncodingNGram._registry.get("torch")(vocab_size, oe_vocab_sizes, n_grams)
    if input_ids.device.type == "npu":
        ttx_ngram = ttx_ngram.npu()
        ref_ngram = ref_ngram.npu()
    elif input_ids.device.type == "cuda":
        ttx_ngram = ttx_ngram.cuda()
        ref_ngram = ref_ngram.cuda()
    perf(lambda: ttx_ngram(input_ids, oe_history_inputs, seq_lens))
    perf(lambda: ref_ngram(input_ids, oe_history_inputs, seq_lens))


@pytest.mark.parametrize(
    "input_ids,seq_lens,oe_history_inputs,vocab_size,oe_vocab_sizes,n_grams",
    (
        (
            torch.tensor(
                [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
                dtype=torch.long,
            ),
            torch.tensor([5, 7, 9], dtype=torch.int),
            torch.tensor(
                [
                    [3, 5, 7, 9],
                    [2, 4, 6, 8],
                    [11, 13, 17, 19],
                ],
                dtype=torch.int,
            ),
            257,
            torch.tensor(
                [263, 269, 271, 277, 281, 283, 293, 307],
                dtype=torch.int,
            ),
            torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int),
        ),
        (
            (torch.arange(1, 49, dtype=torch.long) % 257).view(48, 1),
            None,
            (torch.arange(48 * 4, dtype=torch.int).view(48, 4) % 17),
            257,
            torch.tensor(
                [263, 269, 271, 277, 281, 283, 293, 307],
                dtype=torch.int,
            ),
            torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int),
        ),
    ),
    ids=("prefill-b3-len5-7-9", "decode-b48-s1"),
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_n_gram_additional_shapes(
    input_ids,
    seq_lens,
    oe_history_inputs,
    vocab_size,
    oe_vocab_sizes,
    n_grams,
):
    ttx_ngram = MojoOverEncodingNGram(vocab_size, oe_vocab_sizes, n_grams)
    ref_ngram = MojoOverEncodingNGram._registry.get("torch")(vocab_size, oe_vocab_sizes, n_grams)
    if input_ids.device.type == "npu":
        ttx_ngram = ttx_ngram.npu()
        ref_ngram = ref_ngram.npu()
    elif input_ids.device.type == "cuda":
        ttx_ngram = ttx_ngram.cuda()
        ref_ngram = ref_ngram.cuda()
    perf(lambda: ttx_ngram(input_ids, oe_history_inputs, seq_lens))
    perf(lambda: ref_ngram(input_ids, oe_history_inputs, seq_lens))

