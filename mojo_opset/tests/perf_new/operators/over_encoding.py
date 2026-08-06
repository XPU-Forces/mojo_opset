"""Over-encoding performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoOverEncoding
from mojo_opset import MojoOverEncodingNGram
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor



_LARGE_VOCABS = [10086 + 2**index for index in range(12)]
_LARGE_GRAMS = [value for value in range(2, 8) for _ in range(2)]
_SMALL_VOCABS = [263, 269, 271, 277, 281, 283, 293, 307]
_SMALL_GRAMS = [2, 2, 3, 3, 4, 4, 5, 5]

CASES = (
    perf_case(
        "prefill_b2_len64",
        tags=("full",),
        mode="arange",
        input_shape=[128],
        seq_lens=[64, 64],
        history_shape=[2, 6],
        history_mode="ones",
        vocab_size=10086,
        oe_vocab_sizes=_LARGE_VOCABS,
        n_grams=_LARGE_GRAMS,
        embed_dim=1536,
        oe_embed_dim=192,
    ),
    perf_case(
        "decode_b128",
        tags=("full",),
        mode="arange",
        input_shape=[128, 1],
        seq_lens=None,
        history_shape=[128, 6],
        history_mode="ones",
        vocab_size=10086,
        oe_vocab_sizes=_LARGE_VOCABS,
        n_grams=_LARGE_GRAMS,
        embed_dim=1536,
        oe_embed_dim=192,
    ),
    perf_case(
        "prefill_b3_len5_7_9",
        tags=("smoke", "full"),
        mode="primes",
        input_shape=[21],
        seq_lens=[5, 7, 9],
        history_shape=[3, 4],
        history_mode="fixed",
        vocab_size=257,
        oe_vocab_sizes=_SMALL_VOCABS,
        n_grams=_SMALL_GRAMS,
        embed_dim=640,
        oe_embed_dim=80,
    ),
    perf_case(
        "decode_b48_s1",
        tags=("full",),
        mode="mod257",
        input_shape=[48, 1],
        seq_lens=None,
        history_shape=[48, 4],
        history_mode="mod17",
        vocab_size=257,
        oe_vocab_sizes=_SMALL_VOCABS,
        n_grams=_SMALL_GRAMS,
        embed_dim=320,
        oe_embed_dim=40,
    ),
)


_PRIMES = [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
_FIXED_HISTORY = [[3, 5, 7, 9], [2, 4, 6, 8], [11, 13, 17, 19]]


def _over_encoding_workload(case: Mapping[str, Any], *, with_embedding: bool) -> PerfWorkload:
    input_shape = tuple(int(value) for value in case["input_shape"])
    history_shape = tuple(int(value) for value in case["history_shape"])
    seq_values = case.get("seq_lens")
    inputs = {
        "input_ids": tensor(input_shape, torch.int64),
        "history": tensor(history_shape, torch.int32),
    }
    args: tuple[Any, ...] = ("input_ids", "history", None)
    if seq_values is not None:
        inputs["seq_lens"] = tensor((len(seq_values),), torch.int32)
        args = ("input_ids", "history", "seq_lens")

    def tensor_factory(device: str):
        count = 1
        for dimension in input_shape:
            count *= dimension
        if case["mode"] == "primes":
            input_ids = torch.tensor(_PRIMES, dtype=torch.int64, device=device)
        else:
            input_ids = torch.arange(1, count + 1, dtype=torch.int64, device=device)
            if case["mode"] == "mod257":
                input_ids %= 257
        input_ids = input_ids.reshape(input_shape)

        if case["history_mode"] == "ones":
            history = torch.ones(history_shape, dtype=torch.int32, device=device)
        elif case["history_mode"] == "fixed":
            history = torch.tensor(_FIXED_HISTORY, dtype=torch.int32, device=device)
        else:
            history = torch.arange(
                history_shape[0] * history_shape[1],
                dtype=torch.int32,
                device=device,
            ).reshape(history_shape)
            history %= 17
        mapping = {"input_ids": input_ids, "history": history}
        if seq_values is not None:
            mapping["seq_lens"] = torch.tensor(seq_values, dtype=torch.int32, device=device)
        return mapping

    op_kwargs = {
        "ori_vocab_size": int(case["vocab_size"]),
        "oe_vocab_sizes": torch.tensor(case["oe_vocab_sizes"], dtype=torch.int32),
        "oe_grams": torch.tensor(case["n_grams"], dtype=torch.int32),
    }
    if with_embedding:
        op_kwargs.update(
            ori_embed_dim=int(case["embed_dim"]),
            oe_embed_dim=int(case["oe_embed_dim"]),
        )
    return PerfWorkload(
        op_kwargs=op_kwargs,
        inputs=inputs,
        outputs={},
        args=args,
        tensor_factory=tensor_factory,
    )


@mojo_perf(name="mojo_over_encoding", target=MojoOverEncoding, cases=CASES)
def over_encoding_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _over_encoding_workload(case, with_embedding=True)


@mojo_perf(name="mojo_over_encoding_ngram", target=MojoOverEncodingNGram, cases=CASES)
def over_encoding_ngram_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _over_encoding_workload(case, with_embedding=False)
