import pytest
import torch

from mojo_opset import modules


@pytest.mark.parametrize(
    "embedding,case",
    [
        pytest.param(
            embedding,
            case,
            marks=pytest.mark.api(
                "modules.OverEncoding" if embedding else "modules.OverEncodingNGram",
                ops=["n_gram_prefill" if case.endswith("prefill") else "n_gram_decode"],
            ),
        )
        for embedding in (False, True)
        for case in ("prefill", "decode", "prime-prefill", "prime-decode")
    ],
)
def test_over_encoding(benchmark, perf_environment, case, embedding):
    _, device, _, implementation = perf_environment
    prefill = case.endswith("prefill")
    op = "n_gram_prefill" if prefill else "n_gram_decode"

    def factory():
        if case.startswith("prime"):
            vocab, sizes, grams = 257, [263, 269, 271, 277, 281, 283, 293, 307], [2, 2, 3, 3, 4, 4, 5, 5]
            if prefill:
                ids = torch.tensor(
                    [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89], device=device
                )
                history = torch.tensor([[3, 5, 7, 9], [2, 4, 6, 8], [11, 13, 17, 19]], dtype=torch.int32, device=device)
                lens = torch.tensor([5, 7, 9], dtype=torch.int32, device=device)
                dim, oe_dim = 640, 80
            else:
                ids = (torch.arange(1, 49, device=device) % 257).view(48, 1)
                history = torch.arange(48 * 4, dtype=torch.int32, device=device).view(48, 4) % 17
                lens, dim, oe_dim = None, 320, 40
        else:
            vocab = 10086
            sizes, grams = [10086 + 2**i for i in range(12)], [i for i in range(2, 8) for _ in range(2)]
            ids = torch.arange(1, 129, device=device)
            if not prefill:
                ids = ids.view(128, 1)
            history = torch.ones(2 if prefill else 128, 6, dtype=torch.int32, device=device)
            lens = torch.tensor([64, 64], dtype=torch.int32, device=device) if prefill else None
            dim, oe_dim = 1536, 192
        if embedding:
            module = modules.OverEncoding(
                vocab, dim, oe_dim, sizes, grams, device=device, implementation=implementation
            ).requires_grad_(False)
        else:
            module = modules.OverEncodingNGram(vocab, sizes, grams, device=device, implementation=implementation)
        return lambda: module(ids, history, lens)

    benchmark(factory=factory, op=op, case=case, embedding=embedding, dtype="int64", phase="forward")


@pytest.mark.api("modules.NF4DequantEmbedding", ops=["embedding_nf4_dequant"])
@pytest.mark.parametrize("dim", [1024, 4096])
def test_nf4_dequant_embedding(benchmark, perf_environment, dim):
    _, device, _, implementation = perf_environment

    def factory():
        vocab, group_size = 4096, 64
        indices = torch.randint(16, (vocab, dim), device="cpu", dtype=torch.uint8)
        qweight = (indices[:, ::2] | (indices[:, 1::2] << 4)).to(device=device, dtype=torch.int8)
        scale = torch.randn(vocab, dim // group_size, device=device, dtype=torch.float32)
        mean = torch.randn(vocab, dim // group_size, device=device, dtype=torch.float32)
        module = modules.NF4DequantEmbedding(
            qweight,
            scale,
            mean,
            group_size=group_size,
            output_dtype=torch.bfloat16,
            implementation=implementation,
        )
        ids = torch.randint(vocab, (32, 128), device=device, dtype=torch.int64)
        return lambda: module(ids)

    benchmark(
        factory=factory,
        op="embedding_nf4_dequant",
        vocab=4096,
        tokens=4096,
        dim=dim,
        group_size=64,
        dtype="bfloat16",
        phase="forward",
    )


@pytest.mark.api("modules.OverEncoding", ops=["over_encoding_decode"])
@pytest.mark.parametrize("batch,oe_dim", [(48, 64), (128, 192)])
def test_over_encoding_decode(benchmark, perf_environment, batch, oe_dim):
    _, device, _, implementation = perf_environment

    def factory():
        vocab, dim, group_size = 10086, 1536, 64
        sizes = [10086 + 2**i for i in range(8)]
        grams = [2, 2, 3, 3, 4, 4, 5, 5]
        total_vocab = sum(sizes)
        indices = torch.randint(16, (total_vocab, oe_dim), device="cpu", dtype=torch.uint8)
        qweight = (indices[:, ::2] | (indices[:, 1::2] << 4)).to(device=device, dtype=torch.int8)
        scale = torch.randn(total_vocab, oe_dim // group_size, device=device, dtype=torch.float32)
        mean = torch.randn(total_vocab, oe_dim // group_size, device=device, dtype=torch.float32)
        module = modules.OverEncoding(
            vocab,
            dim,
            oe_dim,
            sizes,
            grams,
            _mega_embedding_weight=qweight,
            _mega_embedding_scale=scale,
            _mega_embedding_mean=mean,
            _mega_embedding_group_size=group_size,
            device=device,
            dtype=torch.bfloat16,
            implementation=implementation,
        ).requires_grad_(False)
        ids = (torch.arange(1, batch + 1, device=device, dtype=torch.int64) % vocab).view(batch, 1)
        history = torch.ones(batch, 4, dtype=torch.int32, device=device)
        return lambda: module(ids, history)

    benchmark(
        factory=factory,
        op="over_encoding_decode",
        batch=batch,
        oe_dim=oe_dim,
        group_size=64,
        dtype="bfloat16",
        phase="forward",
    )

