import pytest
import torch

from triton.testing import assert_close

from mojo_opset import MojoOverEncoding, MojoOverEncodingNGram
from mojo_opset.backends.ttx.kernels import embedding_nf4_dequant
from mojo_opset.tests.utils import bypass_not_implemented, get_torch_device
from mojo_opset.core.operators.over_encoding import (
    n_gram_impl_torch,
    dequantize_nf4_rows,
)

TEST_DEVICE = get_torch_device()


def pack_nf4_uint4_to_int8(q_idx: torch.Tensor) -> torch.Tensor:
    if q_idx.ndim != 2:
        raise ValueError(f"`q_idx` must be 2D, got shape={tuple(q_idx.shape)}.")
    if q_idx.size(1) % 2 != 0:
        raise ValueError(f"`q_idx` width must be even, got {q_idx.size(1)}.")

    q_idx = q_idx.to(torch.uint8)
    low = q_idx[:, 0::2]
    high = q_idx[:, 1::2]
    return (low | (high << 4)).to(torch.int8)


def build_nf4_embedding_lut(
    vocab_size: int,
    embedding_dim: int,
    group_size: int,
    *,
    device: torch.device | str,
):
    if embedding_dim % group_size != 0:
        raise ValueError(
            f"`embedding_dim` must be divisible by `group_size`, got {embedding_dim} and {group_size}."
        )

    q_idx = torch.randint(0, 16, (vocab_size, embedding_dim), dtype=torch.uint8)
    qweight = pack_nf4_uint4_to_int8(q_idx).to(device)
    scale = torch.randn(
        vocab_size,
        embedding_dim // group_size,
        dtype=torch.float32,
        device=device,
    )
    mean = torch.randn(
        vocab_size,
        embedding_dim // group_size,
        dtype=torch.float32,
        device=device,
    )
    return qweight, scale, mean


class TestRefOverEncodingBasic:
    def setup_method(self):
        self.N = 4
        self.K = 2
        self.VOCAB_SIZE = 10
        self.EMBED_DIM = 128
        self.SPLIT_NUM = (self.N - 1) * self.K
        self.OE_EMBED_DIM = self.EMBED_DIM // self.SPLIT_NUM

    @bypass_not_implemented
    def test_n_gram_encoding(self):
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        oe_history = torch.cat(
            (torch.ones(5, 16, dtype=torch.int64), torch.zeros(5, self.N - 1, dtype=torch.int64)),
            dim=-1,
        )
        oe_vocab_sizes = torch.tensor(
            [10**4 for _ in range(self.SPLIT_NUM)], dtype=torch.int64
        )
        oe_vocab_offsets = torch.tensor(
            [0 for _ in range(self.SPLIT_NUM)], dtype=torch.long
        )
        n_grams = torch.tensor(
            [i for i in range(2, self.N + 1) for _ in range(self.K)],
            dtype=torch.int64,
        )

        oe_ngram = MojoOverEncodingNGram(self.VOCAB_SIZE, oe_vocab_sizes, n_grams)
        oe_ngram_ref = MojoOverEncodingNGram._registry.get("torch")(self.VOCAB_SIZE, oe_vocab_sizes, n_grams)

        n_gram_goldens = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [12, 12, 12, 12, 12, 12],
                [23, 23, 123, 123, 123, 123],
                [34, 34, 234, 234, 1234, 1234],
                [45, 45, 345, 345, 2345, 2345],
            ],
            dtype=torch.int64,
        )
        # NOTE(liuyuan): make sure that RefOverEncoding is reliable.
        assert_close(
            n_gram_impl_torch(
                input_ids,
                oe_history[1],
                oe_vocab_sizes,
                oe_vocab_offsets,
                n_grams,
                self.VOCAB_SIZE,
            ),
            n_gram_goldens,
            atol=0,
            rtol=0,
        )

        input_ids = input_ids.to(TEST_DEVICE)
        input_seq_len = torch.Tensor([5]).to(torch.int64).to(TEST_DEVICE)
        oe_history = oe_history.to(TEST_DEVICE)

        oe_ngram = oe_ngram.to(TEST_DEVICE)
        oe_ngram_ref = oe_ngram_ref.to(TEST_DEVICE)

        oe_ngram.forward_diff_with(oe_ngram_ref, input_ids, oe_history[:1], input_seq_len, atol=0, rtol=0)
        oe_history = torch.stack(
            [torch.arange(1, self.N) for _ in range(input_ids.size(0))], dim=0
        ).to(TEST_DEVICE)
        # goldens = torch.Tensor(
        #     [
        #         [31, 31, 231, 231, 1231, 1231],
        #         [32, 32, 232, 232, 1232, 1232],
        #         [33, 33, 233, 233, 1233, 1233],
        #         [34, 34, 234, 234, 1234, 1234],
        #         [35, 35, 235, 235, 1235, 1235],
        #     ],
        # ).to(TEST_DEVICE)
        oe_ngram.forward_diff_with(oe_ngram_ref, input_ids.unsqueeze(-1), oe_history, atol=0, rtol=0)
        oe_history = torch.cat(
            (
                torch.ones(
                    oe_history.size(0),
                    16,
                    dtype=oe_history.dtype,
                    device=oe_history.device,
                ),
                oe_history,
            ),
            dim=1,
        )
        oe_ngram.forward_diff_with(oe_ngram_ref, input_ids.unsqueeze(-1), oe_history, atol=0, rtol=0)

    @bypass_not_implemented
    def test_n_gram_encoding_additional_shapes(self):
        vocab_size = 257
        oe_vocab_sizes = torch.tensor(
            [263, 269, 271, 277, 281, 283, 293, 307],
            dtype=torch.int64,
        )
        oe_vocab_offsets = torch.tensor(
            [0, 263, 532, 803, 1080, 1361, 1644, 1937],
            dtype=torch.long,
        )
        n_grams = torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int64)
        input_ids = torch.tensor(
            [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
            dtype=torch.int64,
        )
        input_seq_lens = torch.tensor([5, 7, 9], dtype=torch.int64)
        oe_history = torch.tensor(
            [
                [3, 5, 7, 9],
                [2, 4, 6, 8],
                [11, 13, 17, 19],
            ],
            dtype=torch.int64,
        )

        oe_ngram = MojoOverEncodingNGram(vocab_size, oe_vocab_sizes, n_grams)
        oe_ngram_ref = MojoOverEncodingNGram._registry.get("torch")(vocab_size, oe_vocab_sizes, n_grams)

        seq_offset = 0
        golden = []
        for seq_idx, seq_len in enumerate(input_seq_lens.tolist()):
            seq_input_ids = input_ids[seq_offset : seq_offset + seq_len]
            golden.append(
                n_gram_impl_torch(
                    seq_input_ids,
                    oe_history[seq_idx],
                    oe_vocab_sizes,
                    oe_vocab_offsets,
                    n_grams,
                    vocab_size,
                )
            )
            seq_offset += seq_len
        golden = torch.cat(golden, dim=0)

        assert_close(
            oe_ngram_ref(input_ids, oe_history, input_seq_lens),
            golden,
            atol=0,
            rtol=0,
        )

        oe_ngram = oe_ngram.to(TEST_DEVICE)
        oe_ngram_ref = oe_ngram_ref.to(TEST_DEVICE)
        oe_ngram.forward_diff_with(
            oe_ngram_ref,
            input_ids.to(TEST_DEVICE),
            oe_history.to(TEST_DEVICE),
            input_seq_lens.to(TEST_DEVICE),
            atol=0,
            rtol=0,
        )

    @bypass_not_implemented
    def test_over_encoding(self):
        input_ids = torch.Tensor([1, 2, 3, 4, 5, 6]).to(torch.int64).to(TEST_DEVICE)
        input_seq_len = torch.Tensor([3, 3]).to(torch.int64).to(TEST_DEVICE)
        oe_history = torch.zeros(2, 3, device=TEST_DEVICE).to(torch.int64)

        @torch.no_grad
        def init_weight(m):
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                m.weight.copy_(
                    torch.arange(
                        m.weight.size(0), device=m.weight.device, dtype=m.weight.dtype
                    ).view(-1, 1).broadcast_to(m.weight.shape)
                )

        oe_layer = MojoOverEncoding._registry.get("torch")(
            self.VOCAB_SIZE,
            self.EMBED_DIM,
            self.OE_EMBED_DIM,
            [self.VOCAB_SIZE for _ in range(self.SPLIT_NUM)],
            [i for i in range(2, self.N + 1) for _ in range(self.K)],
            dtype=torch.int64,
        ).to(TEST_DEVICE)
        oe_layer.apply(init_weight)

        ttx_oe_layer = MojoOverEncoding(
            self.VOCAB_SIZE,
            self.EMBED_DIM,
            self.OE_EMBED_DIM,
            [self.VOCAB_SIZE for _ in range(self.SPLIT_NUM)],
            [i for i in range(2, self.N + 1) for _ in range(self.K)],
            dtype=torch.int64,
        ).to(TEST_DEVICE)
        ttx_oe_layer.apply(init_weight)

        # NOTE(liuyuan): Test Prefill
        ref = oe_layer(input_ids, oe_history, input_seq_len)
        ttx_res = ttx_oe_layer(input_ids, oe_history, input_seq_len)
        assert_close(ttx_res, ref)

        # NOTE(liuyuan): Test Decode
        oe_history = torch.zeros(
            input_ids.size(0), self.N, device=TEST_DEVICE, dtype=torch.int64
        )
        input_ids = input_ids.reshape(-1, 1)
        ref = oe_layer(input_ids, oe_history)
        ttx_res = ttx_oe_layer(input_ids, oe_history)
        assert_close(ttx_res, ref)

        # FIXME(liuyuan): The Triton kernel still suffers from random partial errors caused by Byted-Triton-X. We will re-run this test case once the fix is complete.
        # input_ids = input_ids.reshape(-1, 2)
        # oe_history = torch.zeros(
        #     input_ids.size(0), self.N, device=TEST_DEVICE, dtype=torch.int64
        # )
        # ttx_res = ttx_oe_layer(input_ids, oe_history)
        # ref = oe_layer(input_ids, oe_history)
        # assert_close(ttx_res, ref)


class TestRefOverEncodingParametrized:
    @staticmethod
    @torch.no_grad
    def init_weight(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            m.weight.fill_(2.0)
            m.weight.fill_diagonal_(1.0)

    @staticmethod
    def move_case_to_test_device(
        input_ids,
        seq_lens,
        oe_history_inputs,
        oe_vocab_sizes,
        n_grams,
    ):
        input_ids = input_ids.to(TEST_DEVICE)
        if seq_lens is not None:
            seq_lens = seq_lens.to(TEST_DEVICE)
        oe_history_inputs = oe_history_inputs.to(TEST_DEVICE)
        oe_vocab_sizes = oe_vocab_sizes.to(TEST_DEVICE)
        n_grams = n_grams.to(TEST_DEVICE)
        return input_ids, seq_lens, oe_history_inputs, oe_vocab_sizes, n_grams

    @pytest.mark.parametrize(
        "input_ids,seq_lens,oe_history_inputs",
        (
            (
                torch.arange(1, 129, dtype=torch.long),
                torch.tensor([64, 64], dtype=torch.int64),
                torch.ones(2, 6, dtype=torch.int64),
            ),
            (
                torch.arange(1, 129, dtype=torch.long).view(128, 1),
                None,
                torch.ones(128, 6, dtype=torch.int64),
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
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [i for i in range(2, 8) for _ in range(2)],
                    dtype=torch.int64,
                ),
            ),
            # (
            #     2**31,
            #     torch.tensor(
            #         [2**31 + 2**i for i in range(2)],
            #         dtype=torch.long,
            #         device=TEST_DEVICE,
            #     ),
            #     torch.tensor(
            #         [i for i in range(2, 3) for _ in range(2)],
            #         dtype=torch.long,
            #         device=TEST_DEVICE,
            #     ),
            # ),
        ),
    )
    @pytest.mark.parametrize(
        "embed_dim, oe_embed_dims",
        ((1536, 192),),
    )
    @bypass_not_implemented
    def test_over_encoding_parametrized(
        self,
        input_ids,
        seq_lens,
        oe_history_inputs,
        vocab_size,
        oe_vocab_sizes,
        n_grams,
        embed_dim,
        oe_embed_dims,
    ):
        input_ids, seq_lens, oe_history_inputs, oe_vocab_sizes, n_grams = self.move_case_to_test_device(
            input_ids,
            seq_lens,
            oe_history_inputs,
            oe_vocab_sizes,
            n_grams,
        )

        ref_oe_layer = MojoOverEncoding._registry.get("torch")(
            vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams
        ).to(TEST_DEVICE)
        ref_oe_layer.apply(self.init_weight)

        ttx_oe_layer = MojoOverEncoding(
            vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams
        ).to(TEST_DEVICE)
        ttx_oe_layer.apply(self.init_weight)

        assert_close(
            ttx_oe_layer(input_ids, oe_history_inputs, seq_lens),
            ref_oe_layer(input_ids, oe_history_inputs, seq_lens),
        )

    @pytest.mark.parametrize(
        "input_ids,seq_lens,oe_history_inputs,vocab_size,oe_vocab_sizes,n_grams,embed_dim,oe_embed_dims",
        (
            (
                torch.tensor(
                    [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
                    dtype=torch.long,
                ),
                torch.tensor([5, 7, 9], dtype=torch.int64),
                torch.tensor(
                    [
                        [3, 5, 7, 9],
                        [2, 4, 6, 8],
                        [11, 13, 17, 19],
                    ],
                    dtype=torch.int64,
                ),
                257,
                torch.tensor(
                    [263, 269, 271, 277, 281, 283, 293, 307],
                    dtype=torch.int64,
                ),
                torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int64),
                640,
                80,
            ),
            (
                (torch.arange(1, 49, dtype=torch.long) % 257).view(48, 1),
                None,
                (
                    torch.arange(48 * 4, dtype=torch.int64).view(48, 4) % 17
                ),
                257,
                torch.tensor(
                    [263, 269, 271, 277, 281, 283, 293, 307],
                    dtype=torch.int64,
                ),
                torch.tensor([2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int64),
                320,
                40,
            ),
        ),
        ids=("prefill-b3-len5-7-9", "decode-b48-s1"),
    )
    @bypass_not_implemented
    def test_over_encoding_additional_shapes(
        self,
        input_ids,
        seq_lens,
        oe_history_inputs,
        vocab_size,
        oe_vocab_sizes,
        n_grams,
        embed_dim,
        oe_embed_dims,
    ):
        input_ids, seq_lens, oe_history_inputs, oe_vocab_sizes, n_grams = self.move_case_to_test_device(
            input_ids,
            seq_lens,
            oe_history_inputs,
            oe_vocab_sizes,
            n_grams,
        )

        ref_oe_layer = MojoOverEncoding._registry.get("torch")(
            vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams
        ).to(TEST_DEVICE)
        ref_oe_layer.apply(self.init_weight)

        ttx_oe_layer = MojoOverEncoding(
            vocab_size, embed_dim, oe_embed_dims, oe_vocab_sizes, n_grams
        ).to(TEST_DEVICE)
        ttx_oe_layer.apply(self.init_weight)

        assert_close(
            ttx_oe_layer(input_ids, oe_history_inputs, seq_lens),
            ref_oe_layer(input_ids, oe_history_inputs, seq_lens),
            atol=1e-5,
            rtol=1e-5,
        )

    @bypass_not_implemented
    def test_embedding_nf4_dequant_impl(self):
        vocab_size = 257
        embedding_dim = 128
        group_size = 64
        qweight, scale, mean = build_nf4_embedding_lut(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            group_size=group_size,
            device=TEST_DEVICE,
        )
        dequant_lut = dequantize_nf4_rows(
            qweight,
            scale,
            mean,
            group_size=group_size,
            output_dtype=torch.float32,
        )
        input_ids = torch.tensor(
            [[0, 17, 32], [128, 256, vocab_size]],
            dtype=torch.long,
            device=TEST_DEVICE,
        )

        expected = torch.zeros(
            (*input_ids.shape, embedding_dim),
            dtype=torch.float32,
            device=TEST_DEVICE,
        )
        valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
        expected[valid_mask] = dequant_lut.index_select(0, input_ids[valid_mask])

        output = embedding_nf4_dequant(
            input_ids,
            qweight,
            scale,
            mean,
            group_size=group_size,
            output_dtype=torch.float32,
        )
        assert_close(output, expected, atol=1e-6, rtol=0)

    @bypass_not_implemented
    @torch.no_grad
    def test_over_encoding_with_quantized_mega_embedding(self):
        torch.manual_seed(0)

        vocab_size = 128
        oe_vocab_sizes = torch.tensor(
            [vocab_size + 3, vocab_size + 5, vocab_size + 7, vocab_size + 11],
            dtype=torch.int64,
            device=TEST_DEVICE,
        )
        n_grams = torch.tensor([2, 2, 3, 3], dtype=torch.int64, device=TEST_DEVICE)
        embed_dim = 64
        oe_embed_dim = 64
        group_size = 64
        mega_vocab_size = oe_vocab_sizes.sum().item()

        ori_embedding_lut = torch.randn(vocab_size, embed_dim)
        qweight, scale, mean = build_nf4_embedding_lut(
            vocab_size=mega_vocab_size,
            embedding_dim=oe_embed_dim,
            group_size=group_size,
            device="cpu",
        )
        ref_oe_layer = MojoOverEncoding._registry.get("torch")(
            vocab_size,
            embed_dim,
            oe_embed_dim,
            oe_vocab_sizes,
            n_grams,
            _ori_embedding_weight=ori_embedding_lut,
            _mega_embedding_weight=qweight,
            _mega_embedding_scale=scale,
            _mega_embedding_mean=mean,
            _mega_embedding_group_size=group_size,
        ).to(TEST_DEVICE)
        ttx_oe_layer = MojoOverEncoding(
            vocab_size,
            embed_dim,
            oe_embed_dim,
            oe_vocab_sizes,
            n_grams,
            _ori_embedding_weight=ori_embedding_lut,
            _mega_embedding_weight=qweight,
            _mega_embedding_scale=scale,
            _mega_embedding_mean=mean,
            _mega_embedding_group_size=group_size,
        ).to(TEST_DEVICE)

        proj_weight = torch.randn_like(ref_oe_layer.oe_up_proj.weight)
        ref_oe_layer.oe_up_proj.weight.copy_(proj_weight)
        ttx_oe_layer.oe_up_proj.weight.copy_(proj_weight)

        SEQ_NUM = 32
        prefill_input_ids = (
            torch.arange(1, SEQ_NUM + 1, dtype=torch.long, device=TEST_DEVICE)
            .broadcast_to(SEQ_NUM, SEQ_NUM)
            .flatten()
        )
        prefill_seq_lens = torch.tensor(
            [SEQ_NUM] * SEQ_NUM, dtype=torch.int64, device=TEST_DEVICE
        )
        prefill_history = torch.zeros(SEQ_NUM, 2, dtype=torch.int64, device=TEST_DEVICE)

        decode_input_ids = torch.arange(1, 17, dtype=torch.long, device=TEST_DEVICE).view(-1, 1)
        decode_history = torch.zeros(
            decode_input_ids.size(0),
            2,
            dtype=torch.int64,
            device=TEST_DEVICE,
        )

        assert_close(
            ref_oe_layer(prefill_input_ids, prefill_history, prefill_seq_lens),
            ttx_oe_layer(prefill_input_ids, prefill_history, prefill_seq_lens),
            atol=1e-5,
            rtol=1e-5,
        )
        assert_close(
            ref_oe_layer(decode_input_ids, decode_history),
            ttx_oe_layer(decode_input_ids, decode_history),
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize(
        "input_ids,seq_lens,oe_history_inputs",
        (
            (
                torch.arange(1, 129, dtype=torch.long),
                torch.tensor([64, 64], dtype=torch.int64),
                torch.ones(2, 6, dtype=torch.int64),
            ),
            (
                torch.arange(1, 129, dtype=torch.long).view(128, 1),
                None,
                torch.ones(128, 6, dtype=torch.int64),
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
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [i for i in range(2, 8) for _ in range(2)],
                    dtype=torch.int64,
                ),
            ),
            # (
            #     2**31,
            #     torch.tensor(
            #         [2**31 + 2**i for i in range(2)],
            #         dtype=torch.long,
            #         device=TEST_DEVICE,
            #     ),
            #     torch.tensor(
            #         [i for i in range(2, 3) for _ in range(2)],
            #         dtype=torch.long,
            #         device=TEST_DEVICE,
            #     ),
            # ),
        ),
    )
    @pytest.mark.parametrize(
        "embed_dim, oe_embed_dims",
        ((1536, 192),),
    )
    @bypass_not_implemented
    def test_over_encoding_with_custom_weight_tensor(
        self,
        input_ids,
        seq_lens,
        oe_history_inputs,
        vocab_size,
        oe_vocab_sizes,
        n_grams,
        embed_dim,
        oe_embed_dims,
    ):
        input_ids, seq_lens, oe_history_inputs, oe_vocab_sizes, n_grams = self.move_case_to_test_device(
            input_ids,
            seq_lens,
            oe_history_inputs,
            oe_vocab_sizes,
            n_grams,
        )

        ori_embedding_lut = torch.empty(vocab_size, embed_dim)
        mega_embedding_lut = torch.empty(int(oe_vocab_sizes.sum().item()), oe_embed_dims)

        @torch.no_grad
        def init_weight(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.fill_(2.0)
                m.weight.fill_diagonal_(1.0)

        ref_oe_layer = MojoOverEncoding._registry.get("torch")(
            vocab_size,
            embed_dim,
            oe_embed_dims,
            oe_vocab_sizes,
            n_grams,
            _ori_embedding_weight=ori_embedding_lut,
            _mega_embedding_weight=mega_embedding_lut,
        ).to(TEST_DEVICE)
        ref_oe_layer.apply(init_weight)

        ttx_oe_layer = MojoOverEncoding(
            vocab_size,
            embed_dim,
            oe_embed_dims,
            oe_vocab_sizes,
            n_grams,
            _ori_embedding_weight=ori_embedding_lut,
            _mega_embedding_weight=mega_embedding_lut.cpu(),
            mega_embedding_cpu_only=True,
        ).to(TEST_DEVICE)
        ttx_oe_layer.apply(init_weight)

        assert_close(
            ttx_oe_layer(input_ids, oe_history_inputs, seq_lens),
            ref_oe_layer(input_ids, oe_history_inputs, seq_lens),
        )
