from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import embedding_nf4_dequant
from mojo_opset.backends.ttx.kernels import n_gram_decode
from mojo_opset.backends.ttx.kernels import n_gram_prefill
from mojo_opset.backends.ttx.kernels import over_encoding_decode
from mojo_opset.core import MojoOverEncoding
from mojo_opset.core import MojoOverEncodingNGram
from mojo_opset.core import MojoNF4DequantEmbedding

class TTXOverEncodingNGram(MojoOverEncodingNGram):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, input_tensor: torch.Tensor, oe_history_input: torch.Tensor, input_seq_lens: Optional[torch.Tensor] = None):
        if input_seq_lens is not None:
            assert input_tensor.dim() == 1  # [total_tokens]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_seq_lens.size(0) # [batch_size, max_n_gram - 1]
            oe_ngram_ids = n_gram_prefill(
                input_ids=input_tensor,
                seq_lens=input_seq_lens,
                oe_history_inputs=oe_history_input,
                oe_vocab_sizes=self.oe_vocab_sizes,
                oe_vocab_offsets=self.oe_vocab_offsets,
                n_grams=self.oe_grams,
                vocab_size=self.ori_vocab_size,
            )
        else:
            assert input_tensor.dim() == 2 # [batch_size, seq_len]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_tensor.size(0) # [batch_size, max_n_gram - 1]
            oe_ngram_ids = n_gram_decode(
                input_ids=input_tensor,
                oe_history_inputs=oe_history_input,
                oe_vocab_sizes=self.oe_vocab_sizes,
                oe_vocab_offsets=self.oe_vocab_offsets,
                n_grams=self.oe_grams,
                vocab_size=self.ori_vocab_size,
            )
        return oe_ngram_ids

# NOTE(liuyuan): loop unroll
class TTXOverEncoding(MojoOverEncoding):
    supported_platforms_list = ["npu", "ilu"]

    def _create_mega_embedding(
        self,
        _mega_embedding_weight: torch.Tensor,
        _mega_embedding_scale: torch.Tensor,
        _mega_embedding_mean: torch.Tensor,
        _mega_embedding_group_size: int,
        _mega_embedding_vocab_start_id: int,
    ) -> torch.nn.Module:
        if (
            _mega_embedding_weight is not None
            and _mega_embedding_scale is not None
            and _mega_embedding_mean is not None
        ):
            oe_mega_embedding = TTXNF4DequantEmbedding(
                _mega_embedding_weight,
                _mega_embedding_scale,
                _mega_embedding_mean,
                group_size=_mega_embedding_group_size,
                vocab_start_id=_mega_embedding_vocab_start_id,
                output_dtype=self.tensor_factory_kwargs.get("dtype", None),
                cpu_only=self.mega_embedding_cpu_only,
            )
        else:
            oe_mega_embedding = torch.nn.Embedding(
                sum(self.oe_vocab_sizes).item(),
                self.oe_embed_dim,
                _weight=_mega_embedding_weight,
                # dtype=self.tensor_factory_kwargs.get("dtype", None),
                device=self.tensor_factory_kwargs.get("device", None),
            )

            if self.mega_embedding_cpu_only:
                assert (
                    _mega_embedding_weight is not None
                    and _mega_embedding_weight.device.type == "cpu"
                )
                # NOTE(liuyuan): Unregister the Parameter [weight] so that it will always stays on cpu until someone move it mannually to the device.
                delattr(oe_mega_embedding, 'weight')
                # WARNING(liuyuan): register_buffer(..persistant=False) DO NOT satisfy our expectation.
                oe_mega_embedding.weight = _mega_embedding_weight
        return oe_mega_embedding

    def forward(
        self, input_tensor: torch.Tensor, oe_history_input: torch.Tensor, input_seq_lens: Optional[torch.Tensor] = None
    ):
        """Calculate the word vectors through over encoding.

        Args:
            input_tensor (torch.Tensor): the input token ids.
            oe_history_input (torch.Tensor): the historic input token ids ([n-gram - 1] at most).
            input_seq_lens (Optional[torch.Tensor], optional): the lengths of each sequences for prefill. Defaults to None.

        Returns:
            torch.Tensor: the word vectors.
        """

        if input_seq_lens is not None:
            assert input_tensor.dim() == 1  # [total_tokens]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_seq_lens.size(0) # [batch_size, max_n_gram - 1]
            oe_result = n_gram_prefill(
                input_ids=input_tensor,
                seq_lens=input_seq_lens,
                oe_history_inputs=oe_history_input,
                oe_vocab_sizes=self.oe_vocab_sizes,
                oe_vocab_offsets=self.oe_vocab_offsets,
                n_grams=self.oe_grams,
                vocab_size=self.ori_vocab_size,
            )

            if self.mega_embedding_cpu_only:
                ori_device = oe_result.device
                oe_result = self.oe_mega_embedding(oe_result.cpu()).to(ori_device)
            else:
                oe_result = self.oe_mega_embedding(oe_result)

        else:
            assert input_tensor.dim() == 2 # [batch_size, seq_len]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_tensor.size(0) # [batch_size, max_n_gram - 1]
            if isinstance(self.oe_mega_embedding, MojoNF4DequantEmbedding):
                oe_result = over_encoding_decode(
                    input_ids=input_tensor,
                    oe_history_inputs=oe_history_input,
                    oe_vocab_sizes=self.oe_vocab_sizes,
                    oe_vocab_offsets=self.oe_vocab_offsets,
                    n_grams=self.oe_grams,
                    LUT_qweight=self.oe_mega_embedding.weight,
                    LUT_scale=self.oe_mega_embedding.scale,
                    LUT_mean=self.oe_mega_embedding.mean,
                    ori_vocab_size=self.ori_vocab_size,
                    mega_vocab_size=self.oe_mega_embedding.weight.size(0),
                    mega_vocab_start_id=self.oe_mega_embedding.vocab_start_id,
                    group_size=self.oe_mega_embedding.group_size,
                    codebook=self.oe_mega_embedding.codebook,
                    output_dtype=self.oe_mega_embedding.output_dtype,
                )
            else:
                oe_result = n_gram_decode(
                    input_ids=input_tensor,
                    oe_history_inputs=oe_history_input,
                    oe_vocab_sizes=self.oe_vocab_sizes,
                    oe_vocab_offsets=self.oe_vocab_offsets,
                    n_grams=self.oe_grams,
                    vocab_size=self.ori_vocab_size,
                )
                if self.mega_embedding_cpu_only:
                    ori_device = oe_result.device
                    oe_result = self.oe_mega_embedding(oe_result.cpu()).to(ori_device)
                else:
                    oe_result = self.oe_mega_embedding(oe_result)

        wte_result = self.ori_embedding(input_tensor)
        # WARNING(liuyuan): concat order is necessary.
        concat_result = torch.cat(
            (
                wte_result,
                oe_result.flatten(-2),
            ),
            dim=-1,
        )

        return self.oe_up_proj(concat_result)

########################################################
# NF4 Dequantization fused Embedding
########################################################
class TTXNF4DequantEmbedding(MojoNF4DequantEmbedding):
    """NF4-quantized embedding backed by the TTX ``embedding_nf4_dequant`` kernel."""

    supported_platforms_list = ["npu", "ilu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = embedding_nf4_dequant(
            input=input,
            LUT_qweight=self.weight,
            LUT_scale=self.scale,
            LUT_mean=self.mean,
            group_size=self.group_size,
            codebook=self.codebook,
            vocab_start_id=self.vocab_start_id,
            vocab_size=self.weight.size(0),
            output_dtype=self.output_dtype,
        )
        return output
