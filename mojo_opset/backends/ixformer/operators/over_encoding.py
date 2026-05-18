from typing import Optional

import torch

from ixformer.inference.functions import over_encoding_ngram
from mojo_opset.core import MojoOverEncoding
from mojo_opset.core import MojoOverEncodingNGram


class IxformerOverEncodingNGram(MojoOverEncodingNGram):
    supported_platforms_list = ["ilu"]

    def forward(
        self, input_ids: torch.Tensor, oe_history_input: torch.Tensor, q_lens: Optional[torch.Tensor] = None
    ):
        return over_encoding_ngram(
            input_ids=input_ids,
            oe_history_input=oe_history_input,
            oe_vocab_sizes=self.oe_vocab_sizes,
            oe_grams=self.oe_grams,
            ori_vocab_size=self.ori_vocab_size,
            input_seq_lens=q_lens,
        )


class IxformerOverEncoding(MojoOverEncoding):
    supported_platforms_list = ["ilu"]

    def forward(
        self, input_tensor: torch.Tensor, oe_history_input: torch.Tensor, q_lens: Optional[torch.Tensor] = None
    ):
        oe_ngram_ids = over_encoding_ngram(
            input_ids=input_tensor,
            oe_history_input=oe_history_input,
            oe_vocab_sizes=self.oe_vocab_sizes,
            oe_grams=self.oe_grams,
            ori_vocab_size=self.ori_vocab_size,
            input_seq_lens=q_lens,
        )

        if self.mega_embedding_cpu_only:
            ori_device = oe_ngram_ids.device
            oe_result = self.oe_mega_embedding(oe_ngram_ids.cpu()).to(ori_device)
        else:
            oe_result = self.oe_mega_embedding(oe_ngram_ids)

        wte_result = self.ori_embedding(input_tensor)
        concat_result = torch.cat(
            (
                wte_result,
                oe_result.flatten(-2),
            ),
            dim=-1,
        )

        return self.oe_up_proj(concat_result)
