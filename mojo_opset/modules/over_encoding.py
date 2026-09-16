"""Over-encoding state and projections; kernels are selected by the functions layer."""

import torch

from mojo_opset import functions


def _allow_missing_config(module, incompatible_keys):
    config_names = {"oe_vocab_sizes", "oe_grams", "oe_vocab_offsets"}
    incompatible_keys.missing_keys[:] = [
        key for key in incompatible_keys.missing_keys if key.rsplit(".", 1)[-1] not in config_names
    ]


class OverEncodingNGram(torch.nn.Module):
    def __init__(self, ori_vocab_size, oe_vocab_sizes, oe_grams, *, implementation=None, device=None):
        super().__init__()
        self.ori_vocab_size, self.implementation = ori_vocab_size, implementation
        self.register_buffer(
            "oe_vocab_sizes", torch.as_tensor(oe_vocab_sizes, device=device, dtype=torch.int64), persistent=False
        )
        self.register_buffer("oe_grams", torch.as_tensor(oe_grams, device=device, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "oe_vocab_offsets",
            torch.cat((self.oe_vocab_sizes.new_zeros(1), self.oe_vocab_sizes[:-1].cumsum(0))),
            persistent=False,
        )

    def forward(self, input_ids, oe_history_input, q_lens=None):
        if q_lens is not None:
            if input_ids.ndim != 1 or oe_history_input.shape[0] != q_lens.shape[0]:
                raise ValueError("prefill expects packed input_ids and one history per sequence")
            return functions.n_gram_prefill(
                input_ids,
                q_lens,
                oe_history_input,
                self.oe_vocab_sizes,
                self.oe_vocab_offsets,
                self.oe_grams,
                self.ori_vocab_size,
                implementation=self.implementation,
            )
        if input_ids.ndim != 2 or oe_history_input.shape[0] != input_ids.shape[0]:
            raise ValueError("decode expects batched input_ids and one history per sequence")
        return functions.n_gram_decode(
            input_ids,
            oe_history_input,
            self.oe_vocab_sizes,
            self.oe_vocab_offsets,
            self.oe_grams,
            self.ori_vocab_size,
            implementation=self.implementation,
        )


class NF4DequantEmbedding(torch.nn.Module):
    def __init__(
        self,
        qweight,
        scale,
        mean,
        *,
        group_size,
        vocab_start_id=0,
        cpu_only=False,
        output_dtype=torch.bfloat16,
        implementation=None,
    ):
        super().__init__()
        from mojo_opset.kernels._nf4 import get_nf4_codebook

        if qweight.ndim != 2 or scale.ndim != 2 or scale.shape != mean.shape:
            raise ValueError("NF4 qweight, scale, mean must be 2D with matching scale/mean shapes")
        if group_size <= 0 or qweight.shape[1] * 2 != scale.shape[1] * group_size:
            raise ValueError("NF4 packed columns must match scale groups times group_size")
        self.group_size, self.vocab_start_id = group_size, vocab_start_id
        self.output_dtype, self.implementation, self.cpu_only = output_dtype, implementation, cpu_only
        self.embedding_dim = scale.shape[1] * group_size
        for name, value in (
            ("weight", qweight),
            ("scale", scale),
            ("mean", mean),
            ("codebook", get_nf4_codebook(qweight.device)),
        ):
            if cpu_only:
                setattr(self, name, value)
            else:
                self.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))

    def forward(self, input_ids):
        return functions.embedding_nf4_dequant(
            input_ids,
            self.weight,
            self.scale,
            self.mean,
            group_size=self.group_size,
            codebook=self.codebook,
            vocab_start_id=self.vocab_start_id,
            output_dtype=self.output_dtype,
            implementation="torch_reference" if self.cpu_only else self.implementation,
        )


class OverEncoding(OverEncodingNGram):
    def __init__(
        self,
        ori_vocab_size,
        ori_embed_dim,
        oe_embed_dim,
        oe_vocab_sizes,
        oe_grams,
        _ori_embedding_weight=None,
        _mega_embedding_weight=None,
        _mega_embedding_scale=None,
        _mega_embedding_mean=None,
        _mega_embedding_group_size=1,
        _mega_embedding_vocab_start_id=0,
        mega_embedding_cpu_only=False,
        *,
        implementation=None,
        device=None,
        dtype=None,
    ):
        super().__init__(ori_vocab_size, oe_vocab_sizes, oe_grams, implementation=implementation, device=device)
        self._non_persistent_buffers_set.difference_update({"oe_vocab_sizes", "oe_grams", "oe_vocab_offsets"})
        self.register_load_state_dict_post_hook(_allow_missing_config)
        self.ori_embed_dim, self.oe_embed_dim = ori_embed_dim, oe_embed_dim
        self.mega_embedding_cpu_only = mega_embedding_cpu_only
        self.ori_embedding = torch.nn.Embedding(
            ori_vocab_size, ori_embed_dim, _weight=_ori_embedding_weight, device=device
        )
        if all(x is not None for x in (_mega_embedding_weight, _mega_embedding_scale, _mega_embedding_mean)):
            self.oe_mega_embedding = NF4DequantEmbedding(
                _mega_embedding_weight,
                _mega_embedding_scale,
                _mega_embedding_mean,
                group_size=_mega_embedding_group_size,
                vocab_start_id=_mega_embedding_vocab_start_id,
                output_dtype=dtype or torch.get_default_dtype(),
                cpu_only=mega_embedding_cpu_only,
                implementation=implementation,
            )
        else:
            self.oe_mega_embedding = torch.nn.Embedding(
                int(self.oe_vocab_sizes.sum()), oe_embed_dim, _weight=_mega_embedding_weight, device=device
            )
            if mega_embedding_cpu_only:
                if _mega_embedding_weight is None or _mega_embedding_weight.device.type != "cpu":
                    raise ValueError("CPU-only mega embedding needs an explicit CPU weight")
                del self.oe_mega_embedding.weight
                self.oe_mega_embedding.weight = _mega_embedding_weight
        self.oe_up_proj = torch.nn.Linear(
            len(oe_vocab_sizes) * oe_embed_dim + ori_embed_dim, ori_embed_dim, bias=False, device=device
        )

    def forward(self, input_ids, oe_history_input, q_lens=None):
        embedding = self.oe_mega_embedding
        if q_lens is None and isinstance(embedding, NF4DequantEmbedding) and not self.mega_embedding_cpu_only:
            result = functions.over_encoding_decode(
                input_ids,
                oe_history_input,
                self.oe_vocab_sizes,
                self.oe_vocab_offsets,
                self.oe_grams,
                embedding.weight,
                embedding.scale,
                embedding.mean,
                ori_vocab_size=self.ori_vocab_size,
                group_size=embedding.group_size,
                codebook=embedding.codebook,
                vocab_start_id=embedding.vocab_start_id,
                output_dtype=embedding.output_dtype,
                implementation=self.implementation,
            )
        else:
            ids = super().forward(input_ids, oe_history_input, q_lens)
            result = embedding(ids.cpu()).to(ids.device) if self.mega_embedding_cpu_only else embedding(ids)
        return self.oe_up_proj(torch.cat((self.ori_embedding(input_ids), result.flatten(-2)), dim=-1))
