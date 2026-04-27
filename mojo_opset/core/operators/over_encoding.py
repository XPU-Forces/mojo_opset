from typing import List, Optional

import torch
from torch.nn import Parameter

from mojo_opset.core.operator import MojoOperator

def __make_hook_to_ignore_workspace_buffer__(ignore_keys=()):
    def load_state_dict_post_hook(module, incompatible_keys) -> None:
        key2ignroe = []
        for miss in incompatible_keys.missing_keys:
            if miss.split('.')[-1] in ignore_keys:
                key2ignroe.append(miss)
        # NOTE(liuyuan): incompatible_keys DOES NOT allow to set attribute.
        for key in key2ignroe:
            incompatible_keys.missing_keys.remove(key)

    return load_state_dict_post_hook

def n_gram_impl_torch(
    input_ids: torch.Tensor,
    oe_history_inputs: torch.Tensor,
    oe_vocab_sizes: torch.Tensor,
    oe_vocab_offset: torch.Tensor,
    n_grams: torch.Tensor,
    ori_vocab_size: int,
):
    """_summary_

    Args:
        input_ids (torch.Tensor): _description_
        oe_history_inputs (torch.Tensor): _description_
        oe_vocab_sizes (torch.Tensor): _description_
        oe_vocab_offset (torch.Tensor): _description_
        n_grams (torch.Tensor): _description_
        ori_vocab_size (int): _description_

    Returns:
        _type_: _description_
    """
    n_gram_ids = []
    for gram_idx, gram in map(lambda val: (val[0], val[1].item()), enumerate(n_grams)):
        oe_carry = ori_vocab_size
        n_gram_id = input_ids

        # TODO(liuyuan): make it recomputation free.
        for i in range(1, gram):

            complete_input_ids = torch.cat(
                (oe_history_inputs[..., -i:], input_ids[..., :-i]),
                dim=-1,
            )

            if i > n_gram_id.size(-1):
                complete_input_ids = complete_input_ids[..., : n_gram_id.size(-1)]

            # NOTE(liuyuan): the fowllowing modulo operators are designed for big decimals.
            n_gram_id = (n_gram_id + complete_input_ids * oe_carry) % oe_vocab_sizes[gram_idx]
            oe_carry = oe_carry * ori_vocab_size % oe_vocab_sizes[gram_idx]

        n_gram_id = n_gram_id + oe_vocab_offset[gram_idx]
        n_gram_ids.append(n_gram_id)
    return torch.stack(n_gram_ids, dim=-1)

class MojoOverEncodingNGram(MojoOperator):
    def __init__(
        self,
        ori_vocab_size: int,
        oe_vocab_sizes: List[int] | torch.Tensor,
        oe_grams: List[int] | torch.Tensor,
        **kwargs,
    ):
        """Calculate the ngram ids for over_encoding.
        See  https://bytedance.larkoffice.com/wiki/LLWjwsxlWi90a6kuaUXcFqDPngc?from=space_home_recent&pre_pathname=%2Fdrive%2Fhome%2F&previous_navigation_time=1767172595981

        Args:
            oe_vocab_sizes (List[int] or torch.Tensor): sizes of vocabularies for each gram.
            oe_grams (List[int] or torch.Tensor): grams for Over Encoding.
        """
        super().__init__(**kwargs)
        self.ori_vocab_size = ori_vocab_size
        self.register_buffer(
            "oe_vocab_sizes",
            (
                torch.tensor(oe_vocab_sizes, **self.tensor_factory_kwargs)
                if not isinstance(oe_vocab_sizes, torch.Tensor)
                else oe_vocab_sizes
            ),
            persistent=False
        )
        self.register_buffer(
            "oe_grams",
            (
                torch.tensor(oe_grams, **self.tensor_factory_kwargs)
                if not isinstance(oe_grams, torch.Tensor)
                else oe_grams
            ),
            persistent=False
        )
        self.register_buffer(
            "oe_vocab_offsets",
            torch.cumsum(
                torch.cat(
                    [
                        torch.tensor(
                            [0],
                            device=self.oe_vocab_sizes.device,
                            dtype=self.oe_vocab_sizes.dtype,
                        ),
                        self.oe_vocab_sizes[:-1],
                    ],
                    dim=0,
                ).to(torch.long),
                dim=0,
            ),
            persistent=False
        )

    def forward(
        self, input_ids: torch.Tensor, oe_history_input: torch.Tensor, input_seq_lens: Optional[torch.Tensor] = None
    ):
        
        if input_seq_lens is not None:
            assert input_ids.dim() == 1  # [total_tokens]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_seq_lens.size(0) # [batch_size, max_n_gram - 1]
            seq_offset = 0
            oe_ngram_ids_list = []
            for seq_idx, seq_len in map(lambda x: (x[0], x[1].item()), enumerate(input_seq_lens)):
                input_ids_i = input_ids[seq_offset : seq_offset + seq_len]
                oe_ngram_ids_list.append(
                    n_gram_impl_torch(
                        input_ids_i,
                        oe_history_input[seq_idx],
                        self.oe_vocab_sizes,
                        self.oe_vocab_offsets,
                        self.oe_grams,
                        self.ori_vocab_size,
                    )
                )
                seq_offset += seq_len

            oe_ngram_ids = torch.cat(oe_ngram_ids_list, dim=0)

        else:
            assert input_ids.dim() == 2  # [batch_size, seq_len]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_ids.size(0) # [batch_size, max_n_gram - 1]
            oe_ngram_ids = n_gram_impl_torch(
                input_ids,
                oe_history_input,
                self.oe_vocab_sizes,
                self.oe_vocab_offsets,
                self.oe_grams,
                self.ori_vocab_size,
            )

        return oe_ngram_ids
    
    def extra_repr(self) -> str:
        return f"{self.ori_vocab_size=}, {self.oe_vocab_sizes=}, {self.oe_grams=}, {self.oe_vocab_offsets=}".replace(
            "self.", ""
        )


class MojoOverEncoding(MojoOperator):

    def __init__(
        self,
        ori_vocab_size: int,
        ori_embed_dim: int,
        oe_embed_dim: int,
        oe_vocab_sizes: List[int] | torch.Tensor,
        oe_grams: List[int] | torch.Tensor,
        _ori_embedding_weight: torch.Tensor = None,
        _mega_embedding_weight: torch.Tensor = None,
        _mega_embedding_scale: torch.Tensor = None,
        _mega_embedding_mean: torch.Tensor = None,
        _mega_embedding_group_size: int = 1,
        _mega_embedding_vocab_start_id: int = 0,
        mega_embedding_cpu_only=False,
        **kwargs,
    ):
        """Construct the Over Encoding Layer.
        See  https://bytedance.larkoffice.com/wiki/LLWjwsxlWi90a6kuaUXcFqDPngc?from=space_home_recent&pre_pathname=%2Fdrive%2Fhome%2F&previous_navigation_time=1767172595981

        Args:
            ori_vocab_size (int): the original vocabulary size.(The original embedding.size(0))
            ori_embed_dim (int): the original embedding/hidden dim.(The original embedding.size(1))
            oe_embed_dim (int): the embedding dim used in Over Encoding.
            oe_vocab_sizes (List[int] or torch.Tensor): sizes of vocabularies for each gram.
            oe_grams (List[int] or torch.Tensor): grams for Over Encoding.
            _ori_embedding_weight(torch.Tensor):  the custom tensor for the original embedding. Default to None.
            _mega_embedding_weight(torch.Tensor): the custom tensor for the mega embedding. Default to None.
            _mega_embedding_scale(torch.Tensor): the custom tensor for the mega embedding scale for NF4 quantization. Default to None.
            _mega_embedding_mean(torch.Tensor): the custom tensor for the mega embedding mean for NF4 quantization. Default to None.
            _mega_embedding_group_size(int): the group size used for the mega embedding. Default to 1.
            _mega_embedding_vocab_start_id(int): the vocabulary start id used for the mega embedding. Default to 0.
            mega_embedding_cpu_only(bool): whether to use the cpu only for the mega embedding. Default to False.
        """
        super().__init__(**kwargs)

        self.ori_vocab_size = ori_vocab_size
        self.ori_embed_dim = ori_embed_dim
        self.oe_embed_dim = oe_embed_dim
        self.mega_embedding_cpu_only = mega_embedding_cpu_only

        self.register_buffer(
            "oe_vocab_sizes",
            (
                torch.tensor(oe_vocab_sizes, **self.tensor_factory_kwargs)
                if not isinstance(oe_vocab_sizes, torch.Tensor)
                else oe_vocab_sizes
            ),
        )
        self.register_buffer(
            "oe_grams",
            (
                torch.tensor(oe_grams, **self.tensor_factory_kwargs)
                if not isinstance(oe_grams, torch.Tensor)
                else oe_grams
            ),
        )
        self.register_buffer(
            "oe_vocab_offsets",
            torch.cumsum(
                torch.cat(
                    [
                        torch.tensor(
                            [0],
                            device=self.oe_vocab_sizes.device,
                            dtype=self.oe_vocab_sizes.dtype,
                        ),
                        self.oe_vocab_sizes[:-1],
                    ],
                    dim=0,
                ).to(torch.long),
                dim=0,
            ),
        )

        self.register_load_state_dict_post_hook(
            __make_hook_to_ignore_workspace_buffer__(
                ("oe_vocab_sizes", "oe_grams", "oe_vocab_offsets")
            ),
        )

        self.ori_embedding = torch.nn.Embedding(
            self.ori_vocab_size,
            self.ori_embed_dim,
            _weight=_ori_embedding_weight,
            # dtype=self.tensor_factory_kwargs.get("dtype", None),
            device=self.tensor_factory_kwargs.get("device", None),
        )

        self.oe_mega_embedding = self._create_mega_embedding(
            _mega_embedding_weight,
            _mega_embedding_scale,
            _mega_embedding_mean,
            _mega_embedding_group_size,
            _mega_embedding_vocab_start_id,
        )

        self.oe_up_proj = torch.nn.Linear(
            len(self.oe_vocab_sizes) * self.oe_embed_dim + self.ori_embed_dim,
            self.ori_embed_dim,
            bias=False,
            # dtype=self.tensor_factory_kwargs.get("dtype", None),
            device=self.tensor_factory_kwargs.get("device", None),
        )

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
            oe_mega_embedding = NF4DequantEmbedding(
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
            seq_offset = 0
            oe_ngram_ids_list = []
            for seq_idx, seq_len in map(lambda x: (x[0], x[1].item()), enumerate(input_seq_lens)):
                input_ids = input_tensor[seq_offset : seq_offset + seq_len]
                oe_ngram_ids_list.append(
                    n_gram_impl_torch(
                        input_ids,
                        oe_history_input[seq_idx],
                        self.oe_vocab_sizes,
                        self.oe_vocab_offsets,
                        self.oe_grams,
                        self.ori_vocab_size,
                    )
                )
                seq_offset += seq_len

            oe_ngram_ids = torch.cat(oe_ngram_ids_list, dim=0)

        else:
            assert input_tensor.dim() == 2 # [batch_size, seq_len]
            assert oe_history_input.dim() == 2 and oe_history_input.size(0) == input_tensor.size(0) # [batch_size, max_n_gram - 1]
            oe_ngram_ids = n_gram_impl_torch(
                input_tensor,
                oe_history_input,
                self.oe_vocab_sizes,
                self.oe_vocab_offsets,
                self.oe_grams,
                self.ori_vocab_size,
            )

        if self.mega_embedding_cpu_only:
            ori_device = oe_ngram_ids.device
            oe_result = self.oe_mega_embedding(oe_ngram_ids.cpu()).to(ori_device)
        else:
            oe_result = self.oe_mega_embedding(oe_ngram_ids)

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

    def extra_repr(self) -> str:
        return f"{self.ori_vocab_size=}, {self.ori_embed_dim=}, {self.oe_embed_dim=}, {self.oe_vocab_sizes=}, {self.oe_grams=}, {self.mega_embedding_cpu_only=}".replace(
            "self.", ""
        )

########################################################
# NF4 Dequantization fused Embedding
########################################################
_NF4_CODEBOOK = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

def get_nf4_codebook(
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return torch.tensor(_NF4_CODEBOOK, device=device, dtype=dtype)

def unpack_nf4_int8_to_uint4(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError(f"`packed` must be 2D, got shape={tuple(packed.shape)}")
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError(
            f"`packed` must have dtype torch.int8 or torch.uint8, got {packed.dtype}."
        )

    q_u8 = packed.to(torch.uint8)
    low = q_u8 & 0x0F
    high = (q_u8 >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)

def dequantize_nf4_rows(
    nf4_qweight: torch.Tensor,
    nf4_scale: torch.Tensor,
    nf4_mean: torch.Tensor,
    *,
    group_size: int,
    codebook: torch.Tensor = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if nf4_qweight.ndim != 2:
        raise ValueError(
            f"`nf4_qweight` must be 2D, got shape={tuple(nf4_qweight.shape)}."
        )
    if nf4_scale.ndim != 2 or nf4_mean.ndim != 2:
        raise ValueError(
            "`nf4_scale` and `nf4_mean` must both be 2D, "
            f"got scale={tuple(nf4_scale.shape)}, mean={tuple(nf4_mean.shape)}."
        )
    if nf4_scale.shape != nf4_mean.shape:
        raise ValueError(
            "`nf4_scale` and `nf4_mean` must have the same shape, "
            f"got scale={tuple(nf4_scale.shape)}, mean={tuple(nf4_mean.shape)}."
        )
    if group_size <= 0:
        raise ValueError(f"`group_size` must be > 0, got {group_size}.")

    num_rows = nf4_scale.shape[0]
    num_groups = nf4_scale.shape[1]
    embedding_dim = num_groups * group_size

    if nf4_qweight.shape[0] != num_rows:
        raise ValueError(
            "`nf4_qweight` row count must match scale/mean, "
            f"got qweight={tuple(nf4_qweight.shape)}, scale={tuple(nf4_scale.shape)}."
        )
    if nf4_qweight.shape[1] * 2 != embedding_dim:
        raise ValueError(
            "`nf4_qweight` column count must be embedding_dim / 2, "
            f"got qweight={tuple(nf4_qweight.shape)}, embedding_dim={embedding_dim}."
        )

    if codebook is None:
        codebook = get_nf4_codebook(device=nf4_qweight.device, dtype=torch.float16)
    else:
        codebook = codebook.to(device=nf4_qweight.device, dtype=torch.float16)

    unpack_input = nf4_qweight.cpu() if nf4_qweight.device.type == "npu" else nf4_qweight
    q_idx = unpack_nf4_int8_to_uint4(unpack_input).to(device=nf4_qweight.device).reshape(
        num_rows,
        num_groups,
        group_size,
    ).to(torch.long)
    values = codebook.index_select(0, q_idx.reshape(-1)).reshape(
        num_rows,
        num_groups,
        group_size,
    ).to(torch.float32)

    scale = nf4_scale.to(torch.float32).reshape(num_rows, num_groups, 1)
    mean = nf4_mean.to(torch.float32).reshape(num_rows, num_groups, 1)
    return (values * scale + mean).reshape(num_rows, embedding_dim).to(output_dtype)

class NF4DequantEmbedding(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        mean: torch.Tensor,
        *,
        group_size: int,
        vocab_start_id: int = 0,
        cpu_only: bool = False,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if qweight.ndim != 2 or scale.ndim != 2 or mean.ndim != 2:
            raise ValueError(
                "NF4 embedding tensors must all be 2D, "
                f"got qweight={tuple(qweight.shape)}, scale={tuple(scale.shape)}, mean={tuple(mean.shape)}."
            )

        if scale.shape != mean.shape:
            raise ValueError(
                f"`scale` and `mean` must have the same shape, got {scale.shape} and {mean.shape}."
            )

        if group_size <= 0:
            raise ValueError(f"`group_size` must be > 0, got {group_size}.")

        if qweight.ndim != 2 or scale.ndim != 2 or mean.ndim != 2:
            raise ValueError(
                "NF4 embedding tensors must all be 2D, "
                f"got qweight={tuple(qweight.shape)}, "
                f"scale={tuple(scale.shape)}, mean={tuple(mean.shape)}."
            )

        self.embedding_dim = scale.size(1) * group_size

        if qweight.size(1) * 2 != self.embedding_dim:
            raise ValueError(
                f"`weight` shape {tuple(qweight.shape)} is incompatible with "
                f"`scale` shape {tuple(scale.shape)} and group_size={group_size}."
            )

        self.group_size = group_size
        self.output_dtype = output_dtype
        self.vocab_start_id = vocab_start_id

        if not cpu_only:
            self.register_parameter("weight", Parameter(qweight, requires_grad=False))
            self.register_parameter("scale", Parameter(scale, requires_grad=False))
            self.register_parameter("mean", Parameter(mean, requires_grad=False))
            self.register_parameter(
                "codebook",
                Parameter(
                    get_nf4_codebook(device=qweight.device, dtype=torch.float16),
                    requires_grad=False,
                ),
            )
        else:
            self.weight = qweight
            self.scale = scale
            self.mean = mean
            self.codebook = get_nf4_codebook(device=qweight.device, dtype=torch.float16)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        input_flat = input.contiguous().view(-1)

        if input_flat.numel() == 0:
            return torch.empty(
                (*input.shape, self.embedding_dim),
                dtype=self.output_dtype,
                device=input.device,
            )

        output = torch.zeros(
            (input_flat.numel(), self.embedding_dim),
            dtype=self.output_dtype,
            device=input.device,
        )

        local_ids = input_flat.to(torch.long) - self.vocab_start_id
        valid_mask = (local_ids >= 0) & (local_ids < self.weight.size(0))
        if valid_mask.any():
            valid_ids = local_ids[valid_mask]
            output[valid_mask] = dequantize_nf4_rows(
                self.weight.index_select(0, valid_ids),
                self.scale.index_select(0, valid_ids),
                self.mean.index_select(0, valid_ids),
                group_size=self.group_size,
                codebook=self.codebook,
                output_dtype=self.output_dtype,
            )
        return output.view(*input.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.weight.size(0)}, "
            f"embedding_dim={self.scale.size(1) * self.group_size}, "
            f"group_size={self.group_size}, "
            f"output_dtype={self.output_dtype}"
        )
