"""UC backend implementation of :class:`MojoOverEncoding`.

OverEncoding (``mojo_opset/core/operators/over_encoding.py:159``) is the
composition of three stages:

1. **n-gram id computation** -- per-token modular arithmetic that mixes
   ``input_ids`` with ``oe_history_input`` and the static ``oe_vocab_*``
   buffers (``n_gram_impl_torch``). Cheap on host, no UB win.
2. **mega-embedding lookup** -- a flat indirect read
   ``mega_weight[oe_ngram_ids.flatten()]`` of shape ``(*input_shape, G,
   oe_embed_dim)``. **This is the hot path that benefits from the UC
   block-GATHER primitive.**
3. **original embedding + concat + ``oe_up_proj``** -- standard torch
   ops; inherited from the parent ``MojoOverEncoding.forward``.

The UC backend therefore only swaps stage (2) for the wheel kernel
``mojo_over_encoding_bf16`` (see ``uc-kernel/kernels/mojo_over_encoding_bf16.py``);
stages (1) and (3) reuse the existing torch reference implementation
inherited from ``MojoOverEncoding``.

Wheel API contract (must stay in lockstep with the kernel file)::

    weight        : bf16, shape (MEGA_VOCAB, OE_EMBED_DIM) == (4096, 192)
    flat_ngram_ids: i32,  shape (NUM_LOOKUPS,)             == (128,)
    out           : bf16, shape (NUM_LOOKUPS, OE_EMBED_DIM) == (128, 192)
    arg_order     : inputs_first -> ``api(weight, flat_ngram_ids, out)``
    ABI           : no trailing INT32 scalars (all dims are fixed; no
                    ``T.dynamic`` declarations).

Fallback contract -- anything that deviates from the fixed-shape
bring-up routes back to ``MojoOverEncoding.forward`` so behaviour stays
correct even when the wheel kernel is absent / the call is off-grid:

* ``oe_mega_embedding`` is not a plain ``torch.nn.Embedding`` (the NF4
  dequant path needs bit-shift unpack inside the kernel and is tracked
  as a separate follow-up; the wrapper hands NF4 configurations off to
  the parent torch reference).
* ``mega_embedding_cpu_only`` is set.
* mega-embedding ``weight.dtype != torch.bfloat16``.
* ``mega_weight.shape != (MEGA_VOCAB, OE_EMBED_DIM)``.
* ``input_tensor.numel() * G != NUM_LOOKUPS`` where ``G = len(self.oe_grams)``.
* weight is not on NPU (meta / CPU runs use parent torch).
* wheel does not export ``mojo_over_encoding_bf16``.
"""

import torch

from mojo_opset.core import MojoOverEncoding
from mojo_opset.core.operators.over_encoding import n_gram_impl_torch

from ._utils import _uc_kernels


# Wheel-side fixed-shape bring-up (must match
# ``uc-kernel/kernels/mojo_over_encoding_bf16.py``).
_FIXED_MEGA_VOCAB = 4096
_FIXED_OE_EMBED_DIM = 192
_FIXED_NUM_LOOKUPS = 128
_FIXED_WEIGHT_DTYPE = torch.bfloat16
_API = "mojo_over_encoding_bf16"


class UCOverEncoding(MojoOverEncoding):
    supported_platforms_list = ["npu"]

    def forward(self, input_tensor, oe_history_input, q_lens=None):
        mega_emb = self.oe_mega_embedding

        # The NF4 path (``MojoNF4DequantEmbedding``) needs fused bit-shift
        # unpack + codebook lookup + scale/mean inside the kernel; tracked
        # as a B2-05 NF4 follow-up. Until then hand off to the parent
        # torch reference.
        if not isinstance(mega_emb, torch.nn.Embedding):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        if self.mega_embedding_cpu_only:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        mega_weight = mega_emb.weight
        num_grams = len(self.oe_grams)
        total_lookups = input_tensor.numel() * num_grams

        # Off-grid / dtype-incompatible / non-NPU -> parent torch path.
        if (
            mega_weight.dtype != _FIXED_WEIGHT_DTYPE
            or tuple(mega_weight.shape) != (_FIXED_MEGA_VOCAB, _FIXED_OE_EMBED_DIM)
            or total_lookups != _FIXED_NUM_LOOKUPS
            or mega_weight.device.type != "npu"
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        try:
            kernels = _uc_kernels()
        except Exception:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if _API not in kernels:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ------------------------------------------------------------------
        # Stage 1: host-side n-gram id computation (reuses the parent
        # ``n_gram_impl_torch`` reference; output dtype is int64 / long).
        # ------------------------------------------------------------------
        if q_lens is not None:
            assert input_tensor.dim() == 1  # [total_tokens]
            assert (
                oe_history_input.dim() == 2
                and oe_history_input.size(0) == q_lens.size(0)
            )
            seq_offset = 0
            oe_ngram_ids_list = []
            for seq_idx, seq_len in map(
                lambda x: (x[0], x[1].item()), enumerate(q_lens)
            ):
                input_ids_i = input_tensor[seq_offset : seq_offset + seq_len]
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
            oe_ngram_ids = torch.cat(oe_ngram_ids_list, dim=0)  # (T, G)
            oe_result_shape = (input_tensor.size(0), num_grams, _FIXED_OE_EMBED_DIM)
        else:
            assert input_tensor.dim() == 2  # [batch_size, seq_len]
            assert (
                oe_history_input.dim() == 2
                and oe_history_input.size(0) == input_tensor.size(0)
            )
            oe_ngram_ids = n_gram_impl_torch(
                input_tensor,
                oe_history_input,
                self.oe_vocab_sizes,
                self.oe_vocab_offsets,
                self.oe_grams,
                self.ori_vocab_size,
            )  # (B, S, G)
            oe_result_shape = (
                input_tensor.size(0),
                input_tensor.size(1),
                num_grams,
                _FIXED_OE_EMBED_DIM,
            )

        # ------------------------------------------------------------------
        # Stage 2: flatten ngram ids and dispatch the block GATHER kernel.
        # The lifter requires the indirect index buffer to be ``int32``
        # (kernel.py:1429-1438 single-runtime-index hard rule), so cast
        # explicitly before the call.
        # ------------------------------------------------------------------
        flat_ids = oe_ngram_ids.reshape(-1).to(torch.int32).contiguous()
        mega_weight_c = mega_weight.contiguous()
        out_flat = torch.empty(
            (_FIXED_NUM_LOOKUPS, _FIXED_OE_EMBED_DIM),
            dtype=_FIXED_WEIGHT_DTYPE,
            device=mega_weight.device,
        )
        # arg_order = "inputs_first" -> (weight, flat_ngram_ids, out).
        kernels[_API](mega_weight_c, flat_ids, out_flat)
        oe_result = out_flat.reshape(*oe_result_shape)

        # ------------------------------------------------------------------
        # Stage 3: original embedding + concat + ``oe_up_proj`` (identical
        # to the torch reference -- no UC fast path needed here).
        # ------------------------------------------------------------------
        wte_result = self.ori_embedding(input_tensor)
        # WARNING(liuyuan): concat order is necessary (matches core).
        concat_result = torch.cat(
            (
                wte_result,
                oe_result.flatten(-2),
            ),
            dim=-1,
        )
        return self.oe_up_proj(concat_result)
