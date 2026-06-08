"""UC backend implementation of :class:`MojoRelativeEmbedding`.

Wraps the wheel kernel API ``mojo_relative_embedding_bf16`` (block GATHER,
bf16 embedding table + i32 bucket indices) — see
``uc-kernel/kernels/mojo_relative_embedding_bf16.py``.

T5 relative position bias breaks down cleanly into two stages:

1. ``_relative_position_bucket`` — host-side integer arithmetic
   (``arange``/``abs``/``log``). The maths is data-independent of the
   embedding table, has no UB-friendly form, and is inherited from
   ``MojoRelativeEmbedding`` unchanged.
2. ``embedding(bucket_flat, weight)`` — the only piece that benefits from
   the UC GATHER primitive. This is what the wheel kernel implements:
   ``out_flat[k, :] = weight[bucket_flat[k], :]``, with a fixed contract::

       weight       : bf16, shape (NUM_BUCKETS, H) == (32, 16)
       bucket_flat  : i32,  shape (NUM_TOKENS,)    == (256,)
       out          : bf16, shape (NUM_TOKENS, H)  == (256, 16)
       arg_order    : inputs_first -> api(weight, bucket_flat, out)

The wrapper then reshapes ``(NUM_TOKENS, H)`` -> ``(Lq, Lk, H)``, permutes
to ``(H, Lq, Lk)`` and unsqueezes the batch dim, matching the parent
``forward`` return contract ``[1, num_heads, Lq, Lk]``.

Fallback contract (mirrors UCEmbedding / UCSdpa) — anything that violates
the fixed-shape bring-up contract routes back to
``MojoRelativeEmbedding.forward`` so behaviour stays correct even when the
wheel kernel is absent or the call shape is off-grid:

* ``self.embedding.weight.dtype != torch.bfloat16``
* ``self.num_buckets != _FIXED_NUM_BUCKETS``
* ``self.num_heads   != _FIXED_H``
* ``lq * lk          != _FIXED_NUM_TOKENS``
* wheel does not export ``mojo_relative_embedding_bf16``
* device is not NPU (meta / CPU)
"""

import torch

from mojo_opset.experimental import MojoRelativeEmbedding

from ._utils import _uc_kernels


# Wheel-side fixed-shape bring-up (must stay in lockstep with
# ``uc-kernel/kernels/mojo_relative_embedding_bf16.py``).
_FIXED_NUM_BUCKETS = 32
_FIXED_H = 16
_FIXED_NUM_TOKENS = 256
_FIXED_WEIGHT_DTYPE = torch.bfloat16
_API = "mojo_relative_embedding_bf16"


class UCRelativeEmbedding(MojoRelativeEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        # Parent-class input contract first — keep the error semantics.
        if not isinstance(lq, int) or not isinstance(lk, int) or lq <= 0 or lk <= 0:
            raise ValueError("lq and lk must be positive integers")

        weight = self.embedding.weight

        # Off-grid / dtype-incompatible / non-NPU -> parent torch path.
        if (
            weight.dtype != _FIXED_WEIGHT_DTYPE
            or self.num_buckets != _FIXED_NUM_BUCKETS
            or self.num_heads != _FIXED_H
            or (lq * lk) != _FIXED_NUM_TOKENS
            or weight.device.type != "npu"
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

        device = weight.device

        # Stage 1: host-side bucket grid (identical to parent class).
        rel_pos = (
            torch.arange(lk, device=device).unsqueeze(0)
            - torch.arange(lq, device=device).unsqueeze(1)
        )
        bucket = self._relative_position_bucket(rel_pos)  # (Lq, Lk), int64

        # Stage 2: flatten + cast to i32 (wheel ABI: indirect index buffer
        # must be ``int32``, lifter hard-rule at kernel.py:1429-1438).
        bucket_flat = bucket.reshape(-1).to(torch.int32).contiguous()

        weight_c = weight.contiguous()
        out_flat = torch.empty(
            (_FIXED_NUM_TOKENS, _FIXED_H),
            dtype=_FIXED_WEIGHT_DTYPE,
            device=device,
        )

        # Wheel ABI: fixed-shape (no T.dynamic) -> no trailing scalars.
        # arg_order = "inputs_first" -> (weight, bucket_flat, out).
        kernels[_API](weight_c, bucket_flat, out_flat)

        # Reshape gathered table back to the parent return layout
        # ``[1, num_heads, Lq, Lk]``: (Lq*Lk, H) -> (Lq, Lk, H) -> (H, Lq, Lk)
        # -> (1, H, Lq, Lk).
        out = out_flat.reshape(lq, lk, _FIXED_H).permute(2, 0, 1).unsqueeze(0)
        return out.contiguous()
