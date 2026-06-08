"""UC backend for ``MojoMoEDispatch``.

Maps the high-level MoE token re-routing op
(``mojo_opset/core/operators/moe.py:195``) onto the wheel-side block
GATHER kernel ``mojo_moe_dispatch_bf16``.  Only the per-task row
gather runs on the wheel; the bookkeeping outputs
(``tokens_per_expert``, ``sorted_gates``, ``token_indices``) are host
arithmetic since they are pure index gathers / bincount that already
fit in the torch fast path.

Mapping
-------

Given ``hidden_states (BS, H)``, ``top_k_gates (BS, top_k)`` and
``top_k_indices (BS, top_k)``::

* ``flat_top_k_indices = top_k_indices.flatten()                   # (R,)``
* ``expert_sort_indices = argsort(flat_top_k_indices, stable=True) # (R,)``
* ``batch_token_indices[i] = i // top_k``
* ``token_indices[r] = batch_token_indices[expert_sort_indices[r]] # (R,) int32``

so that ``sorted_hidden_states[r, :] = hidden_states[token_indices[r], :]``
which is the canonical block GATHER pattern the wheel kernel exposes
(see ``uc-kernel/kernels/mojo_moe_dispatch_bf16.py``).

P2-29 perf rewrite
------------------

The previous wrapper materialised ``expanded_src =
hidden_states.repeat_interleave(top_k, dim=0).contiguous()``
(4 MB DRAM read+write at R=2048, H=1024) and built a ``routing_flat``
inverse permutation so the kernel could do a block SCATTER.  Both are
unnecessary: ``token_indices`` is already the gather index needed and
the kernel can read ``hidden_states`` directly.  This drops the
host-side replication (~4 MB DRAM round-trip) so the wrapper +
kernel total stays at the theoretical lower bound of one read of
``hidden_states`` + one write of ``sorted_hidden_states``.

Wheel kernel ABI (fixed-shape, no ``T.dynamic``)::

    mojo_moe_dispatch_bf16(hidden_states, token_indices, out)
        hidden_states : bf16 (BS, H)
        token_indices : int32 (R,)
        out           : bf16 (R, H)

with the fixed contract ``BS = 256``, ``TOP_K = 8``, ``R = 2048``,
``H = 1024``.  Anything outside this contract (different ``BS / TOP_K
/ H``, non-bf16 hidden, etc.) raises ``NotImplementedError`` so the
framework can dispatch a different backend (typically the parent
``MojoMoEDispatch.forward`` which is already torch-native on NPU).
"""

from typing import Tuple

import torch

from mojo_opset.core import MojoMoEDispatch

from ._utils import _uc_kernels


# Must match the wheel kernel constants in
# ``uc-kernel/kernels/mojo_moe_dispatch_bf16.py``.
_KERNEL_API = "mojo_moe_dispatch_bf16"
_FIXED_BS = 256
_FIXED_TOP_K = 8
_FIXED_R = _FIXED_BS * _FIXED_TOP_K  # 2048
_FIXED_H = 1024


def _count_expert_tokens(flat_top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``num_experts``-bin histogram, int32 -- mirrors the core helper at
    ``mojo_opset/core/operators/moe.py:_count_expert_tokens``."""
    return torch.bincount(
        flat_top_k_indices.long(), minlength=num_experts,
    ).to(dtype=torch.int32, device=flat_top_k_indices.device)


class UCMoEDispatch(MojoMoEDispatch):
    """UC backend wrapper for ``MojoMoEDispatch``.

    Only the bf16 ``hidden_states`` + fixed
    ``(BS=256, TOP_K=8, H=1024)`` contract is served by the wheel
    kernel; everything else raises ``NotImplementedError`` for the
    framework dispatcher.
    """

    supported_platforms_list = ["npu"]

    def forward(
        self,
        hidden_states: torch.Tensor,   # (BS, H)
        top_k_gates: torch.Tensor,     # (BS, top_k)
        top_k_indices: torch.Tensor,   # (BS, top_k), int32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ----- shape / dtype guards -----
        if hidden_states.dim() != 2:
            raise NotImplementedError(
                f"UC MoEDispatch expects 2D hidden_states (BS, H); "
                f"got shape {tuple(hidden_states.shape)}."
            )
        if top_k_indices.dim() != 2 or top_k_gates.dim() != 2:
            raise NotImplementedError(
                "UC MoEDispatch expects 2D top_k_gates and top_k_indices "
                "of shape (BS, top_k)."
            )
        if top_k_gates.shape != top_k_indices.shape:
            raise ValueError(
                f"top_k_gates {tuple(top_k_gates.shape)} and top_k_indices "
                f"{tuple(top_k_indices.shape)} must share the (BS, top_k) shape."
            )
        if hidden_states.shape[0] != top_k_indices.shape[0]:
            raise ValueError(
                f"hidden_states BS {hidden_states.shape[0]} must match "
                f"top_k_indices BS {top_k_indices.shape[0]}."
            )
        if hidden_states.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"UC MoEDispatch only supports bf16 hidden_states; got {hidden_states.dtype}."
            )

        bs, h = hidden_states.shape
        top_k = top_k_indices.shape[-1]
        r = bs * top_k

        if (bs, top_k, h) != (_FIXED_BS, _FIXED_TOP_K, _FIXED_H):
            raise NotImplementedError(
                f"UC MoEDispatch wheel kernel is fixed-shape "
                f"(BS={_FIXED_BS}, TOP_K={_FIXED_TOP_K}, H={_FIXED_H}); "
                f"got (BS={bs}, TOP_K={top_k}, H={h})."
            )

        kernels = _uc_kernels()
        # Soft-query fallback per lessons §J.1: wheel may not yet
        # carry this API, fall through to the parent torch-native
        # implementation rather than crashing.  ``KernelRegistry``
        # (see ``uc_kernel/runtime.py``) does not expose ``.get`` so we
        # consult ``.keys()`` first instead of letting ``__getitem__``
        # raise.
        if _KERNEL_API not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        kernel_fn = kernels[_KERNEL_API]

        device = hidden_states.device

        # ----- host arithmetic (mirrors MojoMoEDispatch.forward in
        # mojo_opset/core/operators/moe.py:212-248) -----
        flat_top_k_indices = top_k_indices.reshape(-1).contiguous()
        # ``batch_token_indices[i] = i // top_k`` -- which original token
        # the i-th (token, k) pair came from.
        batch_token_indices = (
            torch.arange(0, bs, device=device, dtype=top_k_indices.dtype)
            .unsqueeze(1)
            .repeat(1, top_k)
            .flatten()
        )
        # Sort to match the parent (``MojoMoEDispatch.forward`` at
        # ``mojo_opset/core/operators/moe.py:241``) bit-for-bit -- the
        # parent uses the default (non-stable) ``.sort()`` so equal
        # expert ids may be permuted; passing ``stable=True`` here
        # would silently re-order ``token_indices`` / ``sorted_gates``
        # relative to the reference even though the dispatch is
        # semantically correct (just a different valid permutation
        # within each expert group).  We mirror the parent so naive
        # bit-exact diff testing passes.
        sorted_experts, expert_sort_indices = torch.sort(flat_top_k_indices)
        del sorted_experts  # not needed downstream

        # ``token_indices[r]`` = original token id whose (token, k) pair
        # landed in sorted slot r.  This is one of the op's outputs AND
        # the gather index the wheel kernel reads.
        token_indices = batch_token_indices[expert_sort_indices].to(
            dtype=torch.int32, copy=False,
        ).contiguous()

        # ``sorted_gates`` and ``tokens_per_expert`` are pure host gathers.
        flat_top_k_gates = top_k_gates.reshape(-1, 1).contiguous()
        sorted_gates = flat_top_k_gates[expert_sort_indices, :]
        tokens_per_expert = _count_expert_tokens(flat_top_k_indices, self.num_experts)

        # ----- wheel kernel call (block GATHER) -----
        # ``hidden_states`` must be contiguous on the (BS, H) layout
        # so the gather DMA address arithmetic ``base + idx * H * 2``
        # is correct.  Most callers already pass contiguous tensors;
        # this is a no-op fast path then.
        hidden_states_c = hidden_states.contiguous()
        sorted_hidden_states = torch.empty(
            (r, h), dtype=hidden_states.dtype, device=device,
        )
        kernel_fn(hidden_states_c, token_indices, sorted_hidden_states)

        return sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices
