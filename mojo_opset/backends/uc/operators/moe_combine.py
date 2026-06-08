"""UC backend for ``MojoMoECombine``.

Maps the high-level MoE token re-combine op
(``mojo_opset/core/operators/moe.py:519``) onto wheel-side block-GATHER
kernels.  Implements the per-output-token re-combine::

    combined[t, :] = output_buffer[t, :]
                   + sum_{k in [0, TOP_K)} weighted_expert_outputs[expert_slot[t, k], :]

where ``weighted_expert_outputs`` is the host-side product
``(expert_outputs * sorted_gates)`` (only when ``multiply_by_gates`` is
``True``; otherwise the expert outputs are forwarded as-is), and
``expert_slot`` is the inverse permutation of ``token_indices`` reshaped
to ``(BS, TOP_K)``.

P2-30 split-call design
-----------------------

A single fixed-shape kernel that unrolls all ``TOP_K=8`` gather +
cast + acc-add iterations was empirically observed to deadlock the
AscendC runtime when ``weighted_expert_outputs`` carries non-zero data
(verified on 910B devices 0-15; chain depth ≥ 4 in
``mojo_moe_combine_bf16.py`` reliably hangs ``WaitFlag<V_MTE2>(0)`` on
the reused gather UB BasePtr).  The threshold is independent of
NUM_PROGRAMS and ``T.serial`` vs Python unroll.

Workaround: split the K=8 reduction over **3 fixed-shape kernel calls**::

    call 1 (KP=3):  combined  = output_buffer + sum_{k in [0,3)} weighted[es[t,k]]
    call 2 (KP=3):  combined += sum_{k in [3,6)} weighted[es[t,k]]
    call 3 (KP=2):  combined += sum_{k in [6,8)} weighted[es[t,k]]

The wheel exports two kernels accordingly:

* ``mojo_moe_combine_kp3_bf16`` — 3 contributions per call (calls 1, 2)
* ``mojo_moe_combine_kp2_bf16`` — 2 contributions per call (call 3)

Calls 2 and 3 reuse ``combined`` as their ``output_buffer`` (running
partial sum) so the bf16 cast happens only at the final ``combined``
write of each call — bf16 round-off across the chain is bounded by
``3 * eps_bf16(sum)`` (still well within the test tolerance).

Wheel kernel ABI (fixed-shape, no ``T.dynamic``)::

    mojo_moe_combine_kpN_bf16(output_buffer, weighted_expert_outputs,
                              expert_slot_kpN, combined)
        output_buffer           : bf16  (BS, H)
        weighted_expert_outputs : bf16  (R = BS * TOP_K, H)
        expert_slot_kpN         : int32 (BS, KP)         # KP=3 or 2
        combined                : bf16  (BS, H)

with the fixed bring-up contract ``BS = 256``, ``TOP_K = 8``,
``H = 1024`` (matching ``num_tokens=256, top_k=8, hidden_size=1024``
in ``mojo_opset/tests/accuracy/operators/test_moe.py``).  Anything
outside this contract raises ``NotImplementedError`` so the framework
can dispatch a different backend.
"""

import torch

from mojo_opset.core import MojoMoECombine

from ._utils import _uc_kernels


# Must match the wheel kernel constants in
# ``uc-kernel/kernels/mojo_moe_combine_bf16.py``.
_KERNEL_KP3 = "mojo_moe_combine_kp3_bf16"
_KERNEL_KP2 = "mojo_moe_combine_kp2_bf16"
_FIXED_BS = 256
_FIXED_TOP_K = 8
_FIXED_H = 1024
_FIXED_R = _FIXED_BS * _FIXED_TOP_K  # 2048

# K-per-call split (sums to TOP_K=8).
_KP_SPLITS = (3, 3, 2)


class UCMoECombine(MojoMoECombine):
    """UC backend wrapper for ``MojoMoECombine``.

    Only the bf16 ``expert_outputs`` + fixed (BS=256, TOP_K=8, H=1024)
    contract is served by the wheel kernel; everything else raises
    ``NotImplementedError`` for the framework dispatcher.
    """

    supported_platforms_list = ["npu"]

    def forward(
        self,
        output_buffer: torch.Tensor,    # (BS, H)
        expert_outputs: torch.Tensor,   # (R = BS * TOP_K, H)
        sorted_gates: torch.Tensor,     # (R, 1) or (R,)
        token_indices: torch.Tensor,    # (R,), int32
    ) -> torch.Tensor:
        # ----- shape / dtype guards -----
        if output_buffer.dim() != 2 or expert_outputs.dim() != 2:
            raise NotImplementedError(
                "UC MoECombine expects 2D output_buffer (BS, H) and 2D "
                f"expert_outputs (R, H); got shapes {tuple(output_buffer.shape)} "
                f"and {tuple(expert_outputs.shape)}."
            )
        if output_buffer.shape[1] != expert_outputs.shape[1]:
            raise ValueError(
                f"output_buffer hidden dim {output_buffer.shape[1]} must match "
                f"expert_outputs hidden dim {expert_outputs.shape[1]}."
            )
        if token_indices.dim() != 1:
            raise NotImplementedError(
                "UC MoECombine expects 1D token_indices; got "
                f"shape {tuple(token_indices.shape)}."
            )
        if token_indices.shape[0] != expert_outputs.shape[0]:
            raise ValueError(
                f"token_indices length {token_indices.shape[0]} must match "
                f"expert_outputs row count {expert_outputs.shape[0]}."
            )
        if expert_outputs.dtype != torch.bfloat16 or output_buffer.dtype != torch.bfloat16:
            raise NotImplementedError(
                "UC MoECombine only supports bf16 output_buffer / expert_outputs; "
                f"got {output_buffer.dtype} / {expert_outputs.dtype}."
            )

        bs, h = output_buffer.shape
        r = expert_outputs.shape[0]
        if r % bs != 0:
            raise ValueError(
                f"expert_outputs row count R={r} must be divisible by output "
                f"row count BS={bs} (R = BS * TOP_K invariant of MoE dispatch)."
            )
        top_k = r // bs

        if (bs, top_k, h) != (_FIXED_BS, _FIXED_TOP_K, _FIXED_H):
            raise NotImplementedError(
                f"UC MoECombine wheel kernel is fixed-shape "
                f"(BS={_FIXED_BS}, TOP_K={_FIXED_TOP_K}, H={_FIXED_H}); "
                f"got (BS={bs}, TOP_K={top_k}, H={h})."
            )

        # ``sorted_gates`` may arrive as (R,) or (R, 1); normalise to (R, 1)
        # so the broadcast multiply over (R, H) below stays unambiguous.
        if sorted_gates.dim() == 1:
            if sorted_gates.shape[0] != r:
                raise ValueError(
                    f"sorted_gates length {sorted_gates.shape[0]} must match "
                    f"R={r}."
                )
            gates_2d = sorted_gates.unsqueeze(-1)
        elif sorted_gates.dim() == 2:
            if sorted_gates.shape != (r, 1):
                raise NotImplementedError(
                    f"UC MoECombine expects sorted_gates of shape (R, 1) or "
                    f"(R,); got {tuple(sorted_gates.shape)}."
                )
            gates_2d = sorted_gates
        else:
            raise NotImplementedError(
                f"UC MoECombine expects 1D or 2D sorted_gates; got dim "
                f"{sorted_gates.dim()}."
            )

        kernels = _uc_kernels()
        # Verify both wheel kernels are present; if either is missing
        # (wheel out of date), fall back to the parent torch native.
        if _KERNEL_KP3 not in kernels.keys() or _KERNEL_KP2 not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        device = expert_outputs.device

        # ----- host arithmetic -----

        # Host-side weighted expert outputs.  Reference does the multiply
        # in fp32 inside ``scatter_reduce``; mirror that promotion so the
        # per-row product is computed in fp32 before the bf16 cast back
        # to the kernel's bf16 input plane.
        expert_outputs_contig = expert_outputs.contiguous()
        if self.multiply_by_gates:
            weighted = (
                expert_outputs_contig.float() * gates_2d.float()
            ).to(torch.bfloat16).contiguous()
        else:
            weighted = expert_outputs_contig

        # Inverse permutation: ``expert_slot[t, k]`` = the r-index in
        # [0, R) such that ``token_indices[r] == t``.  Since the MoE
        # dispatch step replicates every original token exactly
        # ``TOP_K`` times before sorting by expert id,
        # ``token_indices`` contains each value in [0, BS) exactly
        # ``TOP_K`` times.  A stable argsort gives a permutation whose
        # t-th block of ``TOP_K`` entries lists the r-indices for
        # token t.
        perm = torch.argsort(token_indices.to(torch.int64), stable=True)
        expert_slot = perm.reshape(bs, top_k).to(torch.int32).contiguous()

        # ``output_buffer`` may not be contiguous (the framework can
        # hand us a view); the kernel needs a dense (BS, H) bf16 plane.
        output_buffer_contig = output_buffer.contiguous()

        # ----- 3-call split: KP=3, KP=3, KP=2 -----
        # Call N reads its incoming partial sum from prev_partial and
        # writes the updated partial sum to ``combined``.  After the
        # loop ``combined`` holds the full K=8 reduce-sum + the
        # ``include_self`` ``output_buffer`` term.
        #
        # NOTE: input and output buffers must NOT alias the same tensor;
        # the kernel schedule (1 program per token) reads the row then
        # writes the row, but ``output_buffer`` and ``combined`` are
        # treated as independent DRAM args by the lifter — aliasing
        # them produces incorrect numerics (verified on 910B: aliased
        # buffers give max-abs-diff > 3.7 vs reference scatter_reduce).
        # Use a ping-pong between two scratch buffers across calls.
        buf_a = torch.empty((bs, h), dtype=torch.bfloat16, device=device)
        buf_b = torch.empty((bs, h), dtype=torch.bfloat16, device=device)
        scratch = (buf_a, buf_b)
        prev_partial = output_buffer_contig
        k_off = 0
        last_out = None
        for call_idx, kp in enumerate(_KP_SPLITS):
            api = _KERNEL_KP3 if kp == 3 else _KERNEL_KP2
            es_slice = expert_slot[:, k_off:k_off + kp].contiguous()
            out_buf = scratch[call_idx % 2]
            kernels[api](
                prev_partial,
                weighted,
                es_slice,
                out_buf,
            )
            prev_partial = out_buf
            last_out = out_buf
            k_off += kp

        assert k_off == top_k, (
            f"UC MoECombine KP split bug: processed {k_off} contributions, "
            f"expected {top_k}"
        )

        return last_out
