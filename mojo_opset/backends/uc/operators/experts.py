"""UC backend for ``MojoExperts`` (MoE grouped GEMM with SwiGLU).

Strategy
--------
The mojo reference op (``mojo_opset/core/operators/moe.py:251``) is a
host-side grouped GEMM pipeline:

    expert_inputs = split(sorted_hidden_states, tokens_per_expert)
    for i in range(num_experts):
        fc1_i      = F.linear(expert_inputs[i], up_proj_weight[i])      # (M_i, 2*I)
        gate, up   = fc1_i.chunk(2, dim=-1)                             # (M_i, I)
        active_i   = F.silu(gate) * up                                   # (M_i, I)
        fc2_i      = F.linear(active_i, down_proj_weight[i])             # (M_i, H)
    out            = cat(fc2_i)                                          # (sum(M_i), H)

uc-kernel does **not** ship a ``mojo_gemm_bf16`` (let alone a grouped
``mojo_experts``) kernel today, so a tilelang-uc kernel cannot be wired
in.  Per perf taxonomy class § F.8 (wheel-API surface insufficient) we
delegate the dense grouped matmul to the vendor-tuned
``torch_npu.npu_grouped_matmul`` op while keeping the rest of the schedule
on-device.  This is the same fallback pattern HCCL ops use (long-term
memory: "torch.distributed fallback 在 NPU 上底层走 HCCL").

The hot path collapses the original *N_experts*-iteration Python loop
(plus per-iteration ``.float()`` upcasts in the base class) into:

    1. pre-transpose ``up_proj_weight`` / ``down_proj_weight`` to
       ``[E, K, N]`` once and cache (NPU group_matmul expects ``A:[M,K] @
       B[E,K,N] -> C[M,N]``).
    2. one launch of ``npu_grouped_matmul`` for fused fc1.
    3. SwiGLU via the existing ``mojo_swiglu_bf16`` wheel API when present
       (else torch native bf16, no fp32 upcast).
    4. one launch of ``npu_grouped_matmul`` for fused fc2.

Fallback to ``MojoExperts.forward`` is kept for any out-of-scope geometry
(non-bf16 dtype, weight/input layout mismatch, ``activation != "swiglu"``).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from mojo_opset.core import MojoExperts

from ._utils import _uc_kernels


_SWIGLU_API = "mojo_swiglu_bf16"
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _resolve_kernel(api: str):
    """Return ``uc_kernel`` callable for ``api`` or ``None`` if missing.

    The deployed uc-kernel wheel may also reference APIs that were not
    actually built into the C++ extension (manifest / .so drift) — in
    that case ``uc_kernel.load()`` raises ``RuntimeError`` which we
    swallow and fall back to the torch path.
    """
    try:
        return _uc_kernels()[api]
    except (KeyError, RuntimeError):
        return None


def _has_npu_grouped_matmul() -> bool:
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch_npu, "npu_grouped_matmul")
    except ImportError:
        return False


def _validate(
    op: "UCExperts",
    sorted_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> bool:
    if op.activation != "swiglu":
        return False
    if sorted_hidden_states.dtype not in _SUPPORTED_DTYPES:
        return False
    if sorted_hidden_states.dim() != 2:
        return False
    if op.up_proj_weight.dtype != sorted_hidden_states.dtype:
        return False
    if op.down_proj_weight.dtype != sorted_hidden_states.dtype:
        return False
    if op.up_proj_weight.dim() != 3 or op.down_proj_weight.dim() != 3:
        return False
    if op.up_proj_weight.device != sorted_hidden_states.device:
        return False
    if op.down_proj_weight.device != sorted_hidden_states.device:
        return False

    num_experts = op.up_proj_weight.shape[0]
    if op.down_proj_weight.shape[0] != num_experts:
        return False
    if tokens_per_expert.numel() != num_experts:
        return False

    two_I_up = op.up_proj_weight.shape[1]
    H_up = op.up_proj_weight.shape[2]
    H_down = op.down_proj_weight.shape[1]
    I_down = op.down_proj_weight.shape[2]
    if two_I_up != 2 * I_down:
        return False
    if H_up != H_down:
        return False
    if sorted_hidden_states.shape[1] != H_up:
        return False
    return True


class UCExperts(MojoExperts):
    """UC-backend ``MojoExperts``.

    Hot path:
      * collapse per-expert Python loop into two ``npu_grouped_matmul``
        launches (split_item=3 → single fused output tensor);
      * stay in input dtype (no fp32 upcasts);
      * SwiGLU via ``mojo_swiglu_bf16`` wheel kernel when registered.

    Falls back to ``MojoExperts.forward`` when the geometry is out of
    scope or ``torch_npu.npu_grouped_matmul`` is unavailable.
    """

    supported_platforms_list = ["npu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cached transposed weights for npu_grouped_matmul.  The cache key
        # is ``(data_ptr, shape, stride, dtype, device)`` so we recompute
        # transparently after ``load_state_dict`` / weight reassignment.
        self._cached_up_w_T: Optional[torch.Tensor] = None
        self._cached_down_w_T: Optional[torch.Tensor] = None
        self._cached_up_key = None
        self._cached_down_key = None

    @staticmethod
    def _weight_key(t: torch.Tensor):
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

    def _get_up_w_T(self) -> torch.Tensor:
        # up_proj_weight: (E, 2I, H) -> (E, H, 2I) for A:(M,H) @ B:(H,2I).
        key = self._weight_key(self.up_proj_weight)
        if key != self._cached_up_key:
            self._cached_up_w_T = self.up_proj_weight.transpose(-2, -1).contiguous()
            self._cached_up_key = key
        return self._cached_up_w_T

    def _get_down_w_T(self) -> torch.Tensor:
        # down_proj_weight: (E, H, I) -> (E, I, H) for A:(M,I) @ B:(I,H).
        key = self._weight_key(self.down_proj_weight)
        if key != self._cached_down_key:
            self._cached_down_w_T = self.down_proj_weight.transpose(-2, -1).contiguous()
            self._cached_down_key = key
        return self._cached_down_w_T

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not _validate(self, sorted_hidden_states, tokens_per_expert):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not _has_npu_grouped_matmul():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        import torch_npu

        dtype = sorted_hidden_states.dtype
        device = sorted_hidden_states.device

        num_experts = self.up_proj_weight.shape[0]
        I_size = self.down_proj_weight.shape[2]
        H = self.up_proj_weight.shape[2]
        total_tokens = sorted_hidden_states.shape[0]

        if total_tokens == 0:
            return torch.empty((0, H), dtype=dtype, device=device)

        x = sorted_hidden_states.contiguous()
        up_w_T = self._get_up_w_T()      # (E, H, 2*I)
        down_w_T = self._get_down_w_T()  # (E, I, H)

        # group_list as cumulative sum on device (int64).  group_list_type=0
        # means values are cumsum of group sizes along the m-axis.
        counts_i64 = tokens_per_expert.detach().to(device=device, dtype=torch.int64)
        if counts_i64.numel() != num_experts:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        group_list = torch.cumsum(counts_i64, dim=0)

        # ---- fc1: x @ up_w_T  -> (M, 2*I) ----
        fc1_list = torch_npu.npu_grouped_matmul(
            [x],
            [up_w_T],
            group_list=group_list,
            split_item=3,        # single output tensor
            group_type=0,        # m-axis grouping
            group_list_type=0,   # cumsum form
            act_type=0,          # no fused activation
        )
        fc1 = fc1_list[0]        # (M, 2*I), dtype = input dtype

        # ---- SwiGLU: silu(gate) * up over (M, I) ----
        # ``chunk`` on dim=-1 gives non-contiguous views; ``mojo_swiglu_bf16``
        # demands contiguous inputs.
        gate, up_t = fc1.chunk(2, dim=-1)
        gate = gate.contiguous()
        up_t = up_t.contiguous()

        swiglu_kernel = _resolve_kernel(_SWIGLU_API) if dtype == torch.bfloat16 else None
        if swiglu_kernel is not None:
            activated = torch.empty_like(gate)
            swiglu_kernel(gate, up_t, activated, total_tokens, I_size)
        else:
            # bf16 / fp16 native silu: stay in input dtype, no fp32 upcast.
            activated = (F.silu(gate.float()) * up_t.float()).to(dtype).contiguous()

        # ---- fc2: activated @ down_w_T -> (M, H) ----
        out_list = torch_npu.npu_grouped_matmul(
            [activated],
            [down_w_T],
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
            act_type=0,
        )
        return out_list[0]
