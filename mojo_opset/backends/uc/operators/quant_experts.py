"""UC backend for ``MojoQuantExperts``.

Pipeline (per call):

    sorted_hidden (M_tot, H) bf16
      → up_proj_quantize  → (x_int8 (M_tot, H), x_scale (M_tot, 1) fp32)
      → per-expert fc1 via ``mojo_quant_gemm_bf16``:
            x_int8[e] @ up_w[e].T → fc1_out (m_e, 2*I) bf16
      → host SwiGLU in fp32 (gate=left half, up=right half):  silu(gate)*up
      → down_proj_quantize  → (y_int8 (M_tot, I), y_scale (M_tot, 1) fp32)
      → per-expert fc2 via ``mojo_quant_gemm_bf16``:
            y_int8[e] @ down_w[e].T → out (m_e, H) bf16
      → concat along dim 0

Strict gating (any mismatch falls back to ``super().forward()``):
- ``sorted_hidden_states.dtype == torch.bfloat16``
- ``quant_dtype == torch.int8``
- ``up_weight_dtype == down_weight_dtype == torch.int8`` (no int4 / packed)
- ``up_quant_group_size <= 0`` and ``down_quant_group_size <= 0``  (per-channel)
- ``activation == "swiglu"``
- ``mojo_quant_gemm_bf16`` available in the installed wheel

P2-32 optimizations vs original wrapper (host-side, no kernel change):
1. **Weight transpose cache** — `up_proj_weight` / `down_proj_weight` are
   nn.Parameter (effectively read-only between weight loads).  The original
   wrapper did ``weight[e].transpose(0,1).contiguous()`` inside every
   per-expert ``_gemm`` call (2 × num_experts redundant int8 copies per fwd,
   ~30 MB for typical (16 experts, H=512, I=1280)).  We now materialize the
   (E, K, N) transposed buffer once and version-check by ``(data_ptr, shape,
   stride)`` so weight reloads invalidate the cache automatically.
2. **fp32 weight-scale cache** — analogous; the bf16 ``nn.Parameter`` is
   cast to fp32 once per checkpoint instead of per expert per fwd.
3. **Single ``tokens_per_expert.tolist()`` per fwd** — original called
   ``.tolist()`` four times (D2H sync each).
4. **No redundant ``.contiguous()``** — ``torch.split`` of a contiguous
   tensor along dim 0 yields contiguous chunks; we only call ``.contiguous``
   when shape gymnastics actually requires it.
5. **Pre-flattened fp32 scale** — ``x_scale.reshape(-1).float().contiguous()``
   is done once before ``torch.split`` instead of inside ``_gemm`` per expert.
6. **Soft kernel lookup** — original used ``kernels.get(name)`` which the
   current ``KernelRegistry`` does not provide; switched to ``try/except
   KeyError`` per skill ``optimize-uc-kernel-perf`` / lessons §J.

Bug fix: original ``_quant_gemm_kernel`` raised ``AttributeError`` because
``KernelRegistry`` has no ``.get`` method — the wrapper would crash on first
call.  Replaced with ``try: kernels[name] / except KeyError: return None``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mojo_opset.core import MojoQuantExperts

from ._utils import _uc_kernels


_QUANT_GEMM_API = "mojo_quant_gemm_bf16"

# ``mojo_quant_gemm_bf16`` uses BLOCK_M=64 (see uc-kernel/kernels/mojo_quant_gemm.py).
# Per-expert ``m_e`` is typically << 64 in MoE workloads; we must pad rows to a
# multiple of this to avoid OOB reads on the input tile and OOB writes on the
# output tile.  Choosing the exact kernel BLOCK_M lets the kernel run a single
# program per (m_tile, n_tile) and slice the result back to ``m_e`` rows
# (post-process is a contiguous view, near-zero cost).
_KERNEL_BLOCK_M = 64


def _quant_gemm_kernel():
    """Soft-lookup of the int8 GEMM kernel; returns ``None`` if absent.

    ``KernelRegistry`` exposes ``__getitem__`` only (raises ``KeyError`` on
    miss) — we wrap it to match the wrapper-soft-lookup SOP from
    ``lessons-learned.md §J.2`` so the wrapper can gracefully fall back to
    the torch reference path when the wheel does not include the kernel.
    """
    try:
        return _uc_kernels()[_QUANT_GEMM_API]
    except (KeyError, ImportError):
        return None


def _version_key(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()))


class UCQuantExperts(MojoQuantExperts):
    """UC backend for ``MojoQuantExperts``.

    Drives a host-side per-expert loop over ``mojo_quant_gemm_bf16`` plus
    a single fused SwiGLU step in fp32 (mirrors the torch reference).
    """

    supported_platforms_list = ["npu"]

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _can_use_uc_path(self, sorted_hidden_states: torch.Tensor) -> bool:
        if sorted_hidden_states.dtype != torch.bfloat16:
            return False
        if self.quant_dtype != torch.int8:
            return False
        if self.up_weight_dtype != torch.int8 or self.down_weight_dtype != torch.int8:
            return False
        if self.up_quant_group_size > 0 or self.down_quant_group_size > 0:
            return False
        if self.activation != "swiglu":
            return False
        if _quant_gemm_kernel() is None:
            return False
        return True

    # ------------------------------------------------------------------
    # Weight caches (one-time materialization per checkpoint)
    # ------------------------------------------------------------------

    def _weight_kn_cached(self, weight: torch.Tensor, slot: str) -> torch.Tensor:
        """Return the (E, K, N) transposed-and-contiguous int8 weight.

        ``MojoQuantExperts`` stores weights as ``(E, N, K)`` (output channels
        outer); the GEMM kernel ABI wants ``(K, N)``.  We materialize the
        per-expert ``transpose(-2, -1).contiguous()`` once and invalidate on
        any change to ``data_ptr / shape / stride``.
        """
        cache_attr = f"_uc_weight_kn_{slot}"
        ver_attr = f"_uc_weight_kn_{slot}_ver"
        version = _version_key(weight)
        if getattr(self, ver_attr, None) != version:
            cached = weight.transpose(-2, -1).contiguous()
            setattr(self, cache_attr, cached)
            setattr(self, ver_attr, version)
        return getattr(self, cache_attr)

    def _weight_scale_fp32_cached(self, weight_scale: torch.Tensor, slot: str) -> torch.Tensor:
        """Return the (E, N) fp32-and-contiguous weight scale.

        Stored as bf16 nn.Parameter; the GEMM kernel ABI wants fp32.
        """
        cache_attr = f"_uc_ws_fp32_{slot}"
        ver_attr = f"_uc_ws_fp32_{slot}_ver"
        version = _version_key(weight_scale)
        if getattr(self, ver_attr, None) != version:
            cached = weight_scale.to(dtype=torch.float32).contiguous()
            setattr(self, cache_attr, cached)
            setattr(self, ver_attr, version)
        return getattr(self, cache_attr)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not self._can_use_uc_path(sorted_hidden_states):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        device = sorted_hidden_states.device
        out_dtype = sorted_hidden_states.dtype

        if sorted_hidden_states.numel() == 0:
            return torch.empty(
                (0, self.hidden_size), dtype=out_dtype, device=device,
            )

        if sorted_hidden_states.dim() != 2:
            raise ValueError(
                f"UCQuantExperts expects 2D sorted_hidden_states, "
                f"got {tuple(sorted_hidden_states.shape)}."
            )

        kernel = _quant_gemm_kernel()
        assert kernel is not None  # gated above

        # Cache transposed weight and fp32 scale (one-time work per checkpoint).
        up_w_kn = self._weight_kn_cached(self.up_proj_weight, "up")
        down_w_kn = self._weight_kn_cached(self.down_proj_weight, "down")
        up_ws_fp32 = self._weight_scale_fp32_cached(self.up_proj_weight_scale, "up")
        down_ws_fp32 = self._weight_scale_fp32_cached(self.down_proj_weight_scale, "down")

        # Step 1: activation quant for fc1 (handled by MojoMoEDynamicQuant UC).
        x_int8, x_scale = self.up_proj_quantize(sorted_hidden_states, tokens_per_expert)

        # Pre-flatten + cast input_scale to fp32 once; subsequent splits inherit
        # contiguity (split along dim 0 of a contiguous tensor is contiguous).
        x_scale_fp32 = x_scale.reshape(-1).to(torch.float32).contiguous()

        # Single D2H sync for the token-count list; cache num_experts.
        token_counts = tokens_per_expert.tolist()
        num_experts = len(token_counts)
        intermediate_size = self.intermediate_size

        x_int8_chunks = torch.split(x_int8, token_counts, dim=0)
        x_scale_chunks = torch.split(x_scale_fp32, token_counts, dim=0)

        # Step 2: fc1 (int8 GEMM per expert) + fused SwiGLU on host in fp32.
        activated_outs: list[torch.Tensor] = []
        for e in range(num_experts):
            m_e = token_counts[e]
            if m_e == 0:
                activated_outs.append(
                    torch.empty((0, intermediate_size), dtype=out_dtype, device=device)
                )
                continue

            x_i = x_int8_chunks[e]
            s_i = x_scale_chunks[e]
            w_i = up_w_kn[e]                  # (H, 2*I) int8, contiguous slice
            ws_i = up_ws_fp32[e]              # (2*I,)   fp32, contiguous slice
            K_e = x_i.shape[1]
            N_e = w_i.shape[1]

            # Pad rows to BLOCK_M to satisfy kernel tile bounds (BLOCK_M=64).
            m_pad = ((m_e + _KERNEL_BLOCK_M - 1) // _KERNEL_BLOCK_M) * _KERNEL_BLOCK_M
            if m_pad == m_e:
                x_pad, s_pad = x_i, s_i
            else:
                x_pad = torch.zeros((m_pad, K_e), dtype=torch.int8, device=device)
                x_pad[:m_e].copy_(x_i)
                s_pad = torch.zeros((m_pad,), dtype=torch.float32, device=device)
                s_pad[:m_e].copy_(s_i)

            fc1_pad = torch.empty((m_pad, N_e), dtype=torch.bfloat16, device=device)
            kernel(x_pad, w_i, s_pad, ws_i, fc1_pad, m_pad, K_e, N_e)
            fc1 = fc1_pad[:m_e]  # contiguous view of first m_e rows

            # SwiGLU in fp32 for numerical parity with reference; cast back
            # to bf16 so the activation re-enters the UC dynamic-quant kernel
            # (which expects bf16/fp16 inputs).
            gate, up = fc1.float().chunk(2, dim=-1)
            activated_outs.append((F.silu(gate) * up).to(out_dtype))

        activated = torch.cat(activated_outs, dim=0)

        # Step 3: activation quant for fc2.
        y_int8, y_scale = self.down_proj_quantize(activated, tokens_per_expert)
        y_scale_fp32 = y_scale.reshape(-1).to(torch.float32).contiguous()
        y_int8_chunks = torch.split(y_int8, token_counts, dim=0)
        y_scale_chunks = torch.split(y_scale_fp32, token_counts, dim=0)

        # Step 4: fc2 (int8 GEMM per expert).
        outputs: list[torch.Tensor] = []
        hidden_size = self.hidden_size
        for e in range(num_experts):
            m_e = token_counts[e]
            if m_e == 0:
                outputs.append(
                    torch.empty((0, hidden_size), dtype=out_dtype, device=device)
                )
                continue

            x_i = y_int8_chunks[e]
            s_i = y_scale_chunks[e]
            w_i = down_w_kn[e]                # (I, H) int8, contiguous slice
            ws_i = down_ws_fp32[e]            # (H,)   fp32, contiguous slice
            K_e = x_i.shape[1]
            N_e = w_i.shape[1]

            m_pad = ((m_e + _KERNEL_BLOCK_M - 1) // _KERNEL_BLOCK_M) * _KERNEL_BLOCK_M
            if m_pad == m_e:
                x_pad, s_pad = x_i, s_i
            else:
                x_pad = torch.zeros((m_pad, K_e), dtype=torch.int8, device=device)
                x_pad[:m_e].copy_(x_i)
                s_pad = torch.zeros((m_pad,), dtype=torch.float32, device=device)
                s_pad[:m_e].copy_(s_i)

            fc2_pad = torch.empty((m_pad, N_e), dtype=torch.bfloat16, device=device)
            kernel(x_pad, w_i, s_pad, ws_i, fc2_pad, m_pad, K_e, N_e)
            outputs.append(fc2_pad[:m_e])

        return torch.cat(outputs, dim=0)
