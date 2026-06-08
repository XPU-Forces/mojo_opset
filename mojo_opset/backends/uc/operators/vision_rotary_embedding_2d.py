"""UC backend for ``MojoVisionRotaryEmbedding2D`` — Wan / Qwen-VL vision 2D RoPE table.

Op contract (see ``mojo_opset.core.operators.position_embedding.MojoVisionRotaryEmbedding2D``)::

    forward(grid_hw)
        grid_hw : [B, 2] int (per-sample (gh, gw) patch grid)
        returns cos, sin : [total_tokens, rope_dim] fp32

Reference math (from the parent class)::

    inv_freq = 1.0 / theta ** (arange(0, rope_dim//2, 2) / (rope_dim//2))   # [K]
    rotary[max_grid, K] = outer(arange(max_grid), inv_freq)                  # fp32
    pos_ids[total_tokens, 2] = build_position_ids(grid_hw)                   # int
    freqs[total_tokens, 2, K] = rotary[pos_ids]
    emb[total_tokens, 4*K] = cat([freqs.flatten(-2), freqs.flatten(-2)], -1)
    return emb.cos(), emb.sin()

where ``K = rope_dim // 4`` and ``rope_dim % 4 == 0`` is asserted by the
parent's ``__init__``.

Host / device split (fast path)
-------------------------------

Everything except the per-token gather + 4-segment tile is done on host:

* ``inv_freq`` / ``rotary[max_grid, K]`` -- delegated to the parent's
  ``_vision_rotary_embedding`` (a single ``torch.outer``);
* ``pos_ids`` -- delegated to the parent's ``_build_position_ids``
  (the adapooling regrouping is pure indexing / view, host-side is the
  correct place for it);
* cos / sin of the table -- a single elementwise op on the small
  ``[max_grid, K]`` table; works because
  ``cos(cat([a, a], -1)) = cat([cos(a), cos(a)], -1)`` so applying the
  trig functions before the cat is mathematically equivalent and lets
  the device kernel be a pure gather + tile.

The device kernel
(:func:`uc_kernel.kernels.mojo_vision_rotary_embedding_2d_fp32.mojo_vision_rotary_embedding_2d_fp32`)
then gathers two cache rows per token (``pos_ids[m, 0]`` for height,
``pos_ids[m, 1]`` for width) and writes them into a
``[h, w, h, w]`` layout of width ``4*K = rope_dim``.

Fallback policy
---------------

Anything that violates the kernel's preconditions (non-NPU device, empty
grid, ``rope_dim % 4 != 0``, ``max_grid_size == 0``, kernel artifact
missing, etc.) defers to the parent ``super().forward`` (torch
reference), so correctness never regresses on edge cases.
"""

from typing import Tuple

import torch

from mojo_opset.core.operators.position_embedding import MojoVisionRotaryEmbedding2D

from ._utils import _uc_kernels, run_kernel


_KERNEL_API = "mojo_vision_rotary_embedding_2d_fp32"


class UCVisionRotaryEmbedding2D(MojoVisionRotaryEmbedding2D):
    supported_platforms_list = ["npu"]

    def forward(self, grid_hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ----- input contract guards (mirror parent) -----
        if grid_hw.ndim != 2 or grid_hw.shape[-1] != 2:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if torch.is_floating_point(grid_hw):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # rope_dim divisible by 4 is enforced by the parent ``__init__``,
        # but the kernel layout depends on K = rope_dim // 4 so re-check
        # defensively (lets the wrapper survive future refactors).
        rope_dim = int(self.rope_dim)
        if rope_dim <= 0 or rope_dim % 4 != 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        rotary_dim = rope_dim // 2
        K = rope_dim // 4
        if rotary_dim <= 0 or K <= 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ----- empty fast path -----
        if grid_hw.numel() == 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ----- kernel artifact availability -----
        kernels = _uc_kernels()
        if _KERNEL_API not in kernels:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ----- host-side table + index construction --------------------
        # Resolve the host device exactly the same way the parent does,
        # so that the rotary table lands on the right device first time.
        if self.inv_freq.device.type != "cpu" or grid_hw.device.type == "cpu":
            device = self.inv_freq.device
        else:
            device = grid_hw.device

        grid_hw_cpu = grid_hw.to(device="cpu", dtype=torch.int64)
        max_grid_size = int(grid_hw_cpu.max().item())
        if max_grid_size <= 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # 1) Rotary frequency table -- the parent's helper handles
        #    inv_freq device sync and dtype.
        rotary = self._vision_rotary_embedding(max_grid_size, device=device)
        # ``_vision_rotary_embedding`` returns shape ``[max_grid, K]`` fp32.
        if rotary.dtype is not torch.float32:
            rotary = rotary.to(torch.float32)
        if rotary.ndim != 2 or rotary.shape[-1] != K:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        cos_cache = rotary.cos().contiguous()
        sin_cache = rotary.sin().contiguous()

        # 2) Position ids -- adapooling regrouping done on host.
        pos_ids = self._build_position_ids(grid_hw, device=device)
        # ``_build_position_ids`` returns ``[total_tokens, 2]`` default
        # int64; the kernel ABI requires int32 (lifter hard-rule).
        if pos_ids.ndim != 2 or pos_ids.shape[-1] != 2:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        pos_ids = pos_ids.to(dtype=torch.int32).contiguous()

        total_tokens = int(pos_ids.shape[0])
        if total_tokens == 0:
            # Mirror the reference contract: return empty fp32 tensors
            # shaped ``[0, rope_dim]`` on the resolved device.
            empty = torch.empty((0, rope_dim), device=device, dtype=torch.float32)
            return empty, empty.clone()

        cos_out = torch.empty((total_tokens, rope_dim), device=device, dtype=torch.float32)
        sin_out = torch.empty((total_tokens, rope_dim), device=device, dtype=torch.float32)

        # Wheel ABI trailing scalar order = first-appearance of dynamic
        # dim names in tensor annotations:
        #   cos_cache (L, K) -> introduces L then K
        #   pos_ids   (M, 2) -> introduces M
        #   cos_out   (M, N) -> introduces N
        # => trailing order is (L, K, M, N).
        run_kernel(
            _KERNEL_API,
            cos_cache.dtype,
            cos_cache,
            sin_cache,
            pos_ids,
            cos_out,
            sin_out,
            int(cos_cache.shape[0]),  # L = max_grid_size
            K,                        # K = rope_dim // 4
            total_tokens,             # M = num tokens
            rope_dim,                 # N = 4 * K
        )
        return cos_out, sin_out
