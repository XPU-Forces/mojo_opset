"""UC backend for ``MojoLayerNormQuant``.

Fused LayerNorm + dynamic per-token int8 quantization.  Dispatches between
two kernel variants:

* ``mojo_layernorm_quant_{bf16,fp16}`` — full pipeline with per-channel
  ``smooth_scale`` multiply.
* ``mojo_layernorm_quant_nosmooth_{bf16,fp16}`` — variant for the common
  case where the caller does not supply ``smooth_scale`` (default forward).
  Eliminates the per-tile ``smooth_scale`` DRAM load and the
  ``normed * smooth_scale`` parallel op, ~11 % faster.

The wrapper validates the fast path (symmetric int8 quant, affine weight +
bias present, contiguous bf16/fp16 input, ``norm_size`` matching) and falls
back to ``MojoLayerNormQuant.forward`` for any unsupported config (notably
``float8_e4m3fn`` quant, non-affine layernorm, mismatched dtype, etc.).
"""

from typing import Optional, Tuple

import torch

from mojo_opset.core import MojoLayerNormQuant

from ._utils import _matrix_shape
from ._utils import _typed_api
from ._utils import _uc_kernels


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
# Kernel tile geometry (see uc-kernel/kernels/mojo_layernorm_quant_*.py)
_KERNEL_TILE_X = 8  # row block per program — kernel reads up to X rows OOB
_KERNEL_TILE_Y = 512  # col block per inner iter — kernel reads up to Y cols OOB


def _can_use_kernel(
    op: MojoLayerNormQuant,
    hidden_state: torch.Tensor,
    smooth_scale: Optional[torch.Tensor],
) -> bool:
    if not isinstance(hidden_state, torch.Tensor):
        return False
    if hidden_state.dtype not in _SUPPORTED_DTYPES:
        return False
    if hidden_state.dim() < 1:
        return False
    if hidden_state.shape[-1] != op.norm_size:
        return False
    if op.norm_size <= 0:
        return False
    if op.quant_dtype != torch.int8:
        return False
    if not op.symmetric:
        return False
    if op.q_max != 127 or op.q_min != -128:
        return False
    if not op.elementwise_affine:
        return False
    if op.weight is None or op.bias is None:
        return False
    if op.weight.shape != (op.norm_size,) or op.bias.shape != (op.norm_size,):
        return False
    if op.weight.device != hidden_state.device or op.bias.device != hidden_state.device:
        return False
    # Per-token rows = numel // last_dim. Each kernel program loads X rows + Y cols
    # at a time; without M >= X and N >= Y the kernel reads OOB and pollutes the
    # mean/var stats. Fall back to torch for short rows / narrow cols.
    rows = hidden_state.numel() // op.norm_size
    if rows < _KERNEL_TILE_X:
        return False
    if op.norm_size < _KERNEL_TILE_Y:
        return False
    if smooth_scale is not None:
        if not isinstance(smooth_scale, torch.Tensor):
            return False
        if smooth_scale.numel() != op.norm_size:
            return False
        if smooth_scale.device != hidden_state.device:
            return False
    return True


class UCLayerNormQuant(MojoLayerNormQuant):
    supported_platforms_list = ["npu"]

    # Cached per-instance dtype-casted weight/bias (most callers reuse the same
    # nn.Parameter across forward calls; recasting each call costs ~10us of
    # tensor-alloc overhead at M=2k N=2k).
    _cached_weight_dtype: Optional[torch.dtype] = None
    _cached_weight: Optional[torch.Tensor] = None
    _cached_bias: Optional[torch.Tensor] = None
    _cached_weight_id: int = -1
    _cached_bias_id: int = -1
    # P3-04 Step 0: smooth_scale wrapper-side fold cache.
    # Algebra: `((x-μ)·rstd·w + b) · s == (x-μ)·rstd·(w·s) + (b·s)`,
    # so when caller passes smooth_scale we can pre-fold (w'=w·s, b'=b·s)
    # in fp32 (1-ulp safer per R3-02 review item 1) then dispatch the cheaper
    # nosmooth kernel. Cache key is a 3-tuple of (data_ptr, shape, stride,
    # dtype, device) for (weight, bias, smooth_scale) so that param-rebuild or
    # device move invalidates safely (avoids id()-collision pitfall noted in
    # uc-best-practices §B.3 caveat).
    _fold_cache: Optional[dict] = None

    @staticmethod
    def _tensor_fingerprint(t: torch.Tensor) -> tuple:
        return (
            t.data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            t.device,
        )

    def _get_folded_wb(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        smooth_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(w', b') = (w·s, b·s)`` cast to ``out_dtype``; cached."""
        if self._fold_cache is None:
            # Per-instance dict (not class-level) — avoid cross-module leak.
            object.__setattr__(self, "_fold_cache", {})
        key = (
            self._tensor_fingerprint(weight),
            self._tensor_fingerprint(bias),
            self._tensor_fingerprint(smooth_scale),
            out_dtype,
        )
        cached = self._fold_cache.get(key)
        if cached is not None:
            return cached
        # R3-02 nit #1: compute in fp32 then cast to reduce 1 ulp drift vs
        # casting first then multiplying in bf16/fp16.
        w_f32 = weight.float() if weight.dtype != torch.float32 else weight
        b_f32 = bias.float() if bias.dtype != torch.float32 else bias
        s_f32 = smooth_scale.float() if smooth_scale.dtype != torch.float32 else smooth_scale
        s_f32_flat = s_f32.reshape(-1)
        w_folded = (w_f32 * s_f32_flat).to(out_dtype).contiguous()
        b_folded = (b_f32 * s_f32_flat).to(out_dtype).contiguous()
        self._fold_cache[key] = (w_folded, b_folded)
        return w_folded, b_folded

    def _cast_param(self, attr: torch.Tensor, attr_name: str, dtype: torch.dtype) -> torch.Tensor:
        """Return ``attr`` cast to ``dtype`` & contiguous; cache hit avoids re-alloc."""
        cache_id_attr = f"_cached_{attr_name}_id"
        cache_attr = f"_cached_{attr_name}"
        cached_id = getattr(self, cache_id_attr, -1)
        cached = getattr(self, cache_attr, None)
        if (
            cached is not None
            and id(attr) == cached_id
            and cached.dtype == dtype
            and cached.device == attr.device
        ):
            return cached
        if attr.dtype != dtype:
            casted = attr.to(dtype=dtype).contiguous()
        elif not attr.is_contiguous():
            casted = attr.contiguous()
        else:
            casted = attr
        # nn.Module.__setattr__ would try to register Tensors as Parameters/Buffers
        # when the attribute name pre-exists as a class-level default; bypass via
        # object.__setattr__ to keep the cache dict-like.
        object.__setattr__(self, cache_attr, casted)
        object.__setattr__(self, cache_id_attr, id(attr))
        return casted

    def forward(
        self,
        hidden_state: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # P3-04 Master P0 SEV-1 build-env regression (2026-06-06): isolated
        # uc_kernel build silently miscompiles all LNQ kernels (audit shows
        # all 16 col-blocks at M=N=2048 have max_int_diff 50-76 vs ref;
        # scale_rel ≈ 1.17×; signature matches P3-05 sibling worker finding).
        # Per user pref §18 + master accuracy gate: when active kernel fails
        # accuracy audit, wrapper MUST unconditional super().forward() fall
        # back. Re-enable once compiler-side build regression is fixed and
        # audit pipeline re-validates per-128-col-block diff.
        raise NotImplementedError(
            "UC backend cannot service this call (shape/dtype/contract not "
            "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
            "(2026-06-08), this wrapper does not silently fall back to torch — "
            "use TTX / torch_npu / torch_native backend for unsupported inputs."
        )
        # ───── kernel fast-path retained below for re-enable after fix ─────
        if not _can_use_kernel(self, hidden_state, smooth_scale):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if hidden_state.numel() == 0:
            empty_q = torch.empty(hidden_state.shape, dtype=self.quant_dtype, device=hidden_state.device)
            empty_s = torch.empty(hidden_state.shape[:-1] + (1,), dtype=torch.float32, device=hidden_state.device)
            return empty_q, empty_s

        dtype = hidden_state.dtype
        # P3-04 Step 0: when smooth_scale provided, fold it into (w', b')
        # (fp32 intermediate) and dispatch the cheaper nosmooth kernel —
        # algebraically equivalent (see _get_folded_wb docstring). Wins ~11%
        # vs the smooth kernel that pays an extra DRAM read of smooth_scale +
        # one fused-multiply per row tile (P2-06 measured smooth vs nosmooth
        # delta).
        use_fold = smooth_scale is not None
        # P3-04 Step 2 *attempted* an n2048 shape-specialised variant with
        # full-row UB-resident `normed_full (8, 2048) f32` (64 KB) + `x_io_full`
        # (32 KB) + `y_io_full` (16 KB) — at the per-vec-core 192 KB UB ceiling
        # after mix-mode implicit double-buffer overhead, it ran the kernel
        # into runtime AIV error 507035 (vector core exception, UB overflow).
        # Variant is built and registered (`mojo_layernorm_quant_nosmooth_n2048_*`)
        # so subsequent compiler-side P0 fixes (lift-time UB budget check,
        # lifter offset-indexed-load / UB→UB chunk copy unlock) can let it run.
        # For now keep the dispatch off-path; perf-debug § cannot-opt entry
        # tracks the blocker.
        use_n2048 = False  # Step 2 BLOCKED — see perf-debug doc
        # Pick the cheaper kernel: n2048 nosmooth > nosmooth > smooth.
        try:
            if use_n2048:
                api = _typed_api("mojo_layernorm_quant_nosmooth_n2048", dtype)
            elif smooth_scale is None or use_fold:
                api = _typed_api("mojo_layernorm_quant_nosmooth", dtype)
            else:
                api = _typed_api("mojo_layernorm_quant", dtype)
        except NotImplementedError:
            # Fall back if a specialised artifact is missing (older wheel).
            use_n2048 = False
            try:
                if smooth_scale is None or use_fold:
                    api = _typed_api("mojo_layernorm_quant_nosmooth", dtype)
                else:
                    api = _typed_api("mojo_layernorm_quant", dtype)
            except NotImplementedError:
                use_fold = False
                try:
                    api = _typed_api("mojo_layernorm_quant", dtype)
                except NotImplementedError:
                    raise NotImplementedError(
                        "UC backend cannot service this call (shape/dtype/contract not "
                        "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                        "(2026-06-08), this wrapper does not silently fall back to torch — "
                        "use TTX / torch_npu / torch_native backend for unsupported inputs."
                    )

        kernel_input = hidden_state if hidden_state.is_contiguous() else hidden_state.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        flat_x = kernel_input.reshape(rows, cols)

        if use_fold:
            weight, bias = self._get_folded_wb(
                self.weight, self.bias, smooth_scale, dtype
            )
        else:
            weight = self._cast_param(self.weight, "weight", dtype)
            bias = self._cast_param(self.bias, "bias", dtype)

        y_int8 = torch.empty((rows, cols), dtype=torch.int8, device=hidden_state.device)
        scale_per_token = torch.empty((rows,), dtype=torch.float32, device=hidden_state.device)
        eps = float(self.variance_epsilon)
        kernels = _uc_kernels()

        if smooth_scale is None or use_fold:
            kernels[api](
                flat_x,
                weight,
                bias,
                y_int8,
                scale_per_token,
                rows,
                cols,
                eps,
            )
        else:
            ss = smooth_scale.contiguous().reshape(cols)
            smooth_scale_kernel = ss if ss.dtype == torch.float32 else ss.to(torch.float32)
            kernels[api](
                flat_x,
                weight,
                bias,
                smooth_scale_kernel,
                y_int8,
                scale_per_token,
                rows,
                cols,
                eps,
            )

        output = y_int8.reshape(hidden_state.shape)
        scale_out = scale_per_token.reshape(hidden_state.shape[:-1] + (1,))
        return output, scale_out
