from typing import List, Optional
from typing import Tuple

import torch

from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoRotaryEmbedding
from mojo_opset.core.operators.kv_cache import (
    assert_paged_kv_layout_contract,
    build_paged_kv_chunk_metadata,
)
from mojo_opset.core.operators.position_embedding import (
    MojoNormRoPE,
    MojoNormRoPEStoreKV,
    MojoRoPEStoreKV,
)
from mojo_opset.experimental.operators.position_embedding import MojoGridRoPE
from mojo_opset.utils.logging import get_logger

from ._utils import _uc_kernels, run_kernel


logger = get_logger(__name__)


class UCRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu"]

    def __init__(
        self,
        rope_theta,
        rope_dim,
        attention_scaling: float = 1.0,
        init_max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(rope_theta, rope_dim, attention_scaling, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for UCRotaryEmbedding")

    def forward(
        self,
        x: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cu_q_lens is not None:
            assert cu_q_lens.dtype == torch.int32
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        assert position_ids is None or cu_q_lens is None, "At most one of cu_q_lens or position_ids should be provided"

        if cu_q_lens is not None:
            logger.warning_once(
                "UC rotary embedding does not support varlen mode yet; falling back to torch implementation."
            )
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        elif position_ids is not None:
            assert position_ids.shape == x.shape[:-1], "position_ids must have the same shape as x except the hidden dimension"
            position_ids = position_ids.contiguous()
        else:
            return self._arange_cache(x)

        return self._position_ids_cache(position_ids)

    def _arange_cache(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padded-prefill arange path: return ``self.cos[:seq_len], self.sin[:seq_len]``.

        TTX (``rot_pos_embed_impl``) and torch_npu both return a view on this
        path (zero device time in profiler) — the cached cos/sin are already
        laid out contiguously, so a row slice of the leading dim is itself
        contiguous and downstream consumers (``MojoApplyRoPE``) only read it.
        Doing an actual kernel copy here used to add ~45us / ~18us per call
        for B=32 S=8192 / B=4 S=2048; the slice path is "free".
        """
        if x.dim() < 2:
            raise AssertionError("x must have at least two dimensions for padded prefill rotary embedding")

        seq_len = x.shape[1]
        if seq_len > self.cos.shape[0]:
            raise ValueError(f"seq_len {seq_len} exceeds rotary cache length {self.cos.shape[0]}")

        # Contiguous slice on dim 0 of a contiguous source is itself
        # contiguous; no copy needed. Match TTX / torch_npu zero-overhead path.
        return self.cos[:seq_len], self.sin[:seq_len]

    def _position_ids_cache(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-row block-GATHER path: ``cos[pids], sin[pids]``.

        Falls back to the parent torch implementation (``cos[position_ids]``)
        when the wheel does not provide a kernel artifact for the cache dtype
        (e.g. wheel built without the rotary kernels). The kernel artifact is
        an fp32-only 1-row block GATHER (Y=128 column tile), so non-fp32 cache
        dtypes also fall back to torch.
        """
        rope_dim = self.cos.shape[-1]
        rows = position_ids.numel()
        output_shape = tuple(position_ids.shape) + (rope_dim,)

        if rows == 0 or rope_dim == 0:
            cos_out = torch.empty(output_shape, device=self.cos.device, dtype=self.cos.dtype)
            sin_out = torch.empty(output_shape, device=self.sin.device, dtype=self.sin.dtype)
            return cos_out, sin_out

        # Strict lookup: per "wheel 没实现的就直接给报错", missing kernel /
        # unsupported dtype raises instead of silently doing the torch
        # ``cos[position_ids]`` index_select path.
        kernels = _uc_kernels()
        api = "mojo_rotary_embedding_position_ids_fp32"
        if self.cos.dtype != torch.float32:
            raise NotImplementedError(
                f"UCRotaryEmbedding._position_ids_cache requires fp32 cos/sin cache, got {self.cos.dtype}. "
                "No UC kernel registered for non-fp32 cache dtype."
            )
        if api not in kernels:
            raise NotImplementedError(
                f"UC kernel {api!r} is not in the loaded uc-kernel wheel. "
                "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
            )

        cos_out = torch.empty(output_shape, device=self.cos.device, dtype=self.cos.dtype)
        sin_out = torch.empty(output_shape, device=self.sin.device, dtype=self.sin.dtype)

        # The kernel processes pairs of consecutive rows (row-pair pattern, ~1.21x
        # faster than per-row on B=32 S=8192). Pair iteration count is
        # ``(M + 1) // 2``: when M is odd the kernel would also touch
        # ``position_ids[M]`` / ``cos_out[M]`` past the buffers and crash with
        # "DDR address MTE out of range". We pad to even M with one duplicate
        # of the last pid (any in-range int32 works — output of the duplicate
        # row is discarded). The trim is a view, no copy.
        flat_pids = position_ids.reshape(-1).contiguous()
        even_rows = (rows + 1) & ~1  # round up to even
        if even_rows == rows:
            pids_arg = flat_pids
            cos_arg = cos_out.reshape(rows, rope_dim)
            sin_arg = sin_out.reshape(rows, rope_dim)
        else:
            # Allocate one extra row of scratch — the kernel will write into it
            # (using pids_arg[-1] = duplicate of flat_pids[-1]) and we discard.
            pids_arg = torch.empty(even_rows, dtype=torch.int32, device=flat_pids.device)
            pids_arg[:rows] = flat_pids
            pids_arg[rows] = flat_pids[rows - 1]
            cos_arg = torch.empty((even_rows, rope_dim), device=self.cos.device, dtype=self.cos.dtype)
            sin_arg = torch.empty((even_rows, rope_dim), device=self.sin.device, dtype=self.sin.dtype)

        kernels[api](
            self.cos.contiguous(),
            self.sin.contiguous(),
            pids_arg,
            cos_arg,
            sin_arg,
            self.cos.shape[0],
            rope_dim,
            even_rows,
        )

        if even_rows != rows:
            # Slice off the tail row that we appended to satisfy the even-M
            # requirement of the row-pair kernel — view, no copy.
            cos_out = cos_arg[:rows].reshape(output_shape)
            sin_out = sin_arg[:rows].reshape(output_shape)
        return cos_out, sin_out


class UCApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["npu"]
    _STATIC_APPLY_ROPE_KERNELS = frozenset(
        {
            (96, 96, torch.float16),
            (96, 32, torch.bfloat16),
            (88, 88, torch.bfloat16),
            (128, 48, torch.float16),
            (128, 128, torch.float16),
            (128, 128, torch.bfloat16),
        }
    )

    # bf16 D=128 R=128 kernels (Worker P-03) use head-pair tiles. QH must be a
    # multiple of 32 (XQ tile), KH must be a multiple of 8 (XK tile). Otherwise
    # fall back to torch (super().forward) since the static kernel cannot
    # express a partial tile (no T.if_then_else in T.Parallel).
    _D128_R128_BF16_Q_HEAD_TILE = 32
    _D128_R128_BF16_K_HEAD_TILE = 8

    @staticmethod
    def _normalize_to_token_head(
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]:
        if q.ndim == 3:
            if head_first:
                q = q.transpose(0, 1).contiguous()
                k = k.transpose(0, 1).contiguous()
            else:
                q = q.contiguous()
                k = k.contiguous()
            return q, k, None, None

        if head_first:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
        else:
            q = q.contiguous()
            k = k.contiguous()

        batch_size, seq_len = q.shape[0], q.shape[1]
        q = q.reshape(batch_size * seq_len, q.shape[2], q.shape[3]).contiguous()
        k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3]).contiguous()
        return q, k, batch_size, seq_len

    @staticmethod
    def _restore_from_token_head(
        x: torch.Tensor,
        original_shape: torch.Size,
        head_first: bool,
        batch_size: Optional[int],
        seq_len: Optional[int],
    ) -> torch.Tensor:
        if len(original_shape) == 3:
            if head_first:
                x = x.transpose(0, 1).contiguous()
            return x.reshape(original_shape)

        x = x.reshape(batch_size, seq_len, x.shape[1], x.shape[2])
        if head_first:
            x = x.transpose(1, 2).contiguous()
        return x.reshape(original_shape)

    @staticmethod
    def _run_static_token_head_kernel(
        api: str,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *shape_args: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_out = q.clone(memory_format=torch.contiguous_format)
        k_out = k.clone(memory_format=torch.contiguous_format)
        run_kernel(api, q.dtype, q, k, cos, sin, q_out, k_out, *shape_args)
        return q_out, k_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q.ndim == k.ndim, "q and k must have the same dimension"
        assert q.ndim == 3 or q.ndim == 4, "q and k must be 3D or 4D"
        assert cos.shape == sin.shape, "cos and sin must have the same shape"
        if q.ndim == 3:
            assert cos.ndim == 2, "3D q/k inputs expect 2D cos/sin"
        elif cos.ndim not in (2, 3):
            raise ValueError("4D q/k inputs expect 2D or 3D cos/sin")

        if q.dtype != k.dtype:
            raise ValueError(f"q and k must have the same dtype, got {q.dtype} and {k.dtype}.")
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise NotImplementedError("UC backend MojoApplyRoPE expects float32 cos/sin tensors.")

        if q.numel() == 0 or k.numel() == 0:
            return torch.empty_like(q), torch.empty_like(k)

        rope_dim = cos.shape[-1]
        q_norm, k_norm, batch_size, seq_len = self._normalize_to_token_head(q, k, head_first)
        rows, q_heads, head_dim = q_norm.shape
        k_rows, k_heads, k_head_dim = k_norm.shape
        if rows != k_rows or head_dim != k_head_dim:
            raise ValueError("q and k must have matching token count and head dimension")

        config_key = (head_dim, rope_dim, q_norm.dtype)
        if config_key not in self._STATIC_APPLY_ROPE_KERNELS:
            raise NotImplementedError(
                "UC backend MojoApplyRoPE does not provide an aligned static kernel for "
                f"head_dim={head_dim}, rope_dim={rope_dim}, dtype={q_norm.dtype}."
            )

        # bf16 D=128 R=128 fast-path uses head-pair tiles (XQ=32, XK=8); requires
        # QH%32==0 and KH%8==0. Otherwise fall back to torch native.
        if config_key == (128, 128, torch.bfloat16):
            if q_heads % self._D128_R128_BF16_Q_HEAD_TILE != 0 or \
               k_heads % self._D128_R128_BF16_K_HEAD_TILE != 0:
                raise NotImplementedError(
                    "UC backend cannot service this call (shape/dtype/contract not "
                    "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                    "(2026-06-08), this wrapper does not silently fall back to torch — "
                    "use TTX / torch_npu / torch_native backend for unsupported inputs."
                )

        if cos.ndim == 2:
            cos_kind = "cos2d"
            cos_kernel = cos.contiguous()
            sin_kernel = sin.contiguous()
            shape_args = (rows, q_heads, k_heads, cos.shape[0])
        else:
            cos_kind = "costoken"
            cos_kernel = cos.reshape(rows, rope_dim).contiguous()
            sin_kernel = sin.reshape(rows, rope_dim).contiguous()
            shape_args = (rows, q_heads, k_heads)

        api = f"mojo_apply_rope_tnh_d{head_dim}_r{rope_dim}_{cos_kind}"
        q_out, k_out = self._run_static_token_head_kernel(
            api, q_norm, k_norm, cos_kernel, sin_kernel, *shape_args
        )
        return (
            self._restore_from_token_head(q_out, q.shape, head_first, batch_size, seq_len),
            self._restore_from_token_head(k_out, k.shape, head_first, batch_size, seq_len),
        )


# ============================================================================
# 4 RoPE-fusion UC backends (Worker C-27, task b7e713cf).
#
# Kernels (all bf16-only, shape-specialized to D=128 / rope_dim=128 /
# costoken-mode cos/sin):
#   * mojo_rope_store_kv_d128_r128_costoken_bf16
#       fused Q+K RoPE + paged KV SCATTER
#   * mojo_norm_rope_d128_r128_costoken_bf16
#       fused per-head RMSNorm + RoPE on Q and K (X=8 head-pair tile)
#   * mojo_norm_rope_store_kv_d128_r128_costoken_bf16
#       full fusion (norm + rope + scatter)
#   * mojo_grid_rope_d128_bf16
#       complex-pair rotation (Wan-style 3D grid RoPE)
#
# All 4 dry-run UIR-PASS (no on-board accuracy / perf done -- per task
# hard constraint "不 ssh / 不编 / 不 commit"). Each wrapper falls back
# to ``super().forward`` whenever the fast-path preconditions are not
# met (dtype/shape/head-count constraints + wheel kernel availability).
# ============================================================================


_RSK_HEAD_DIM = 128
_RSK_ROPE_DIM = 128
_ROPE_STORE_KV_KERNEL = "mojo_rope_store_kv_d128_r128_costoken_bf16"

_NORM_TILE_XQ = 32  # Q head-pair tile in mojo_norm_rope kernel; QH must be multiple.
_NORM_TILE_XK = 8   # K head-pair tile; KH must be multiple. (32B c0 min on (X,) f32)
_NORM_HEAD_DIM = 128
_NORM_HEAD_HALF = 64  # rope rotation half-dim; halfcos ABI (P3-06-B §3.1a)
_NORM_ROPE_DIM = 128
_NORM_ROPE_KERNEL = "mojo_norm_rope_d128_r128_costoken_bf16"
_NORM_ROPE_HALFCOS_KERNEL = "mojo_norm_rope_d128_r128_halfcos_costoken_bf16"
_NORM_ROPE_HALFCOS_HP1_KERNEL = "mojo_norm_rope_d128_r128_halfcos_hp1_costoken_bf16"
_NORM_ROPE_STORE_KV_KERNEL = "mojo_norm_rope_store_kv_d128_r128_costoken_bf16"

_GRID_HEAD_DIM = 128
_GRID_HEAD_HALF = 64
_GRID_ROPE_KERNEL = "mojo_grid_rope_d128_bf16"


def _build_paged_flat_slot_indices(
    block_table: torch.Tensor,
    cu_q_lens: Optional[torch.Tensor],
    context_kv_lens: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> Optional[torch.Tensor]:
    """Compute per-token flat ``slot = phys_block * block_size + block_off`` indices.

    Mirrors the helper used by ``UCStorePagedMLAKVCache`` but for the
    standard (kv_heads first inside the cache row) layout. Returns
    ``None`` if any token would map to an invalid (-1) slot or overflow
    the block_table column count -- signals the caller to fall back to
    the parent torch path.
    """
    device = block_table.device
    batch_size = context_kv_lens.shape[0]
    is_decode = cu_q_lens is None

    context_i32 = context_kv_lens.to(torch.int32)
    if is_decode:
        q_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
        cu = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(q_lens, dim=0, dtype=torch.int32),
            ]
        )
        total_tokens = int(batch_size)
    else:
        cu = cu_q_lens.to(torch.int32)
        q_lens = cu[1:] - cu[:-1]
        total_tokens = int(cu[-1].item())

    if total_tokens == 0:
        return torch.empty(0, dtype=torch.int32, device=device)

    valid_batch = (context_i32 >= 0) & (q_lens > 0)
    if not bool(valid_batch.all().item()):
        return None

    token_idx = torch.arange(total_tokens, dtype=torch.int32, device=device)
    token_batch_id = (
        torch.searchsorted(cu[1:].contiguous(), token_idx, right=True)
        .clamp_max(batch_size - 1)
        .to(torch.int32)
    )
    token_off_in_seq = token_idx - cu[token_batch_id]

    write_pos = context_i32[token_batch_id] + token_off_in_seq
    logical_block = torch.div(write_pos, block_size, rounding_mode="floor")
    block_offset = write_pos - logical_block * block_size

    max_blocks_per_seq = block_table.shape[1]
    if int(logical_block.max().item()) >= max_blocks_per_seq:
        return None

    physical_block = block_table[token_batch_id.long(), logical_block.long()]
    if bool((physical_block < 0).any().item()):
        return None

    flat_slot = physical_block.to(torch.int32) * block_size + block_offset
    if int(flat_slot.max().item()) >= num_blocks * block_size:
        return None
    return flat_slot


def _rope_store_kv_fast_path_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Cheap dtype/shape guards shared by RoPEStoreKV / NormRoPEStoreKV."""
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        return False
    if key_cache.dtype is not torch.bfloat16 or value_cache.dtype is not torch.bfloat16:
        return False
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        return False
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return False
    return True


class UCRoPEStoreKV(MojoRoPEStoreKV):
    """UC fast-path: fused Q/K RoPE + paged KV SCATTER (single kernel)."""

    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        context_kv_lens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        *,
        chunk_metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- 0. Cheap dtype/shape guards ------------------------------
        if head_first:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not _rope_store_kv_fast_path_layout(q, k, v, cos, sin, key_cache, value_cache):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Hard-coded fast-path shape: head_dim == rope_dim == 128, full rope.
        if q.shape[-1] != _RSK_HEAD_DIM or cos.shape[-1] != _RSK_ROPE_DIM:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        rows, q_heads, head_dim = q.shape
        k_rows, k_heads, k_head_dim = k.shape
        v_rows, v_k_heads, v_head_dim = v.shape
        if rows != k_rows or rows != v_rows or head_dim != k_head_dim or head_dim != v_head_dim:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if k_heads != v_k_heads:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cos.shape[0] != rows:
            # costoken requires per-token cos/sin (no broadcast).
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Cache layout (num_blocks, kv_heads, block_size, head_dim).
        num_blocks, cache_kv_heads, block_size, cache_head_dim = key_cache.shape
        if (
            cache_kv_heads != k_heads
            or cache_head_dim != head_dim
            or value_cache.shape != key_cache.shape
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ---- 1. Build per-token flat slot indices ---------------------
        if chunk_metadata is not None:
            # We use the per-token flat-slot index format; chunk_metadata
            # is a different layout -> bail to parent path.
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if block_table is None or context_kv_lens is None:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        try:
            assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
        except AssertionError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        flat_slot = _build_paged_flat_slot_indices(
            block_table, cu_q_lens, context_kv_lens, block_size, num_blocks,
        )
        if flat_slot is None or flat_slot.shape[0] != rows:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ---- 2. Wheel availability ------------------------------------
        kernels = _uc_kernels()
        if _ROPE_STORE_KV_KERNEL not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ---- 3. Launch the kernel -------------------------------------
        # Cache view: (num_blocks, kv_heads, block_size, head_dim)
        # -> flat (num_blocks * block_size, kv_heads, head_dim).
        # The cache is layout-contiguous so the view is no-copy iff
        # ``kv_heads * block_size * head_dim == stride[0]``; require
        # contiguous as a cheap proxy.
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        cache_rows = num_blocks * block_size
        # Note: the kernel expects (cache_rows, kv_heads, head_dim) but
        # the natural cache view from (num_blocks, kv_heads, block_size,
        # head_dim) flattens to (num_blocks * block_size, kv_heads,
        # head_dim) only if (block_size, kv_heads) are swapped first
        # because the original layout puts kv_heads BEFORE block_size in
        # memory order. Permute then flatten gives a non-contiguous view
        # that the kernel cannot accept without a copy; we materialise a
        # contiguous flat cache only on the wrapper level (one transpose
        # + clone) and copy back -- so this fast path is only profitable
        # if the upstream caller already stores the cache as
        # (num_blocks * block_size, kv_heads, head_dim). Detect that
        # special layout and fall back otherwise.
        try:
            key_cache_flat = key_cache.permute(0, 2, 1, 3).reshape(cache_rows, k_heads, head_dim)
            value_cache_flat = value_cache.permute(0, 2, 1, 3).reshape(
                cache_rows, k_heads, head_dim
            )
        except RuntimeError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not key_cache_flat.is_contiguous() or not value_cache_flat.is_contiguous():
            # Non-contiguous flat view -> fall back to parent (the
            # parent's torch loop handles arbitrary cache layouts).
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        q_contig = q.contiguous()
        k_contig = k.contiguous()
        v_contig = v.contiguous()
        cos_contig = cos.contiguous()
        sin_contig = sin.contiguous()
        slot_idx = flat_slot.contiguous().to(torch.int32)
        q_out = torch.empty_like(q_contig)
        k_out = torch.empty_like(k_contig)

        kernels[_ROPE_STORE_KV_KERNEL](
            q_contig, k_contig, v_contig, cos_contig, sin_contig, slot_idx,
            q_out, k_out, key_cache_flat, value_cache_flat,
            rows, q_heads, k_heads, cache_rows,
        )
        return q_out, k_out


def _norm_rope_fast_path_layout(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    head_dim: int,
) -> bool:
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        return False
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        return False
    if q.ndim != 3 or k.ndim != 3:
        return False
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        return False
    if q.shape[-1] != head_dim or k.shape[-1] != head_dim:
        return False
    # P3-06-B §3.1a: also accept halfcos pre-sliced inputs (cos.shape[-1] == head_dim // 2).
    if cos.shape[-1] != head_dim and cos.shape[-1] != head_dim // 2:
        return False
    if q.shape[0] != k.shape[0] or q.shape[0] != cos.shape[0]:
        return False
    return True


class UCNormRoPE(MojoNormRoPE):
    """UC fast-path: per-head RMSNorm + RoPE for Q and K (no KV store)."""

    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if head_first:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not (self.use_query_norm and self.use_key_norm):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if self.head_dim != _NORM_HEAD_DIM:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not _norm_rope_fast_path_layout(q, k, cos, sin, _NORM_HEAD_DIM):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        rows, q_heads, _ = q.shape
        _, k_heads, _ = k.shape
        if q_heads % _NORM_TILE_XQ != 0 or k_heads % _NORM_TILE_XK != 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Weight dtype must match input bf16; weights live in bf16 storage
        # for this fast path.
        if self.q_weight.dtype is not torch.bfloat16 or self.k_weight.dtype is not torch.bfloat16:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        kernels = _uc_kernels()
        # P3-06-B §3.1a/b dispatch policy (perf-validated):
        #   * halfcos variant       -- 50% smaller cos/sin DMA. Dispatched ONLY
        #     when the caller already supplies pre-sliced (M, HALF) cos/sin;
        #     in that case there is no host-side copy overhead and the kernel
        #     wins ~5-10% vs the full-cos kernel.
        #   * full-cos legacy kernel -- (M, D) cos ABI. Dispatched when the
        #     caller supplies (M, D) cos/sin. We intentionally do NOT
        #     auto-slice on the host: ``cos[..., :HALF].contiguous()`` is a
        #     full 16 MB tensor copy at production shape (M=262144) which
        #     overshoots the ~5% kernel DMA saving (perf-debug-B § 4).
        #     Callers that want the halfcos win must pre-slice once at
        #     model-init and reuse across many forward passes.
        #   * parent torch path     -- ultimate fallback.
        #
        # NOTE: a `halfcos_hp1` variant exists in the wheel manifest that
        # statically unrolls the trip-1 ``hp`` loop and shares the per-row
        # cos/sin reload between Q and K segments. The lifter accepts the
        # variant (UIR + tvmscript dump OK) and the package links cleanly,
        # but the generated kernel **hangs at runtime on NPU** -- a smoke
        # call did not return after >10 s and had to be killed. The
        # dispatch is therefore disabled here; the hang is tracked as
        # P3-06-B §3.1b cannot-optimize (unblock_axis=cce_codegen, evidence
        # at /tmp/p3-06B_audit.log on uc-910b-docker).
        has_halfcos = _NORM_ROPE_HALFCOS_KERNEL in kernels.keys()
        has_full = _NORM_ROPE_KERNEL in kernels.keys()
        if not (has_halfcos or has_full):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        q_contig = q.contiguous()
        k_contig = k.contiguous()
        q_out = torch.empty_like(q_contig)
        k_out = torch.empty_like(k_contig)

        if has_halfcos and cos.shape[-1] == _NORM_HEAD_HALF:
            # Caller pre-sliced (M, HALF) cos/sin -- pure win, no host copy.
            cos_h = cos.contiguous()
            sin_h = sin.contiguous()
            kernels[_NORM_ROPE_HALFCOS_KERNEL](
                q_contig, k_contig,
                self.q_weight.contiguous(), self.k_weight.contiguous(),
                cos_h, sin_h,
                q_out, k_out,
                rows, q_heads, k_heads,
                float(self.variance_epsilon),
            )
            return q_out, k_out

        # Full-cos path (cos.shape[-1] == _NORM_HEAD_DIM).
        if not has_full:
            # Halfcos exists but caller passed full cos; the auto-slice
            # would regress perf, so prefer the parent torch path over
            # silently slowing down.
            if cos.shape[-1] == _NORM_HEAD_DIM and has_halfcos:
                # Last-resort halfcos with explicit auto-slice (measurable
                # regression at large M, but still correct and likely faster
                # than torch native on NPU).
                cos_h = cos[..., :_NORM_HEAD_HALF].contiguous()
                sin_h = sin[..., :_NORM_HEAD_HALF].contiguous()
                kernels[_NORM_ROPE_HALFCOS_KERNEL](
                    q_contig, k_contig,
                    self.q_weight.contiguous(), self.k_weight.contiguous(),
                    cos_h, sin_h,
                    q_out, k_out,
                    rows, q_heads, k_heads,
                    float(self.variance_epsilon),
                )
                return q_out, k_out
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cos.shape[-1] != _NORM_HEAD_DIM:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        cos_contig = cos.contiguous()
        sin_contig = sin.contiguous()
        kernels[_NORM_ROPE_KERNEL](
            q_contig, k_contig,
            self.q_weight.contiguous(), self.k_weight.contiguous(),
            cos_contig, sin_contig,
            q_out, k_out,
            rows, q_heads, k_heads,
            float(self.variance_epsilon),
        )
        return q_out, k_out


class UCNormRoPEStoreKV(MojoNormRoPEStoreKV):
    """UC fast-path: full fusion of per-head RMSNorm + RoPE + paged KV SCATTER."""

    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        context_kv_lens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        *,
        chunk_metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if head_first:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not (self.use_query_norm and self.use_key_norm):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if self.head_dim != _NORM_HEAD_DIM:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not _rope_store_kv_fast_path_layout(q, k, v, cos, sin, key_cache, value_cache):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cos.shape[-1] != _NORM_ROPE_DIM or q.shape[-1] != _NORM_HEAD_DIM:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        rows, q_heads, head_dim = q.shape
        _, k_heads, _ = k.shape
        if q_heads % _NORM_TILE_XQ != 0 or k_heads % _NORM_TILE_XK != 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cos.shape[0] != rows:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if self.q_weight.dtype is not torch.bfloat16 or self.k_weight.dtype is not torch.bfloat16:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        num_blocks, cache_kv_heads, block_size, cache_head_dim = key_cache.shape
        if (
            cache_kv_heads != k_heads
            or cache_head_dim != head_dim
            or value_cache.shape != key_cache.shape
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        if chunk_metadata is not None or block_table is None or context_kv_lens is None:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        try:
            assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
        except AssertionError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        flat_slot = _build_paged_flat_slot_indices(
            block_table, cu_q_lens, context_kv_lens, block_size, num_blocks,
        )
        if flat_slot is None or flat_slot.shape[0] != rows:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        kernels = _uc_kernels()
        if _NORM_ROPE_STORE_KV_KERNEL not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        cache_rows = num_blocks * block_size
        try:
            key_cache_flat = key_cache.permute(0, 2, 1, 3).reshape(cache_rows, k_heads, head_dim)
            value_cache_flat = value_cache.permute(0, 2, 1, 3).reshape(
                cache_rows, k_heads, head_dim
            )
        except RuntimeError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not key_cache_flat.is_contiguous() or not value_cache_flat.is_contiguous():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        q_contig = q.contiguous()
        k_contig = k.contiguous()
        v_contig = v.contiguous()
        cos_contig = cos.contiguous()
        sin_contig = sin.contiguous()
        slot_idx = flat_slot.contiguous().to(torch.int32)
        q_out = torch.empty_like(q_contig)
        k_out = torch.empty_like(k_contig)

        kernels[_NORM_ROPE_STORE_KV_KERNEL](
            q_contig, k_contig, v_contig,
            self.q_weight.contiguous(), self.k_weight.contiguous(),
            cos_contig, sin_contig, slot_idx,
            q_out, k_out, key_cache_flat, value_cache_flat,
            rows, q_heads, k_heads, cache_rows,
            float(self.variance_epsilon),
        )
        return q_out, k_out


class UCGridRoPE(MojoGridRoPE):
    """DISABLED — ``mojo_grid_rope_d128_bf16`` is not in the current uc-kernel wheel.

    Reason: CCE ``ELEMENT_PER_BLOCK`` static assert (kernel side, Class C,
    see ``docs/project-ops/uc-kernel-fail-todo-2026-06-08.md``).

    Per project rule "wheel 没实现的就直接给报错" (2026-06-08), this wrapper
    raises ``NotImplementedError`` instead of silently falling back to torch
    — use TTX / torch_npu / torch_native backend for this op.
    """

    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UCGridRoPE is disabled: mojo_grid_rope_d128_bf16 is not in the "
            "current uc-kernel wheel (CCE ELEMENT_PER_BLOCK static assert, "
            "Class C; see docs/project-ops/uc-kernel-fail-todo-2026-06-08.md). "
            "Per project rule '没实现的就直接报错', this wrapper does not "
            "silently fall back to torch. Use TTX / torch_npu / torch_native "
            "backend instead."
        )
