"""UC backend for ``MojoParallelEmbedding``.

``MojoParallelEmbedding`` (``mojo_opset/core/operators/embedding.py``) is a
vocabulary-parallel embedding: the embedding table is sharded along the
``num_embeddings`` (vocab) axis, every rank performs a local lookup that
zeros out-of-range indices, and an HCCL ``all_reduce(SUM)`` assembles the
final result across the TP group.

Strategy for the UC backend (P2-16 perf revision):

* **Dedicated UC kernel** ``mojo_parallel_embedding_h<HT>_<dtype>`` (block
  GATHER with dynamic ``(V, H, N)``, compile-time inner ``HT``). Two HT
  variants shipped per dtype -- ``HT=128`` for small / accuracy-test
  embeddings and ``HT=4096`` for LLaMA-style ``H=4096``; sweep showed
  HT=4096 cuts ``N=1024 H=4096`` from ~1125 us to ~30 us (gather-DMA
  count drops 32x). See
  ``uc-kernel/kernels/mojo_parallel_embedding.py``.
* **Single-rank fast-path** (``world_size == 1``): skip the parent's
  redundant ``input - 0`` / range-mask / clamp / ``output * 1`` work; the
  index path is already in ``[0, num_embeddings)`` and the
  ``all_reduce`` is a no-op, so the kernel call is the only device work.
* **TP / multi-rank** path: shift / clamp / mask on the host (the lifter
  v0.3 ``T.Parallel`` allowlist does not include comparisons, so the mask
  must stay on the host -- see lessons § A.1), then call the kernel, then
  multiply by the boolean range mask and ``all_reduce(SUM)``.
* **Hard-fallback guards**: ``max_norm`` (would mutate weight), unusual
  index dtype, dtype mismatch, H not divisible by any HT variant, etc.
  all delegate to ``MojoParallelEmbedding.forward`` so behaviour stays
  identical to the reference path.

Wheel ABI: ``mojo_parallel_embedding_h<HT>_<dtype>(weight, indices, out,
V, H, N)`` -- the trailing INT32 scalar order follows the
"first-occurrence in type annotations" rule (lessons § B.1): ``weight
(V, H)`` then ``indices (N,)`` then ``out (N, H)`` => ``(V, H, N)``.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mojo_opset.core import MojoParallelEmbedding

from ._utils import _DTYPE_API_SUFFIX
from ._utils import _uc_kernels


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)

# Inner H-tile sizes the kernel ships, biggest first; the wrapper picks
# the largest one that divides ``H``. Keep in lockstep with
# ``uc-kernel/kernels/mojo_parallel_embedding.py``.
_HT_VARIANTS = (4096, 128)

# Per-DMA setup overhead in the current UC block-GATHER lowering is
# ~1.7 us at HT=128 and ~110 us at HT=4096 (measured 2026-06-05). That
# makes UC competitive only for small ``ceil(N/48) * H`` budgets; for
# anything bigger ``aclnnEmbedding`` (torch_npu / parent path) is faster,
# so the wrapper transparently falls back. The threshold below comes from
# the perf model
#     UC_us ~= ceil(N / 48) * (H / 128) * 1.7
# matched against a torch_npu baseline of ~25-30 us at H<=4096 (see
# ``docs/project-ops/perf-debug/parallel-embedding-p2-16.md``).
_KERNEL_BUDGET_PRODUCT = 2200  # ceil(N/48) * H upper bound for UC fast-path


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _pick_kernel_api(embedding_dim: int, dtype: torch.dtype) -> Optional[str]:
    """Return the largest-HT registered API whose HT divides ``H``."""
    suffix = _DTYPE_API_SUFFIX.get(dtype)
    if suffix is None:
        return None
    try:
        kernels = _uc_kernels()
    except Exception:
        return None
    for ht in _HT_VARIANTS:
        if embedding_dim % ht != 0:
            continue
        api = f"mojo_parallel_embedding_h{ht}_{suffix}"
        if api in kernels.keys():
            return api
    return None


def _is_kernel_profitable(num_tokens: int, embedding_dim: int) -> bool:
    """Heuristic: only call the UC kernel when it should beat torch.

    The UC block-GATHER costs ~1.7 us per (HT=128) DMA; with 48 vector
    cores in parallel and ``H / 128`` gathers per token, the wall-clock
    grows as ``ceil(N / 48) * (H / 128) * 1.7``. Compared against the
    near-constant ~25-30 us of ``aclnnEmbedding`` the UC kernel wins only
    when the product ``ceil(N / 48) * H`` stays well under a few thousand.
    """
    if num_tokens <= 0:
        return False
    programs = 48
    return ((num_tokens + programs - 1) // programs) * embedding_dim <= _KERNEL_BUDGET_PRODUCT


class UCParallelEmbedding(MojoParallelEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------
        # Strict guards.  Per project rule "wheel 没实现的就直接给报错"
        # (2026-06-08), any condition the UC kernel cannot honour raises
        # instead of silently falling back to ``super().forward`` (torch).
        # ------------------------------------------------------------------
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"UCParallelEmbedding expects a torch.Tensor input, got {type(input)}.")
        if input.dtype not in _SUPPORTED_INDEX_DTYPES:
            raise NotImplementedError(
                f"UCParallelEmbedding supports int32/int64 indices, got {input.dtype}."
            )
        if self.weight.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                f"UCParallelEmbedding supports bf16/fp16 weights, got {self.weight.dtype}."
            )
        if self.weight.device != input.device:
            raise ValueError(
                f"UCParallelEmbedding requires weight and indices on the same device, "
                f"got weight={self.weight.device} input={input.device}."
            )
        if self.embedding_dim <= 0 or self.local_num_embeddings <= 0:
            raise ValueError(
                f"UCParallelEmbedding has invalid shape: embedding_dim={self.embedding_dim}, "
                f"local_num_embeddings={self.local_num_embeddings}."
            )
        if self.max_norm is not None:
            raise NotImplementedError(
                "UCParallelEmbedding does not implement max_norm (would mutate weight in-place); "
                "the UC kernel path cannot reproduce that semantics."
            )

        api = _pick_kernel_api(self.embedding_dim, self.weight.dtype)
        if api is None:
            raise NotImplementedError(
                f"UCParallelEmbedding has no matching kernel for embedding_dim={self.embedding_dim}, "
                f"dtype={self.weight.dtype}: built variants ship for H ∈ {_HT_VARIANTS}; "
                "see docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
            )

        # ------------------------------------------------------------------
        # Single-rank fast-path.  When the global vocab equals the local
        # shard (no TP) and ``torch.distributed`` is not initialised the
        # parent's shift / range-mask / clamp / multiply / all_reduce all
        # collapse to a no-op; running them costs 5 extra host->NPU
        # launches we can skip.
        # ------------------------------------------------------------------
        single_rank = (
            self.vocab_start_index == 0
            and self.local_num_embeddings == self.num_embeddings
            and not _is_dist_initialized()
        )

        if single_rank:
            return self._gather(api, input)

        # ------------------------------------------------------------------
        # TP / multi-rank path: do the shift / mask / clamp on the host
        # (lifter A.1 forbids comparisons inside the kernel), then call
        # the gather, then zero out-of-range rows and all_reduce.
        # ------------------------------------------------------------------
        local_input = input - self.vocab_start_index
        in_range = (local_input >= 0) & (local_input < self.local_num_embeddings)
        masked_input = local_input.clamp(0, self.local_num_embeddings - 1)

        output = self._gather(api, masked_input)

        # Zero contributions from out-of-range indices.
        output = output * in_range.unsqueeze(-1).to(output.dtype)

        if _is_dist_initialized():
            world_size = dist.get_world_size(group=self.process_group)
            if world_size > 1:
                dist.all_reduce(
                    output, op=dist.ReduceOp.SUM, group=self.process_group
                )
        return output

    # ----------------------------------------------------------------------
    # UC kernel call helper.  Raises on missing API / launch error per
    # "wheel 没实现的就直接给报错".
    # ----------------------------------------------------------------------
    def _gather(self, api: str, indices: torch.Tensor) -> torch.Tensor:
        weight = self.weight.contiguous()
        dtype = weight.dtype

        kernels = _uc_kernels()
        if api not in kernels:
            raise NotImplementedError(
                f"UC kernel {api!r} is not in the loaded uc-kernel wheel. "
                "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
            )
        kernel = kernels[api]

        indices_flat = indices.reshape(-1).to(torch.int32).contiguous()
        rows = indices_flat.numel()
        out_shape = tuple(indices.shape) + (self.embedding_dim,)
        if rows == 0:
            return torch.empty(out_shape, dtype=dtype, device=weight.device)

        flat_out = torch.empty(
            (rows, self.embedding_dim),
            dtype=dtype,
            device=weight.device,
        )
        # Trailing INT32 ABI: (V, H, N) -- "first-occurrence" rule.
        kernel(
            weight,
            indices_flat,
            flat_out,
            self.local_num_embeddings,
            self.embedding_dim,
            rows,
        )

        return flat_out.reshape(*indices.shape, self.embedding_dim)
