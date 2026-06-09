from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from ...core.operator import MojoOperator
from ...core.operators.gemm import MojoQuantGemm

__all__ = ["MojoA2AQuantGemmDualHead"]


class MojoA2AQuantGemmDualHead(MojoOperator):
    """A2A-then-quantized-GEMM with dual-head (full + SWA) channel layout.

    Specialised variant of an A2A→GEMM fusion: the per-rank input channel is
    laid out as ``[full_local heads | swa_local heads]`` (two attention head
    groups concatenated), as opposed to the single-group layout a plain
    ``MojoA2AQuantGemm`` would assume. The GEMM weight is unsharded HF-order
    ``[full_global | swa_global]``, so the rank-axis reorder after A2A must
    split each rank's slice at the full/swa boundary and re-cat into HF order.

    Replaces the open-coded sequence between per-token int8 quantization of
    the attention output and the o_proj GEMM::

        send = attn_int8.view(tp, m_local, ch_local).contiguous()
        dist.all_to_all_single(recv, send, group)
        recv = recv.transpose(0, 1).contiguous().view(m_local, -1)
        recv = recv.view(m_local, tp, ch_local)
        recv = torch.cat([
            recv[..., :full_local_dim].reshape(m_local, -1),
            recv[..., full_local_dim:].reshape(m_local, -1),
        ], dim=-1).contiguous()
        # optional GQA-strided permute when yoco shards heads non-contiguously
        if perm is not None:
            full = recv[:, :full_total].view(m_local, full_global, D).index_select(1, perm_full)
            swa = recv[:, full_total:].view(m_local, swa_global, D).index_select(1, perm_swa)
            recv = torch.cat([full.flatten(1), swa.flatten(1)], dim=-1).contiguous()
        sp_scale = unified_scale.view(tp, -1)[tp_rank].contiguous()
        out = o_proj(recv, sp_scale)

    Why this op exists
    -------------------
    Each rank's attention output channel is ``[full_local | swa_local]``.
    After ``all_to_all_single`` the per-token concat along channels is
    ``[full_R0 | swa_R0 | full_R1 | swa_R1 | ...]``, while the unsharded
    GEMM weight expects HF order ``[full_R0..full_R(tp-1) | swa_R0..swa_R(tp-1)]``.
    The reorder + optional GQA-strided permute bridge the two layouts. Bundling
    them with the GEMM lets a future backend fuse the rearrange into the GEMM
    prologue.

    Construction
    ------------
    ``full_query_global_head_indices`` / ``swa_query_global_head_indices`` are
    the *local* head index tuples this rank owns (length
    ``full_nh_global / tp_size`` and ``swa_nh_global / tp_size`` respectively).
    The constructor performs a one-shot ``dist.all_gather_object`` on
    ``tp_group`` to learn the other ranks' indices, then computes argsort
    permutations. When the concat-by-rank order is already sorted (the
    contiguous even-share case), permutation is None and the index_select is
    skipped at runtime.
    """

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        full_nh_global: int,
        swa_nh_global: int,
        head_dim: int,
        hidden_size: int,
        full_query_global_head_indices: Tuple[int, ...],
        swa_query_global_head_indices: Tuple[int, ...],
        tp_group: Optional[dist.ProcessGroup] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        o_proj: Optional[MojoQuantGemm] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        if full_nh_global % tp_size != 0:
            raise ValueError(
                f"full_nh_global={full_nh_global} must be divisible by tp_size={tp_size}"
            )
        if swa_nh_global % tp_size != 0:
            raise ValueError(
                f"swa_nh_global={swa_nh_global} must be divisible by tp_size={tp_size}"
            )

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.full_nh_global = full_nh_global
        self.swa_nh_global = swa_nh_global
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.full_local = full_nh_global // tp_size
        self.swa_local = swa_nh_global // tp_size
        self.full_local_dim = self.full_local * head_dim
        self.swa_local_dim = self.swa_local * head_dim
        self.full_total = full_nh_global * head_dim
        self.ch_local = self.full_local_dim + self.swa_local_dim
        self.ch_global = self.full_total + swa_nh_global * head_dim

        if o_proj is not None:
            # Caller supplies a pre-built MojoQuantGemm so the parent module
            # owns the weight (single state_dict entry, no duplicate buffers).
            # Bypass nn.Module's auto-registration: setting ``self.o_proj``
            # directly would re-register the sub-module under this op's path
            # too, causing strict load_state_dict to see duplicate keys.
            # ``object.__setattr__`` keeps it as a plain attribute; .to(device)
            # is driven from the parent's registered path.
            object.__setattr__(self, "o_proj", o_proj)
        else:
            QuantGemmCls = MojoQuantGemm._registry.get(self._backend)
            self.o_proj = QuantGemmCls(
                in_features=self.ch_global,
                out_features=hidden_size,
                output_dtype=output_dtype,
                trans_weight=True,
                **kwargs,
            )

        # Compute argsort permutations for the post-A2A rank-cat order. See
        # docstring above. Stored as Python tuples so the constructor stays
        # safe under torch.device("meta") lazy-init contexts; lazily
        # materialised to torch tensors at first forward and cached per-device.
        self._build_permutations(
            tuple(full_query_global_head_indices),
            tuple(swa_query_global_head_indices),
        )

    def _build_permutations(
        self,
        full_local_indices: Tuple[int, ...],
        swa_local_indices: Tuple[int, ...],
    ) -> None:
        if self.tp_size == 1 or not (dist.is_available() and dist.is_initialized()):
            self._full_perm: Optional[Tuple[int, ...]] = None
            self._swa_perm: Optional[Tuple[int, ...]] = None
            self._full_perm_cache: Optional[torch.Tensor] = None
            self._swa_perm_cache: Optional[torch.Tensor] = None
            return

        full_per_rank: List[List[int]] = [None] * self.tp_size
        swa_per_rank: List[List[int]] = [None] * self.tp_size
        dist.all_gather_object(full_per_rank, list(full_local_indices), group=self.tp_group)
        dist.all_gather_object(swa_per_rank, list(swa_local_indices), group=self.tp_group)
        full_concat = [h for rk in range(self.tp_size) for h in full_per_rank[rk]]
        swa_concat = [h for rk in range(self.tp_size) for h in swa_per_rank[rk]]
        full_identity = full_concat == list(range(len(full_concat)))
        swa_identity = swa_concat == list(range(len(swa_concat)))
        if full_identity and swa_identity:
            self._full_perm = None
            self._swa_perm = None
        else:
            self._full_perm = tuple(
                sorted(range(len(full_concat)), key=lambda i: full_concat[i])
            )
            self._swa_perm = tuple(
                sorted(range(len(swa_concat)), key=lambda i: swa_concat[i])
            )
        self._full_perm_cache = None
        self._swa_perm_cache = None

    def forward(
        self,
        attn_int8: torch.Tensor,
        unified_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            attn_int8 (torch.Tensor): ``[n_pad, ch_local]`` int8 — gated attention
                output already smoothed and per-token quantized on this rank.
                ``n_pad`` must be divisible by ``tp_size``.
            unified_scale (torch.Tensor): ``[n_pad]`` or ``[n_pad, 1]`` float — per-token
                scale produced by the global (cross-rank) amax of the smoothed
                pre-quant tensor. The tp_rank's slice is used as o_proj input scale.

        Returns:
            torch.Tensor: ``[n_pad / tp_size, hidden_size]`` bf16 — local share of
                the o_proj output. The caller should AllGather this back to
                ``[n_pad, hidden_size]`` if a full output is required (kept outside
                this op so the caller can overlap or fuse the AllGather).
        """
        if attn_int8.dim() != 2:
            raise ValueError(f"attn_int8 must be 2D, got {tuple(attn_int8.shape)}")
        n_pad, ch = attn_int8.shape
        if ch != self.ch_local:
            raise ValueError(
                f"attn_int8 channel dim {ch} != expected ch_local {self.ch_local}"
            )
        if n_pad % self.tp_size != 0:
            raise ValueError(
                f"n_pad={n_pad} must be divisible by tp_size={self.tp_size}"
            )
        m_local = n_pad // self.tp_size

        if self.tp_size > 1 and dist.is_available() and dist.is_initialized():
            send = attn_int8.view(self.tp_size, m_local, self.ch_local).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.tp_group)
            # rank-axis swap so per-token slice is contiguous downstream
            recv = recv.transpose(0, 1).contiguous().view(m_local, -1)
            # split per-rank chunks at full/swa boundary, then rank-cat each branch
            recv = recv.view(m_local, self.tp_size, self.ch_local)
            recv = torch.cat(
                [
                    recv[:, :, : self.full_local_dim].reshape(m_local, -1),
                    recv[:, :, self.full_local_dim :].reshape(m_local, -1),
                ],
                dim=-1,
            ).contiguous()
            # restore HF head order when yoco shards with a GQA-strided pattern
            if self._full_perm is not None:
                if (
                    self._full_perm_cache is None
                    or self._full_perm_cache.device != recv.device
                ):
                    self._full_perm_cache = torch.tensor(
                        self._full_perm, dtype=torch.long, device=recv.device
                    )
                    self._swa_perm_cache = torch.tensor(
                        self._swa_perm, dtype=torch.long, device=recv.device
                    )
                full_part = recv[:, : self.full_total].view(
                    m_local, self.full_nh_global, self.head_dim
                )
                swa_part = recv[:, self.full_total :].view(
                    m_local, self.swa_nh_global, self.head_dim
                )
                full_part = full_part.index_select(1, self._full_perm_cache)
                swa_part = swa_part.index_select(1, self._swa_perm_cache)
                recv = torch.cat(
                    [full_part.flatten(1), swa_part.flatten(1)], dim=-1
                ).contiguous()
            sp_scale = unified_scale.view(self.tp_size, -1)[self.tp_rank].contiguous()
        else:
            recv = attn_int8
            sp_scale = unified_scale.view(-1)

        return self.o_proj(recv, sp_scale)

    def extra_repr(self) -> str:
        return (
            f"tp_size={self.tp_size}, tp_rank={self.tp_rank}, "
            f"full_nh_global={self.full_nh_global}, swa_nh_global={self.swa_nh_global}, "
            f"head_dim={self.head_dim}, hidden_size={self.hidden_size}, "
            f"perm={'identity' if self._full_perm is None else 'gqa-strided'}"
        )
