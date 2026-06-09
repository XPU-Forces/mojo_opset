"""DeepEP-style cross-rank MoE dispatch / combine operators.

These operators model the all-to-all token-routing protocol used by DeepEP:
the dispatch step routes (token, expert) pairs to their target rank, sorts
by local expert, and (optionally) fuses smooth + per-token int8 quantization;
the combine step reverses the routing and reduces by top-k gates.

The torch backend runs purely local at ``group_size == 1``; for
``group_size > 1`` it uses ``torch.distributed`` collectives to reconstruct
the global state and slice for the local-expert range, and therefore
requires an initialized process group. The xops backend provides a real
symmetric-memory implementation for production use.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from mojo_opset.core.operator import MojoOperator


def _local_dispatch_indices(
    top_k_indices: torch.Tensor,
    num_experts: int,
    top_k: int,
):
    """Compute local routing indices from per-rank top_k_indices.

    Returns:
        sort_perm: int64 [BS*top_k] argsort of flat top_k_indices, stable.
        scatter_index: int32 [BS, top_k] inverse permutation.
        expert_token_count: int32 [num_experts] local per-expert token count.
    """
    flat = top_k_indices.reshape(-1).to(torch.int64)
    sort_perm = flat.argsort(stable=True)
    scatter_index = sort_perm.argsort(stable=True).reshape(-1, top_k).to(torch.int32)
    expert_token_count = torch.bincount(flat, minlength=num_experts).to(torch.int32)
    return sort_perm, scatter_index, expert_token_count


class MojoDeepEPDispatch(MojoOperator):
    """All-to-all MoE token dispatch with optional fused per-token int8 quantization.

    Init params (kernel binds these on init — must match the paired combine):
    - num_experts (int): Total experts across all ranks. Must be divisible by group_size.
    - top_k (int): Top-k experts per token.
    - group_size (int): Number of ranks (a.k.a. ep_size). Defaults to 1.
    - rank (int): Local rank id in [0, group_size). Defaults to 0.
    - buffer_size (int): Symmetric-memory scratch in bytes.

    Forward returns a 6-tuple ``(expand_hidden_states, expert_token_cnt_per_rank,
    expert_token_cnt_cumsum, expand_scale, scatter_index, expert_token_count)``:
    - expand_hidden_states: [R, hidden] — int8 if fused per-token quant ran (smooth_scale provided),
      otherwise input dtype. R = total (token, expert) pairs landing on this rank,
      sorted by local expert id.
    - expert_token_cnt_per_rank: [local_experts] int32 — non-cumsum count per local expert. Sums to R.
    - expert_token_cnt_cumsum: [local_experts] int64 — end-offsets (cumsum without leading 0).
    - expand_scale: [R, 1] float32 — per-token quant scale (one entry per row of expand_hidden_states).
    - scatter_index: [q_len, top_k] int32 — local routing index, consumed by combine.
    - expert_token_count: [num_experts] int32 — local per-expert token count, consumed by combine.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        group_size: int = 1,
        rank: int = 0,
        buffer_size: int = 256 * 1024 * 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_experts % group_size != 0:
            raise ValueError(
                f"MojoDeepEPDispatch: num_experts must be divisible by group_size, "
                f"got num_experts={num_experts}, group_size={group_size}."
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size = group_size
        self.rank = rank
        self.buffer_size = buffer_size
        self.local_experts = num_experts // group_size
        self.start_expert_id = rank * self.local_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        top_k = top_k_indices.size(-1)
        if top_k != self.top_k:
            raise ValueError(f"top_k_indices last dim must be {self.top_k}, got {top_k}.")

        q_len = hidden_states.size(0)
        device = hidden_states.device
        top_k_indices = top_k_indices.reshape(-1, top_k).to(torch.int32)

        # Local routing — returned as-is for combine to consume.
        _, scatter_index, expert_token_count = _local_dispatch_indices(
            top_k_indices, self.num_experts, top_k
        )

        # Dispatch is defined on the global token tensor; for group_size>1 we
        # all_gather to reconstruct it, then slice for this rank's experts.
        if self.group_size == 1:
            global_hidden = hidden_states
            global_top_k = top_k_indices
        else:
            global_hidden = torch.empty(
                self.group_size * q_len,
                hidden_states.size(1),
                dtype=hidden_states.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(global_hidden, hidden_states.contiguous())
            global_top_k = torch.empty(
                self.group_size * q_len,
                top_k,
                dtype=top_k_indices.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(global_top_k, top_k_indices.contiguous())

        global_q = global_hidden.size(0)
        global_flat = global_top_k.reshape(-1).to(torch.int64)
        global_sort_perm = global_flat.argsort(stable=True)
        global_token_idx = (
            torch.arange(global_q, device=device, dtype=torch.int64)
            .unsqueeze(1)
            .expand(global_q, top_k)
            .reshape(-1)
        )
        pack_index = global_token_idx[global_sort_perm]
        expand_global = global_hidden[pack_index]
        sorted_experts_global = global_flat[global_sort_perm]

        global_expert_count = torch.bincount(global_flat, minlength=self.num_experts).to(torch.int32)
        cum = global_expert_count.to(torch.int64).cumsum(0)
        start_e = self.start_expert_id
        end_e = start_e + self.local_experts
        start_t = 0 if start_e == 0 else int(cum[start_e - 1].item())
        end_t = int(cum[end_e - 1].item())

        expand_local = expand_global[start_t:end_t]
        sorted_experts_local = sorted_experts_global[start_t:end_t]

        if smooth_scale is not None:
            # Fused smooth + per-token int8 quant.
            smoothed = expand_local.float() * smooth_scale[sorted_experts_local].float()
            expand_scale = smoothed.abs().amax(-1, keepdim=True) / 127.0
            x = smoothed / expand_scale
            expand_out = torch.clamp(
                torch.floor(x.abs() + 0.5) * x.sign(), -128, 127
            ).to(torch.int8)
        else:
            expand_out = expand_local
            expand_scale = torch.empty(
                (expand_local.size(0), 1), dtype=torch.float32, device=device
            )

        expert_token_cnt_per_rank = global_expert_count[start_e:end_e]
        expert_token_cnt_cumsum = expert_token_cnt_per_rank.to(torch.int64).cumsum(0)

        return (
            expand_out,
            expert_token_cnt_per_rank,
            expert_token_cnt_cumsum,
            expand_scale,
            scatter_index,
            expert_token_count,
        )


class MojoDeepEPCombine(MojoOperator):
    """All-to-all MoE expert-output combine — reverse of MojoDeepEPDispatch.

    Gathers per-local-expert outputs from all ranks, weights by top-k gates, and
    scatters back to the original [q_len, hidden] layout.

    Init params (must match the paired MojoDeepEPDispatch):
    - num_experts, top_k, group_size, rank, buffer_size.

    Forward args:
    - expert_outputs: [R, hidden] — local experts' outputs (sorted by local expert id).
    - top_k_gates: [q_len, top_k] — gating weights for the top-k reduction.
    - scatter_index: [q_len, top_k] int32 — from the paired dispatch.
    - expert_token_count: [num_experts] int32 — global per-expert count from the paired dispatch.
    - q_len (int): Original token count — sizes the output buffer.
    - output: optional pre-allocated [q_len, hidden] tensor.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        group_size: int = 1,
        rank: int = 0,
        buffer_size: int = 256 * 1024 * 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_experts % group_size != 0:
            raise ValueError(
                f"MojoDeepEPCombine: num_experts must be divisible by group_size, "
                f"got num_experts={num_experts}, group_size={group_size}."
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size = group_size
        self.rank = rank
        self.buffer_size = buffer_size
        self.local_experts = num_experts // group_size
        self.start_expert_id = rank * self.local_experts

    def forward(
        self,
        expert_outputs: torch.Tensor,
        top_k_gates: torch.Tensor,
        scatter_index: torch.Tensor,
        expert_token_count: torch.Tensor,
        q_len: int,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        top_k = top_k_gates.size(-1)
        if top_k != self.top_k:
            raise ValueError(f"top_k_gates last dim must be {self.top_k}, got {top_k}.")

        top_k_gates = top_k_gates.reshape(-1, top_k)
        scatter_index = scatter_index.reshape(-1, top_k).to(torch.int64)
        device = expert_outputs.device

        if self.group_size == 1:
            # All experts are local — scatter_index directly indexes expert_outputs.
            gathered = expert_outputs[scatter_index.reshape(-1)].reshape(q_len, top_k, -1)
        else:
            # Recover local top_k_indices from (scatter_index, expert_token_count):
            # sorted_local_experts[p] is the expert id at local sort position p,
            # so top_k_indices_local[t, k] = sorted_local_experts[scatter_index[t, k]].
            sorted_local_experts = torch.repeat_interleave(
                torch.arange(self.num_experts, device=device, dtype=torch.int64),
                expert_token_count.to(torch.int64),
            )
            top_k_indices_local = sorted_local_experts[scatter_index.reshape(-1)].reshape(
                q_len, top_k
            ).to(torch.int32)

            # all_gather top_k_indices across ranks to rebuild the global routing.
            global_top_k = torch.empty(
                self.group_size * q_len,
                top_k,
                dtype=top_k_indices_local.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(global_top_k, top_k_indices_local.contiguous())

            # Variable-size all_gather of expert_outputs: pad to max R, gather, trim.
            global_expert_count = expert_token_count.clone()
            dist.all_reduce(global_expert_count, op=dist.ReduceOp.SUM)
            cum_global = global_expert_count.to(torch.int64).cumsum(0)
            r_per_rank = []
            for r in range(self.group_size):
                s = r * self.local_experts
                e = s + self.local_experts
                start = 0 if s == 0 else int(cum_global[s - 1].item())
                end = int(cum_global[e - 1].item())
                r_per_rank.append(end - start)
            max_r = max(r_per_rank) if r_per_rank else 0

            padded = torch.zeros(
                max(max_r, 1),
                expert_outputs.size(1),
                dtype=expert_outputs.dtype,
                device=device,
            )
            if expert_outputs.size(0) > 0:
                padded[: expert_outputs.size(0)] = expert_outputs
            gathered_padded = [torch.zeros_like(padded) for _ in range(self.group_size)]
            dist.all_gather(gathered_padded, padded)
            global_expand = torch.cat(
                [gathered_padded[r][: r_per_rank[r]] for r in range(self.group_size)],
                dim=0,
            )

            # Global scatter (inverse of the global stable sort by expert id).
            global_flat = global_top_k.reshape(-1).to(torch.int64)
            global_sort_perm = global_flat.argsort(stable=True)
            global_scatter = global_sort_perm.argsort(stable=True)

            # Slice the global scatter for this rank's (token, k) pairs.
            offset = self.rank * q_len * top_k
            local_global_pos = global_scatter[offset : offset + q_len * top_k]
            gathered = global_expand[local_global_pos].reshape(q_len, top_k, -1)

        combined = (gathered.float() * top_k_gates.float().unsqueeze(-1)).sum(dim=1)
        result = combined.to(expert_outputs.dtype)

        if output is not None:
            output.copy_(result)
            return output
        return result
