from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
import torch.distributed._functional_collectives as fc

from ..operator import MojoOperator


def _gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    trans_weight: bool,
) -> torch.Tensor:
    if trans_weight:
        output = input @ weight
        if bias is not None:
            output = output + bias
    else:
        output = F.linear(input, weight, bias)
    return output


def _quant_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    trans_weight: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    input_fp = input.float()
    weight_fp = weight.float()
    if trans_weight:
        output = input_fp @ weight_fp
    else:
        output = input_fp @ weight_fp.transpose(-2, -1)

    scale = weight_scale.float()
    while scale.dim() < output.dim():
        scale = scale.unsqueeze(0)

    token_scale = per_token_scale.float()
    while token_scale.dim() < output.dim():
        token_scale = token_scale.unsqueeze(-1)

    return (output * scale * token_scale).to(output_dtype)


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


class MojoGemmAllReduce(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Row-parallel fused GEMM + AllReduce.

        In tensor parallelism each rank holds a column-shard of the input
        features and the corresponding row-shard of the weight.  Each rank
        computes a partial GEMM, then AllReduce (sum) produces the full result.

        Semantics::

            output = allreduce(input @ weight [+ bias])

        When ``torch.distributed`` is not initialised, AllReduce is an identity
        and the operator behaves as a standard GEMM projection.

        Args:
            weight (torch.Tensor): Weight matrix.
                ``trans_weight=False`` → shape ``(out_features, in_features_local)``;
                ``trans_weight=True``  → shape ``(in_features_local, out_features)``.
            bias (Optional[torch.Tensor]): Shape ``(out_features,)``.
            trans_weight (bool): Whether weight layout is transposed.
            process_group (Optional[ProcessGroup]): Distributed group for
                AllReduce.  ``None`` means the default group.
        """
        super().__init__()
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.weight = weight
        self.bias = bias
        self.trans_weight = trans_weight
        self.process_group = process_group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute GEMM then AllReduce (sum) across the process group.

        Args:
            input (torch.Tensor): ``(*, in_features_local)`` — each rank's
                column-shard of the activation.

        Returns:
            torch.Tensor: ``(*, out_features)`` — the fully-reduced result.
        """
        output = _gemm(input, self.weight, self.bias, self.trans_weight)
        if _is_dist_initialized():
            process_group = self.process_group or _get_default_group()
            output = fc.all_reduce(output, reduceOp="sum", group=process_group)
        return output

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        has_bias = self.bias is not None
        return f"{weight_shape=}, {has_bias=}, {self.trans_weight=}".replace("self.", "")


class MojoAllGatherGemm(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        gather_dim: int = 0,
    ):
        """
        Sequence-parallel fused AllGather + GEMM.

        Each rank holds a sequence shard of the activation.  AllGather
        reconstructs the full sequence across ranks, then the GEMM is computed.
        Typical use: QKV projection or first FFN GEMM in an SP layer.

        Semantics::

            gathered = allgather(input, dim=gather_dim)   # (S, ...) → (S*tp, ...)
            output   = gathered @ weight [+ bias]

        When ``torch.distributed`` is not initialised, AllGather is an identity.

        Args:
            weight (torch.Tensor): Weight matrix. Layout follows ``trans_weight``.
            bias (Optional[torch.Tensor]): Shape ``(out_features,)``.
            trans_weight (bool): Whether weight layout is transposed.
            process_group (Optional[ProcessGroup]): Distributed group for
                AllGather.  ``None`` means the default group.
            gather_dim (int): Dimension along which to AllGather the input.
                Defaults to 0 (sequence / token dimension).
        """
        super().__init__()
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.weight = weight
        self.bias = bias
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.gather_dim = gather_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        AllGather input then compute GEMM.

        Args:
            input (torch.Tensor): ``(*, in_features)`` — local shard along
                ``gather_dim``.

        Returns:
            torch.Tensor: ``(*, out_features)`` where the ``gather_dim``
                extent is ``world_size × local_extent`` (single-rank: unchanged).
        """
        if _is_dist_initialized():
            process_group = self.process_group or _get_default_group()
            input = fc.all_gather_tensor(input, gather_dim=self.gather_dim, group=process_group)
        output = _gemm(input, self.weight, self.bias, self.trans_weight)
        return output

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        has_bias = self.bias is not None
        return (
            f"{weight_shape=}, {has_bias=}, {self.trans_weight=}, "
            f"gather_dim={self.gather_dim}"
        ).replace("self.", "")


class MojoGemmAll2All(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        scatter_dim: int = 0,
        gather_dim: int = 1,
    ):
        """
        Ulysses-style fused GEMM + All2All.

        After the matrix multiplication the output is redistributed via
        All2All: split along ``scatter_dim`` across ranks, then concatenate
        along ``gather_dim``.  This switches the sharding axis, e.g. from
        sequence-parallel to head-parallel (or vice-versa).

        Semantics::

            gemm_out = input @ weight [+ bias]
            output   = all_to_all(gemm_out,
                                  scatter_dim=scatter_dim,
                                  gather_dim=gather_dim)

        When ``torch.distributed`` is not initialised, All2All is an identity.

        Args:
            weight (torch.Tensor): Weight matrix. Layout follows ``trans_weight``.
            bias (Optional[torch.Tensor]): Shape ``(out_features,)``.
            trans_weight (bool): Whether weight layout is transposed.
            process_group (Optional[ProcessGroup]): Distributed group.
            scatter_dim (int): Dimension to split and scatter. Default 0.
            gather_dim (int): Dimension to gather and concatenate. Default 1.
        """
        super().__init__()
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.weight = weight
        self.bias = bias
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.scatter_dim = scatter_dim
        self.gather_dim = gather_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute GEMM then All2All.

        Args:
            input (torch.Tensor): ``(*, in_features)``.

        Returns:
            torch.Tensor: ``(*, out_features)`` with the sharding axis
                switched from ``scatter_dim`` to ``gather_dim``.
        """
        output = _gemm(input, self.weight, self.bias, self.trans_weight)
        if _is_dist_initialized():
            process_group = self.process_group or _get_default_group()
            world_size = dist.get_world_size(group=process_group)
            send_chunks = list(output.chunk(world_size, dim=self.scatter_dim))
            recv_chunks: List[torch.Tensor] = [
                torch.empty_like(c) for c in send_chunks
            ]
            dist.all_to_all(recv_chunks, send_chunks, group=process_group)
            output = torch.cat(recv_chunks, dim=self.gather_dim)
        return output

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        has_bias = self.bias is not None
        return (
            f"{weight_shape=}, {has_bias=}, {self.trans_weight=}, "
            f"scatter_dim={self.scatter_dim}, gather_dim={self.gather_dim}"
        ).replace("self.", "")


class MojoGemmReduceScatter(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        scatter_dim: int = 0,
    ):
        """
        Sequence-parallel fused GEMM + ReduceScatter.

        Each rank computes a full GEMM, then ReduceScatter sums partial
        results across TP ranks and scatters the sum so each rank holds its
        local sequence shard.  Typical use: FFN down-projection or attention
        output projection in an SP layer.

        Semantics::

            gemm_out = input @ weight [+ bias]
            output   = reduce_scatter(gemm_out, dim=scatter_dim)
                     # shape along scatter_dim shrinks by factor of world_size

        When ``torch.distributed`` is not initialised, ReduceScatter is an
        identity.

        Args:
            weight (torch.Tensor): Weight matrix. Layout follows ``trans_weight``.
            bias (Optional[torch.Tensor]): Shape ``(out_features,)``.
            trans_weight (bool): Whether weight layout is transposed.
            process_group (Optional[ProcessGroup]): Distributed group.
            scatter_dim (int): Dimension along which to scatter. Default 0.
        """
        super().__init__()
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.weight = weight
        self.bias = bias
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.scatter_dim = scatter_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute GEMM then ReduceScatter.

        Args:
            input (torch.Tensor): ``(*, in_features)`` — each rank's
                column-shard of the activation.

        Returns:
            torch.Tensor: The local scatter shard after reduce-sum.
                Shape along ``scatter_dim`` is ``original / world_size``
                (single-rank: unchanged).
        """
        output = _gemm(input, self.weight, self.bias, self.trans_weight)
        if _is_dist_initialized():

            process_group = self.process_group or _get_default_group()
            world_size = dist.get_world_size(group=process_group)
            rank = dist.get_rank(group=process_group)
            chunks = list(output.chunk(world_size, dim=self.scatter_dim))

            reduced = torch.empty_like(chunks[rank])
            dist.reduce_scatter(
                reduced, chunks, op=dist.ReduceOp.SUM, group=process_group
            )
            output = reduced
        return output

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        has_bias = self.bias is not None
        return (
            f"{weight_shape=}, {has_bias=}, {self.trans_weight=}, "
            f"scatter_dim={self.scatter_dim}"
        ).replace("self.", "")


class MojoQuantGemmAll2All(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_internal_format: bool = True,
        comm_context=None,
    ):
        """
        Quantized fused GEMM + All2All.

        Semantics::

            gemm_out = quant_gemm(input, weight, weight_scale, per_token_scale)
            output   = all_to_all(gemm_out split by output columns, gathered by rows)

        Args:
            weight (torch.Tensor): Int8 weight. ``trans_weight=True`` expects
                ``(K, N)``; ``False`` expects ``(N, K)``.
            weight_scale (torch.Tensor): Per-output-channel scale, shape ``(N,)``.
            trans_weight (bool): Whether weight layout is transposed.
            process_group (Optional[ProcessGroup]): Distributed group.
            output_dtype (torch.dtype): Output dtype for reference path.
            use_internal_format (bool): Backend hint for xops, ignored by torch reference.
            comm_context: Optional runtime/context object for backend implementations.
        """
        super().__init__()
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.weight = weight
        self.weight_scale = weight_scale
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.output_dtype = output_dtype
        self.use_internal_format = use_internal_format
        self.comm_context = comm_context

    def forward(self, input: torch.Tensor, per_token_scale: torch.Tensor, workspace: Optional[torch.Tensor] = None):
        output = _quant_gemm(
            input,
            self.weight,
            self.weight_scale,
            per_token_scale,
            self.trans_weight,
            self.output_dtype,
        )
        if _is_dist_initialized():
            process_group = self.process_group or _get_default_group()
            world_size = dist.get_world_size(group=process_group)
            if output.shape[-1] % world_size != 0:
                raise ValueError(f"output columns {output.shape[-1]} must be divisible by world_size {world_size}.")
            rank = dist.get_rank(group=process_group)
            send = torch.stack(list(output.chunk(world_size, dim=-1)), dim=0).contiguous()
            gathered = [torch.empty_like(send) for _ in range(world_size)]
            dist.all_gather(gathered, send, group=process_group)
            output = torch.cat([peer_chunks[rank] for peer_chunks in gathered], dim=0)
        return output

    def estimate_shmem_size_mb(self, *, process_group: Optional[dist.ProcessGroup] = None, max_tokens: Optional[int] = None) -> int:
        del process_group, max_tokens
        return 20

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        return (
            f"{weight_shape=}, weight_scale_shape={tuple(self.weight_scale.shape)}, "
            f"{self.trans_weight=}, output_dtype={self.output_dtype}"
        ).replace("self.", "")


class MojoAll2AllQuantGemm(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_internal_format: bool = True,
        comm_context=None,
    ):
        """
        Quantized fused All2All + GEMM.

        Semantics::

            gathered = all_to_all(input split by rows, gathered by K shards)
            output   = quant_gemm(gathered, weight, weight_scale, per_token_scale)

        Args:
            weight (torch.Tensor): Int8 weight. ``trans_weight=True`` expects
                ``(K, N)``; ``False`` expects ``(N, K)``.
            weight_scale (torch.Tensor): Per-output-channel scale, shape ``(N,)``.
            trans_weight (bool): Whether weight layout is transposed.
            process_group (Optional[ProcessGroup]): Distributed group.
            output_dtype (torch.dtype): Output dtype for reference path.
            use_internal_format (bool): Backend hint for xops, ignored by torch reference.
            comm_context: Optional runtime/context object for backend implementations.
        """
        super().__init__()
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.weight = weight
        self.weight_scale = weight_scale
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.output_dtype = output_dtype
        self.use_internal_format = use_internal_format
        self.comm_context = comm_context

    def forward(self, input: torch.Tensor, per_token_scale: torch.Tensor, workspace: Optional[torch.Tensor] = None):
        if _is_dist_initialized():
            process_group = self.process_group or _get_default_group()
            world_size = dist.get_world_size(group=process_group)
            rank = dist.get_rank(group=process_group)
            if input.shape[0] % world_size != 0:
                raise ValueError(f"input rows {input.shape[0]} must be divisible by world_size {world_size}.")
            send = torch.stack(list(input.chunk(world_size, dim=0)), dim=0).contiguous()
            gathered = [torch.empty_like(send) for _ in range(world_size)]
            dist.all_gather(gathered, send, group=process_group)
            input = torch.cat([peer_chunks[rank] for peer_chunks in gathered], dim=-1)
            rows_per_rank = per_token_scale.shape[0] // world_size
            per_token_scale = per_token_scale[rank * rows_per_rank:(rank + 1) * rows_per_rank]

        output = _quant_gemm(
            input,
            self.weight,
            self.weight_scale,
            per_token_scale,
            self.trans_weight,
            self.output_dtype,
        )
        return output

    def estimate_shmem_size_mb(self, *, process_group: Optional[dist.ProcessGroup] = None, max_tokens: Optional[int] = None) -> int:
        del process_group, max_tokens
        return 20

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        return (
            f"{weight_shape=}, weight_scale_shape={tuple(self.weight_scale.shape)}, "
            f"{self.trans_weight=}, output_dtype={self.output_dtype}"
        ).replace("self.", "")
