from typing import List, Optional, Union

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
    input_i8: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trans_weight: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference int8 GEMM emulated in float32, mirroring MojoQuantGemm.

    Computation: ``output = (input_i8 @ weight_i8) * input_scale[:, None] * weight_scale``.
    ``trans_weight=True`` means the weight is stored as ``(N, K)``; otherwise
    ``(K, N)`` and is transposed via ``.mT`` to align with the int8-GEMM
    contract.
    """
    input_scale = input_scale.reshape(-1)
    if input_scale.numel() != input_i8.shape[0]:
        raise ValueError(
            f"input_scale must contain one scale per input row, got {input_scale.numel()} and {input_i8.shape[0]}"
        )
    w = weight if trans_weight else weight.mT  # [N, K]
    out = input_i8.float() @ w.float().T       # [M, N]
    out = out * input_scale.float().unsqueeze(-1) * weight_scale.float()
    return out.to(output_dtype)


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


class MojoAllGatherQuantGemm(MojoOperator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        process_group: Optional[dist.ProcessGroup] = None,
        gather_dim: int = 0,
        **kwargs,
    ):
        """SP-fused AllGather + quantized (int8) GEMM.

        Each rank holds an int8 sequence shard plus its per-token scale.
        AllGather reconstructs the full sequence + scale across TP ranks, then
        an int8 @ int8 GEMM with the per-channel weight scale produces the
        bf16 output. Typical use: Megatron QKV projection on a smooth-quant
        path where activations are pre-quantized to int8 with a per-token
        scale.

        Semantics::

            x_full     = all_gather(x_local,     dim=gather_dim)   # int8
            scale_full = all_gather(scale_local, dim=gather_dim)   # fp32
            output     = quant_gemm(x_full, scale_full,
                                    weight, weight_scale)          # output_dtype

        When ``torch.distributed`` is not initialised, AllGather is identity.

        Args:
            in_features (int): Logical K dim of the int8 weight.
            out_features (int): Logical N dim of the int8 weight and weight_scale.
            output_dtype (torch.dtype): Dequantized output dtype (default bf16).
            trans_weight (bool): If True the weight is stored as (N, K).
            quant_dtype (torch.dtype): Quantized dtype, only ``int8`` supported.
            process_group (Optional[ProcessGroup]): TP group for AllGather.
            gather_dim (int): Dimension to AllGather along (default 0 = sequence).
            **kwargs: Tensor factory kwargs forwarded to the weight buffers.
        """
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8."
            )
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = (out_features, in_features) if trans_weight else (in_features, out_features)
        weight_factory_kwargs = {**self.tensor_factory_kwargs, "dtype": quant_dtype}
        weight_scale_factory_kwargs = {**self.tensor_factory_kwargs, "dtype": torch.bfloat16}
        self.quant_dtype = quant_dtype
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.gather_dim = gather_dim
        self.register_buffer("weight", torch.empty(self.weight_shape, **weight_factory_kwargs))
        self.register_buffer("weight_scale", torch.empty(out_features, **weight_scale_factory_kwargs))

    def forward(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): int8 ``(seq_local, in_features)`` — local
                sequence shard of the quantized activation.
            input_scale (torch.Tensor): fp32 ``(seq_local,)`` — per-token
                runtime activation scale, also sequence-sharded.

        Returns:
            torch.Tensor: ``output_dtype`` ``(seq_full, out_features)`` after
                AllGather along ``gather_dim`` and the quantized GEMM. In the
                single-rank fallback, ``seq_full == seq_local``.
        """
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input K {input.shape[-1]} must match in_features {self.in_features}."
            )
        if _is_dist_initialized():
            pg = self.process_group or _get_default_group()
            input = fc.all_gather_tensor(input, gather_dim=self.gather_dim, group=pg)
            input_scale = fc.all_gather_tensor(input_scale, gather_dim=self.gather_dim, group=pg)
        return _quant_gemm(
            input, input_scale, self.weight, self.weight_scale,
            self.trans_weight, self.output_dtype,
        )

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        return (
            f"{weight_shape=}, in={self.in_features}, out={self.out_features}, "
            f"trans_weight={self.trans_weight}, gather_dim={self.gather_dim}"
        )


class MojoQuantGemmReduceScatter(MojoOperator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        process_group: Optional[dist.ProcessGroup] = None,
        scatter_dim: int = 0,
        **kwargs,
    ):
        """SP-fused quantized (int8) GEMM + ReduceScatter.

        Each rank computes a local quantized GEMM on its column-shard of the
        input; the resulting bf16 partial output is reduce-scattered along
        ``scatter_dim`` so each rank ends up with its sequence shard of the
        summed result. Typical use: Megatron output projection on a
        smooth-quant path.

        Semantics::

            partial = quant_gemm(input_local, input_scale,
                                 weight_local, weight_scale)   # bf16, full seq
            output  = reduce_scatter(partial, dim=scatter_dim)
                    # shape along scatter_dim shrinks by world_size

        When ``torch.distributed`` is not initialised, ReduceScatter is identity.

        Args:
            in_features (int): Local K dim of the int8 weight (= K_global / tp).
            out_features (int): Logical N dim of the int8 weight and weight_scale.
            output_dtype (torch.dtype): Dequantized output dtype (default bf16).
            trans_weight (bool): If True the weight is stored as (N, K).
            quant_dtype (torch.dtype): Quantized dtype, only ``int8`` supported.
            process_group (Optional[ProcessGroup]): TP group for ReduceScatter.
            scatter_dim (int): Dimension to scatter along (default 0 = sequence).
            **kwargs: Tensor factory kwargs forwarded to the weight buffers.
        """
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(
                f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8."
            )
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = (out_features, in_features) if trans_weight else (in_features, out_features)
        weight_factory_kwargs = {**self.tensor_factory_kwargs, "dtype": quant_dtype}
        weight_scale_factory_kwargs = {**self.tensor_factory_kwargs, "dtype": torch.bfloat16}
        self.quant_dtype = quant_dtype
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight
        self.process_group = process_group
        self.scatter_dim = scatter_dim
        self.register_buffer("weight", torch.empty(self.weight_shape, **weight_factory_kwargs))
        self.register_buffer("weight_scale", torch.empty(out_features, **weight_scale_factory_kwargs))

    def forward(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): int8 ``(seq_full, in_features)`` — each rank's
                column-shard of the quantized activation, full sequence.
            input_scale (torch.Tensor): fp32 ``(seq_full,)`` — per-token scale,
                same on every rank for the full sequence.

        Returns:
            torch.Tensor: ``output_dtype`` ``(seq_full / world_size, out_features)``
                after the quantized GEMM and ReduceScatter along ``scatter_dim``.
                Single-rank fallback returns the full unscattered output.
        """
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input K {input.shape[-1]} must match in_features {self.in_features}."
            )
        partial = _quant_gemm(
            input, input_scale, self.weight, self.weight_scale,
            self.trans_weight, self.output_dtype,
        )
        if _is_dist_initialized():
            pg = self.process_group or _get_default_group()
            world_size = dist.get_world_size(group=pg)
            rank = dist.get_rank(group=pg)
            chunks = list(partial.chunk(world_size, dim=self.scatter_dim))
            reduced = torch.empty_like(chunks[rank])
            dist.reduce_scatter(reduced, chunks, op=dist.ReduceOp.SUM, group=pg)
            return reduced
        return partial

    def extra_repr(self) -> str:
        weight_shape = tuple(self.weight.shape) if isinstance(self.weight, torch.Tensor) else None
        return (
            f"{weight_shape=}, in={self.in_features}, out={self.out_features}, "
            f"trans_weight={self.trans_weight}, scatter_dim={self.scatter_dim}"
        )
