import os
from typing import Optional

import torch
import torch.distributed as dist

from mojo_opset.backends.ttx.kernels import allgather_gemm_impl
from mojo_opset.backends.ttx.kernels import allgather_gemm_peer_mem_size
from mojo_opset.backends.ttx.kernels import gemm_allreduce_impl
from mojo_opset.backends.ttx.kernels import gemm_allreduce_peer_mem_size
from mojo_opset.backends.ttx.kernels import gemm_reduce_scatter_impl
from mojo_opset.backends.ttx.kernels import gemm_reduce_scatter_peer_mem_size
from mojo_opset.core import MojoAllGatherGemm
from mojo_opset.core import MojoGemmAllReduce
from mojo_opset.core import MojoGemmReduceScatter
from mojo_opset.runtime import MojoSymmetricMemoryManager


def _get_ttx_shmem_size_mb():
    return int(os.environ.get("MOJO_TTX_SHMEM_SIZE_MB", "1024"))


class TTXAllGatherGemm(MojoAllGatherGemm):
    """Triton-based fused AllGather + GEMM on Ascend NPU via aclshmem."""

    supported_platforms_list = ["npu"]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        gather_dim: int = 0,
    ):
        super().__init__(weight, bias, trans_weight, process_group, gather_dim)
        self._peer_mem = None
        self._runtime = None
        self._rank = None
        self._world_size = None

    def _ensure_shmem(self, K: int, dtype: torch.dtype) -> None:
        if self._rank is None:
            runtime = MojoSymmetricMemoryManager.get_or_create(
                process_group=self.process_group,
                backend="ttx",
                shmem_heap_size_mb=_get_ttx_shmem_size_mb(),
            )
            runtime.get_backend_manager()
            self._runtime = runtime
            self._rank = runtime.rank
            self._world_size = runtime.world_size

        flat_size = allgather_gemm_peer_mem_size(K, self._world_size)
        self._peer_mem = self._runtime.allocate_peer_mem(flat_size, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return super().forward(input)

        if self.gather_dim != 0:
            input = input.movedim(self.gather_dim, 0)

        orig_shape = input.shape
        K = input.shape[-1]
        input_2d = input.reshape(-1, K).contiguous()

        self._ensure_shmem(K, input.dtype)

        if self.trans_weight:
            weight = self.weight
        else:
            weight = self.weight.t().contiguous()

        N = weight.shape[1]
        M = input_2d.shape[0]
        output = torch.empty(
            [M * self._world_size, N],
            dtype=input.dtype,
            device=input.device,
        )

        allgather_gemm_impl(
            input_2d, weight, output, self._peer_mem,
            self._rank, self._world_size,
        )

        if self.bias is not None:
            output.add_(self.bias)

        out_shape = list(orig_shape)
        out_shape[0] *= self._world_size
        out_shape[-1] = N
        output = output.reshape(out_shape)

        if self.gather_dim != 0:
            output = output.movedim(0, self.gather_dim)

        return output


class TTXGemmAllReduce(MojoGemmAllReduce):
    """Triton-based fused GEMM + AllReduce on Ascend NPU via aclshmem."""

    supported_platforms_list = ["npu"]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(weight, bias, trans_weight, process_group)
        self._peer_mem = None
        self._runtime = None
        self._rank = None
        self._world_size = None

    def _ensure_shmem(self, dtype: torch.dtype) -> None:
        if self._rank is None:
            runtime = MojoSymmetricMemoryManager.get_or_create(
                process_group=self.process_group,
                backend="ttx",
                shmem_heap_size_mb=_get_ttx_shmem_size_mb(),
            )
            runtime.get_backend_manager()
            self._runtime = runtime
            self._rank = runtime.rank
            self._world_size = runtime.world_size

        flat_size = gemm_allreduce_peer_mem_size()
        self._peer_mem = self._runtime.allocate_peer_mem(flat_size, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return super().forward(input)

        orig_shape = input.shape
        K = input.shape[-1]
        input_2d = input.reshape(-1, K).contiguous()

        self._ensure_shmem(input.dtype)

        if self.trans_weight:
            weight = self.weight
        else:
            weight = self.weight.t().contiguous()

        N = weight.shape[1]
        M = input_2d.shape[0]
        # zero-init required: kernel uses tl.atomic_add to accumulate from all ranks
        output = torch.zeros(
            [M, N], dtype=input.dtype, device=input.device,
        )

        gemm_allreduce_impl(
            input_2d, weight, output, self._peer_mem,
            self._rank, self._world_size,
        )

        if self.bias is not None:
            output.add_(self.bias)

        out_shape = list(orig_shape)
        out_shape[-1] = N
        return output.reshape(out_shape)


class TTXGemmReduceScatter(MojoGemmReduceScatter):
    """Triton-based fused GEMM + ReduceScatter on Ascend NPU via aclshmem."""

    supported_platforms_list = ["npu"]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        trans_weight: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        scatter_dim: int = 0,
    ):
        super().__init__(weight, bias, trans_weight, process_group, scatter_dim)
        self._peer_mem = None
        self._runtime = None
        self._rank = None
        self._world_size = None

    def _ensure_shmem(self, dtype: torch.dtype) -> None:
        if self._rank is None:
            runtime = MojoSymmetricMemoryManager.get_or_create(
                process_group=self.process_group,
                backend="ttx",
                shmem_heap_size_mb=_get_ttx_shmem_size_mb(),
            )
            runtime.get_backend_manager()
            self._runtime = runtime
            self._rank = runtime.rank
            self._world_size = runtime.world_size

        flat_size = gemm_reduce_scatter_peer_mem_size()
        self._peer_mem = self._runtime.allocate_peer_mem(flat_size, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return super().forward(input)

        if self.scatter_dim != 0:
            input = input.movedim(self.scatter_dim, 0)

        orig_shape = input.shape
        K = input.shape[-1]
        input_2d = input.reshape(-1, K).contiguous()

        self._ensure_shmem(input.dtype)

        if self.trans_weight:
            weight = self.weight
        else:
            weight = self.weight.t().contiguous()

        N = weight.shape[1]
        M = input_2d.shape[0]
        M_local = M // self._world_size
        # zero-init required: kernel uses tl.atomic_add to accumulate from all ranks
        output = torch.zeros(
            [M_local, N], dtype=input.dtype, device=input.device,
        )

        gemm_reduce_scatter_impl(
            input_2d, weight, output, self._peer_mem,
            self._rank, self._world_size,
        )

        if self.bias is not None:
            output.add_(self.bias)

        out_shape = list(orig_shape)
        out_shape[0] = M_local
        out_shape[-1] = N
        output = output.reshape(out_shape)

        if self.scatter_dim != 0:
            output = output.movedim(0, self.scatter_dim)

        return output
