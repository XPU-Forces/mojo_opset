import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist


def _resolve_rank_world_size(process_group: Optional[dist.ProcessGroup]) -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        group = process_group if process_group is not None else dist.group.WORLD
        return dist.get_rank(group=group), dist.get_world_size(group=group)
    return 0, 1


class MojoSymmetricMemoryManager:
    """Process-group scoped symmetric-memory runtime handle.

    The first xops implementation intentionally does not expose a public
    symmetric allocator. xpu_ops operators still own their internal shmem
    allocations; this class owns the backend manager and team cache.
    """

    supports_external_symmetric_allocation = False

    _instances: Dict[Tuple[str, int, int, int, int], "MojoSymmetricMemoryManager"] = {}
    _instances_lock = threading.RLock()

    @classmethod
    def get_or_create(
        cls,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        backend: str = "xops",
        shmem_heap_size_mb: Optional[int] = None,
    ) -> "MojoSymmetricMemoryManager":
        rank, world_size = _resolve_rank_world_size(process_group)
        heap_mb = cls._resolve_heap_size_mb(shmem_heap_size_mb)
        key = (backend, id(process_group), rank, world_size, heap_mb)
        with cls._instances_lock:
            manager = cls._instances.get(key)
            if manager is None:
                manager = cls(
                    process_group=process_group,
                    backend=backend,
                    shmem_heap_size_mb=heap_mb,
                )
                cls._instances[key] = manager
        return manager

    @classmethod
    def finalize_all(cls) -> None:
        with cls._instances_lock:
            managers = list(cls._instances.values())
            cls._instances.clear()
        for manager in managers:
            manager.close()

    @staticmethod
    def _resolve_heap_size_mb(shmem_heap_size_mb: Optional[int]) -> int:
        if shmem_heap_size_mb is not None:
            return int(shmem_heap_size_mb)
        return int(os.getenv("MOJO_SHMEM_SIZE_MB", os.getenv("MOJO_XOPS_SHMEM_SIZE_MB", "256")))

    def __init__(
        self,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        backend: str = "xops",
        shmem_heap_size_mb: int = 256,
    ):
        if backend not in ("xops", "ttx"):
            raise NotImplementedError(f"Unsupported symmetric memory backend: {backend}")
        self.process_group = process_group
        self.backend = backend
        self.rank, self.world_size = _resolve_rank_world_size(process_group)
        self.shmem_heap_size_mb = int(shmem_heap_size_mb)
        self._backend_manager = None
        self._team_cache: Dict[Tuple[int, int, int, int], int] = {}
        self._closed = False
        self._lock = threading.RLock()

    @property
    def shmem_heap_size_bytes(self) -> int:
        return self.shmem_heap_size_mb * 1024 * 1024

    def get_backend_manager(self):
        with self._lock:
            self._ensure_open()
            if self._backend_manager is None:
                if self.backend == "xops":
                    self._init_xops_backend()
                elif self.backend == "ttx":
                    self._init_ttx_backend()
            return self._backend_manager

    def _init_xops_backend(self):
        import xpu_ops
        from xpu_ops.modules import ShmemManager

        xpu_ops.load_xpu_ops(False)
        self._backend_manager = ShmemManager(
            mem_size=self.shmem_heap_size_bytes,
            process_group=self.process_group,
        )

    def _init_ttx_backend(self):
        import ctypes
        import importlib.util

        spec = importlib.util.find_spec("shmem")
        if spec is None:
            raise ImportError("shmem package not found; required for TTX backend")
        shmem_dir = os.path.dirname(spec.origin)
        ctypes.CDLL(os.path.join(shmem_dir, "libshmem_utils.so"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.path.join(shmem_dir, "libshmem.so"), mode=ctypes.RTLD_GLOBAL)
        import shmem as ash

        process_group = self.process_group or dist.group.WORLD
        _UID_SIZE = 136
        uid_tensor = torch.zeros(_UID_SIZE, dtype=torch.uint8, device=f"npu:{torch.npu.current_device()}")
        if self.rank == 0:
            uid_bytes = ash.aclshmem_get_unique_id()
            uid_tensor.copy_(torch.tensor(list(uid_bytes), dtype=torch.uint8))
        dist.broadcast(uid_tensor, src=0, group=process_group)
        uid = bytes(uid_tensor.cpu().tolist())

        ret = ash.aclshmem_init_using_unique_id(
            self.rank, self.world_size,
            self.shmem_heap_size_bytes, uid,
        )
        if ret != 0:
            raise RuntimeError(f"aclshmem_init_using_unique_id failed: ret={ret}")
        self._backend_manager = ash

    def allocate_peer_mem(self, flat_size: int, dtype: torch.dtype) -> torch.Tensor:
        """Allocate or reuse a symmetric tensor from the shmem heap (TTX backend).

        Returns a cached tensor if one exists with sufficient size and matching dtype.
        Only one peer_mem buffer is active per dtype at a time.
        """
        with self._lock:
            self._ensure_open()
            if self.backend != "ttx":
                raise RuntimeError("allocate_peer_mem is only supported for TTX backend")
            cache_key = ("_peer_mem", dtype)
            existing = self._team_cache.get(cache_key)
            if existing is not None and existing.numel() >= flat_size:
                return existing
            import shmem as ash
            tensor = ash.aclshmem_create_tensor([flat_size], dtype=dtype, device_id=torch.npu.current_device())
            self._team_cache[cache_key] = tensor
            return tensor

    def get_or_create_team(
        self,
        *,
        parent_team: int = 0,
        pe_start: int = 0,
        pe_stride: int = 1,
        pe_size: Optional[int] = None,
    ) -> int:
        pe_size = self.world_size if pe_size is None else int(pe_size)
        key = (int(parent_team), int(pe_start), int(pe_stride), pe_size)
        if key == (0, 0, 1, self.world_size):
            return 0

        with self._lock:
            self._ensure_open()
            if key not in self._team_cache:
                manager = self.get_backend_manager()
                self._team_cache[key] = manager.team_split_strided(*key)
            return self._team_cache[key]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            manager = self._backend_manager
            self._backend_manager = None
            self._team_cache.clear()
            self._closed = True

        if manager is not None:
            if self.backend == "ttx":
                manager.aclshmem_finalize()
            elif hasattr(manager, "finalize"):
                manager.finalize()
            elif hasattr(manager, "destroy"):
                manager.destroy()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MojoSymmetricMemoryManager has been closed.")


class MojoComputeCommContext:
    """Per-operator cache for communication-computation fused operators."""

    def __init__(self, runtime: MojoSymmetricMemoryManager, op_name: str):
        self.runtime = runtime
        self.op_name = op_name
        self._op_cache: Dict[Tuple[Any, ...], Any] = {}
        self._tensor_cache: Dict[Tuple[str, Tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}
        self._lock = threading.RLock()

    def get_or_create_op(self, key: Tuple[Any, ...], factory):
        with self._lock:
            if key not in self._op_cache:
                self._op_cache[key] = factory()
            return self._op_cache[key]

    def get_tensor(
        self,
        name: str,
        shape,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        shape = tuple(shape)
        key = (name, shape, dtype, device)
        with self._lock:
            tensor = self._tensor_cache.get(key)
            if tensor is None:
                tensor = torch.empty(shape, dtype=dtype, device=device)
                self._tensor_cache[key] = tensor
            return tensor

    def close(self) -> None:
        with self._lock:
            self._op_cache.clear()
            self._tensor_cache.clear()
