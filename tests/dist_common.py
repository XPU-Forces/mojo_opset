# WARNING(liuyuan): DO NOT import torch with TORCH_DEVICE_BACKEND_AUTOLOAD=1 (default is 1) or import torch_npu globally in this file.
# WARNING(liuyuan): DO NOT import torch with TORCH_DEVICE_BACKEND_AUTOLOAD=1 (default is 1) or import torch_npu globally in this file.
# WARNING(liuyuan): DO NOT import torch with TORCH_DEVICE_BACKEND_AUTOLOAD=1 (default is 1) or import torch_npu globally in this file.
# NOTE(liuyuan): remove the comments above when torch_npu.distributed.gather keeps the same signature with torch.distributed.gather
import multiprocessing as mp
import socket
import os
import functools
import traceback
import pytest


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _dist_worker(rank, world_size, backend, port, fn, error_queue, args, kwargs):
    import torch.distributed as dist
    # NOTE(liuyuan): specially desgined for 'spawn' and 'forkserver' which require the function object is the true one to pickle after we have wrapped it.
    actual_fn = getattr(fn, '__wrapped__', fn)
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        actual_fn(*args, **kwargs)
    except Exception:
        error_queue.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def dist_test(world_size=8, backend="gloo", start_process="spawn"):
    """Decorator that runs the wrapped function in *world_size* forked processes
    with ``torch.distributed`` already initialised.

    Usage::

        @dist_test(world_size=8)
        def test_tensor_parallel():
            rank = dist.get_rank()
            mesh = init_device_mesh(...)
            ...
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            port = _find_free_port()
            ctx = mp.get_context(start_process)
            error_queue = ctx.SimpleQueue()
            procs = []

            for rank in range(world_size):
                p = ctx.Process(
                    target=_dist_worker,
                    args=(rank, world_size, backend, port, wrapper, error_queue, args, kwargs),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            errors = []
            while not error_queue.empty():
                errors.append(error_queue.get())

            if errors:
                msg = "\n".join(f"[Rank {rank}]\n{tb}" for rank, tb in errors)
                pytest.fail(f"Distributed test failed:\n{msg}")

            for i, p in enumerate(procs):
                if p.exitcode != 0:
                    pytest.fail(f"[Rank {i}] exited with code {p.exitcode}")

        return wrapper

    return decorator
