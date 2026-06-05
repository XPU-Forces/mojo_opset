"""Accuracy tests for MojoDeepEPDispatch / MojoDeepEPCombine."""

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset import MojoDeepEPCombine
from mojo_opset import MojoDeepEPDispatch
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.acc import check_tol_diff
from mojo_opset.utils.platform import get_torch_device


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _make_global_inputs(world_size, num_tokens_sp, hidden, num_experts, top_k, dtype, device):
    global_tokens = num_tokens_sp * world_size
    if dtype == torch.int8:
        hidden_states = torch.randint(-128, 127, (global_tokens, hidden), dtype=torch.int8, device=device)
    else:
        hidden_states = torch.randn(global_tokens, hidden, dtype=dtype, device=device)
    gating = torch.rand(global_tokens, num_experts, dtype=torch.float32, device=device)
    top_k_logits, top_k_indices = torch.topk(gating, top_k)
    top_k_gates = torch.nn.functional.softmax(top_k_logits, dim=-1)
    return hidden_states, top_k_gates, top_k_indices.to(torch.int32)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _xops_skip_if_unsupported(num_experts, world_size):
    if world_size < 1:
        pytest.skip("MOJO_XOPS_TEST_WORLD_SIZE must be >= 1")
    if num_experts % world_size != 0:
        pytest.skip(f"num_experts={num_experts} must be divisible by world_size={world_size}")
    if torch.npu.device_count() < world_size:
        pytest.skip(f"Need {world_size} NPU devices, got {torch.npu.device_count()}")
    local_experts = num_experts // world_size
    from mojo_opset_ext.backends.xpu_ops.operators.moe import is_deep_ep_local_experts_supported

    if world_size > 1 and not is_deep_ep_local_experts_supported(local_experts):
        pytest.skip(
            f"DeepEPMoe kernels require local_experts==1 or local_experts%8==0, got {local_experts}"
        )


def _run_distributed(case_args, world_size, worker):
    ctx = mp.get_context("forkserver")
    port = _find_free_port()
    result_queue = ctx.Queue()
    processes = []
    for rank in range(world_size):
        process = ctx.Process(
            target=worker,
            args=(rank, world_size, port, result_queue, case_args),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    errors = [(rank, error) for rank, error in results if error is not None]
    if errors:
        message = "\n".join(f"[Rank {rank}]\n{error}" for rank, error in errors)
        pytest.fail(f"Distributed DeepEP test failed:\n{message}")

    for rank, process in enumerate(processes):
        if process.exitcode != 0:
            pytest.fail(f"[Rank {rank}] exited with code {process.exitcode}")


# (num_experts, top_k, hidden, num_tokens_sp) — kept moderate to keep CI cost bounded.
deep_ep_cases = [
    (8, 2, 256, 16),
    (16, 4, 512, 32),
    (64, 8, 1024, 64),
    (384, 8, 3072, 64),
]


# ---------------------------------------------------------------------------
# Dispatch — single unified test driving forward_diff_with for ops vs torch.
# ---------------------------------------------------------------------------


def _dispatch_compare(rank, world_size, port, queue, case_args):
    """Ops-vs-torch dispatch comparison. Sets up hccl + symmetric memory when
    world_size>1; ``queue`` is the multiprocess result queue (``None`` for an
    in-process single-rank call)."""
    shmem_manager = None
    try:
        if world_size > 1:
            import torch_npu
            from mojo_opset.runtime import MojoSymmetricMemoryManager

            torch_npu.npu.set_device(rank)
            dist.init_process_group(
                backend="hccl",
                rank=rank,
                world_size=world_size,
                init_method=f"tcp://127.0.0.1:{port}",
            )
            shmem_manager = MojoSymmetricMemoryManager.get_or_create(
                backend="xops", shmem_heap_size_mb=2048
            )
            shmem_manager.get_backend_manager()

        num_tokens_sp, hidden, top_k, num_experts, dtype, use_smooth_scale = case_args
        device = get_torch_device()
        torch.manual_seed(0)
        global_hidden, global_gates, global_indices = _make_global_inputs(
            world_size, num_tokens_sp, hidden, num_experts, top_k, dtype, device
        )
        smooth_scale = (
            torch.rand(num_experts, hidden, dtype=torch.float32, device=device) + 0.5
            if use_smooth_scale
            else None
        )
        s = rank * num_tokens_sp
        e = s + num_tokens_sp
        local_hidden = global_hidden[s:e].contiguous()
        local_gates = global_gates[s:e].contiguous()
        local_indices = global_indices[s:e].contiguous()

        op = MojoDeepEPDispatch(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
            quant_mode="per_token" if use_smooth_scale else "none",
        ).to(device)
        op_ref = MojoDeepEPDispatch._registry.get("torch")(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)

        # xops dispatch returns an upper-bound buffer (q_len*group_size*top_k rows);
        # only the first R rows are valid where R = sum(expert_token_cnt_per_rank).
        # The torch reference returns exactly R rows. Trim before comparing index 0 / 3.
        out = op.forward(local_hidden, local_gates, local_indices, smooth_scale=smooth_scale)
        ref = op_ref.forward(local_hidden, local_gates, local_indices, smooth_scale=smooth_scale)

        r_actual = int(out[1].sum().item())
        r_ref = ref[0].size(0)
        assert r_actual == r_ref, f"R mismatch: xops={r_actual}, ref={r_ref}"

        # idx 0: int8 quant rounding can differ by ±1 between hw kernel and torch ref.
        hidden_atol = 1 if use_smooth_scale else 0
        torch.testing.assert_close(out[0][:r_actual], ref[0], atol=hidden_atol, rtol=0)
        torch.testing.assert_close(out[1], ref[1], atol=0, rtol=0)
        # xops cumsum dtype is int32 (wrapper init), torch ref is int64 — values must agree.
        torch.testing.assert_close(out[2].to(torch.int64), ref[2], atol=0, rtol=0)
        if use_smooth_scale:
            torch.testing.assert_close(out[3][:r_actual], ref[3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[4], ref[4], atol=0, rtol=0)
        torch.testing.assert_close(out[5], ref[5], atol=0, rtol=0)

        if queue is not None:
            queue.put((rank, None))
    except Exception:
        if queue is not None:
            queue.put((rank, traceback.format_exc()))
        else:
            raise
    finally:
        if shmem_manager is not None:
            shmem_manager.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("num_experts, top_k, hidden, num_tokens_sp", deep_ep_cases)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("use_smooth_scale", [False, True])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@auto_switch_platform()
@bypass_not_implemented
def test_deep_ep_dispatch(world_size, num_experts, top_k, hidden, num_tokens_sp, dtype, use_smooth_scale):
    """Compare active backend's dispatch with torch backend via forward_diff_with."""
    if dtype == torch.int8 and use_smooth_scale:
        pytest.skip("int8 input + per_token quant is not supported by the kernel.")
    if os.environ.get("MOJO_BACKEND", "").strip().lower() != "xops":
        pytest.skip("ops-vs-torch comparison requires MOJO_BACKEND=xops")

    _xops_skip_if_unsupported(num_experts, world_size)
    case_args = (num_tokens_sp, hidden, top_k, num_experts, dtype, use_smooth_scale)
    _run_distributed(case_args, world_size, _dispatch_compare)


# ---------------------------------------------------------------------------
# Combine — single unified test driving forward_diff_with for ops vs torch.
# ---------------------------------------------------------------------------


def _combine_compare(rank, world_size, port, queue, case_args):
    """Ops-vs-torch combine comparison. Sets up hccl + symmetric memory when
    world_size>1; ``queue`` is the multiprocess result queue (``None`` for an
    in-process single-rank call)."""
    shmem_manager = None
    try:
        if world_size > 1:
            import torch_npu
            from mojo_opset.runtime import MojoSymmetricMemoryManager

            torch_npu.npu.set_device(rank)
            dist.init_process_group(
                backend="hccl",
                rank=rank,
                world_size=world_size,
                init_method=f"tcp://127.0.0.1:{port}",
            )
            shmem_manager = MojoSymmetricMemoryManager.get_or_create(
                backend="xops", shmem_heap_size_mb=2048
            )
            shmem_manager.get_backend_manager()

        num_tokens_sp, hidden, top_k, num_experts, dtype = case_args
        device = get_torch_device()
        torch.manual_seed(0)
        global_hidden, global_gates, global_indices = _make_global_inputs(
            world_size, num_tokens_sp, hidden, num_experts, top_k, dtype, device
        )
        s = rank * num_tokens_sp
        e = s + num_tokens_sp
        local_hidden = global_hidden[s:e].contiguous()
        local_gates = global_gates[s:e].contiguous()
        local_indices = global_indices[s:e].contiguous()

        # Build deterministic combine inputs by running torch dispatch.
        dispatch_op = MojoDeepEPDispatch._registry.get("torch")(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)
        expand, _, _, _, scatter_index, expert_token_count = dispatch_op(
            local_hidden, local_gates, local_indices
        )

        op = MojoDeepEPCombine(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)
        op_ref = MojoDeepEPCombine._registry.get("torch")(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)

        # xops combine asserts expand.size(0) >= q_len * top_k (deep_ep.cpp:468).
        # Torch dispatch returns R-sized expand; pad to the required upper bound for
        # xops, but keep the R-sized version for torch combine which expects it.
        upper = num_tokens_sp * world_size * top_k
        if expand.size(0) < upper:
            padded = torch.zeros(upper, expand.size(1), dtype=expand.dtype, device=device)
            padded[: expand.size(0)] = expand
            expand_for_xops = padded
        else:
            expand_for_xops = expand

        xops_out = op.forward(
            expand_for_xops, local_gates, scatter_index, expert_token_count, num_tokens_sp,
        )
        ref_out = op_ref.forward(
            expand, local_gates, scatter_index, expert_token_count, num_tokens_sp,
        )
        check_tol_diff(xops_out, ref_out, mixed_tol=True)

        if queue is not None:
            queue.put((rank, None))
    except Exception:
        if queue is not None:
            queue.put((rank, traceback.format_exc()))
        else:
            raise
    finally:
        if shmem_manager is not None:
            shmem_manager.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("num_experts, top_k, hidden, num_tokens_sp", deep_ep_cases)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@auto_switch_platform()
@bypass_not_implemented
def test_deep_ep_combine(world_size, num_experts, top_k, hidden, num_tokens_sp, dtype):
    """Compare active backend's combine with torch backend via forward_diff_with."""
    if os.environ.get("MOJO_BACKEND", "").strip().lower() != "xops":
        pytest.skip("ops-vs-torch comparison requires MOJO_BACKEND=xops")

    _xops_skip_if_unsupported(num_experts, world_size)
    case_args = (num_tokens_sp, hidden, top_k, num_experts, dtype)
    _run_distributed(case_args, world_size, _combine_compare)
