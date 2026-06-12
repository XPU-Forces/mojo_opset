import os
import socket
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset import MojoAll2AllQuantGemm
from mojo_opset import MojoQuantGemmAll2All
from mojo_opset.experimental import MojoFusedAGScaleQuant
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_dist_backend
from mojo_opset.utils.platform import get_platform


torch.manual_seed(42)
_PLATFORM = get_platform()
COMM_BACKEND = get_dist_backend()
DEVICE = _PLATFORM if _PLATFORM in ("npu", "mlu") else "cpu"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_gloo(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _destroy_pg():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _quant_gemm_ref(input, weight, weight_scale, per_token_scale, trans_weight=True):
    if trans_weight:
        out = input.float() @ weight.float()
    else:
        out = input.float() @ weight.float().transpose(-2, -1)
    return (out.float() * weight_scale.float().unsqueeze(0) * per_token_scale.float().unsqueeze(-1)).to(torch.bfloat16)


@bypass_not_implemented
def test_quant_gemm_all2all_single_rank():
    m, k, n = 8, 16, 12
    input = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cpu")
    weight = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="cpu")
    weight_scale = torch.rand(n, dtype=torch.float32, device="cpu")
    per_token_scale = torch.rand(m, dtype=torch.float32, device="cpu")

    op = MojoQuantGemmAll2All(weight=weight, weight_scale=weight_scale, trans_weight=True)
    out = op(input, per_token_scale)
    ref = _quant_gemm_ref(input, weight, weight_scale, per_token_scale)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@bypass_not_implemented
def test_all2all_quant_gemm_single_rank():
    m, k, n = 8, 16, 12
    input = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cpu")
    weight = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="cpu")
    weight_scale = torch.rand(n, dtype=torch.float32, device="cpu")
    per_token_scale = torch.rand(m, dtype=torch.float32, device="cpu")

    op = MojoAll2AllQuantGemm(weight=weight, weight_scale=weight_scale, trans_weight=True)
    out = op(input, per_token_scale)
    ref = _quant_gemm_ref(input, weight, weight_scale, per_token_scale)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def _worker_quant_gemm_all2all(rank, world_size, port, inputs, weight, weight_scale, token_scales):
    _init_gloo(rank, world_size, port)
    try:
        op = MojoQuantGemmAll2All(weight=weight, weight_scale=weight_scale, trans_weight=True)
        out = op(inputs[rank], token_scales[rank])

        expected_chunks = []
        for src in range(world_size):
            full = _quant_gemm_ref(inputs[src], weight, weight_scale, token_scales[src])
            expected_chunks.append(full.chunk(world_size, dim=-1)[rank])
        expected = torch.cat(expected_chunks, dim=0)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    finally:
        _destroy_pg()


def test_quant_gemm_all2all_gloo():
    world_size = 2
    m, k, n = 6, 8, 10
    port = _free_port()
    inputs = [torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cpu") for _ in range(world_size)]
    token_scales = [torch.rand(m, dtype=torch.float32, device="cpu") for _ in range(world_size)]
    weight = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="cpu")
    weight_scale = torch.rand(n, dtype=torch.float32, device="cpu")
    mp.spawn(
        _worker_quant_gemm_all2all,
        args=(world_size, port, inputs, weight, weight_scale, token_scales),
        nprocs=world_size,
        join=True,
    )


def _worker_all2all_quant_gemm(rank, world_size, port, inputs, weight, weight_scale, token_scales):
    _init_gloo(rank, world_size, port)
    try:
        op = MojoAll2AllQuantGemm(weight=weight, weight_scale=weight_scale, trans_weight=True)
        out = op(inputs[rank], token_scales[rank])

        gathered = torch.cat([inputs[src].chunk(world_size, dim=0)[rank] for src in range(world_size)], dim=-1)
        rows_per_rank = token_scales[rank].shape[0] // world_size
        local_scale = token_scales[rank][rank * rows_per_rank:(rank + 1) * rows_per_rank]
        expected = _quant_gemm_ref(gathered, weight, weight_scale, local_scale)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    finally:
        _destroy_pg()


def test_all2all_quant_gemm_gloo():
    world_size = 2
    m, k, n = 6, 8, 10
    port = _free_port()
    inputs = [torch.randint(-8, 8, (m, k // world_size), dtype=torch.int8, device="cpu") for _ in range(world_size)]
    token_scales = [torch.rand(m, dtype=torch.float32, device="cpu") for _ in range(world_size)]
    weight = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="cpu")
    weight_scale = torch.rand(n, dtype=torch.float32, device="cpu")
    mp.spawn(
        _worker_all2all_quant_gemm,
        args=(world_size, port, inputs, weight, weight_scale, token_scales),
        nprocs=world_size,
        join=True,
    )


def _is_dist_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _device_count() -> int:
    if DEVICE == "cpu":
        return 1
    device_module = getattr(torch, DEVICE, None)
    if device_module is None or not hasattr(device_module, "device_count"):
        return 0
    return int(device_module.device_count())


def _to_dev(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE) if DEVICE != "cpu" else t


def _synchronize_device():
    if DEVICE == "cpu":
        return
    device_module = getattr(torch, DEVICE, None)
    if device_module is not None and hasattr(device_module, "synchronize"):
        device_module.synchronize()


def _init_dist():
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if DEVICE == "npu":
        import torch_npu  # noqa: F401

    if DEVICE != "cpu":
        device_module = getattr(torch, DEVICE)
        if hasattr(device_module, "set_device"):
            device_module.set_device(rank)

    if not dist.is_initialized():
        dist.init_process_group(COMM_BACKEND)
    return dist.get_rank(), dist.get_world_size()


def _run_torchrun_test(test_fn_name: str, *fn_args, nproc: int = 2, timeout: int = 600):
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, "run_dist_test.py")
        args_repr = ", ".join(repr(arg) for arg in fn_args)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(
                "from mojo_opset.tests.accuracy.operators.test_compute_with_comm_quant "
                f"import {test_fn_name}\n"
                f"{test_fn_name}({args_repr})\n"
            )

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            script_path,
        ]
        subprocess.run(cmd, check=True, timeout=timeout)


def _fused_ag_scale_quant_team_inputs(world_size: int, token_num: int, head_num: int, head_dim: int, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2028)
    return [
        torch.randn(token_num, head_num, head_dim, dtype=torch.float32, generator=generator).to(dtype)
        for _ in range(world_size)
    ]


def _dist_fused_ag_scale_quant(norm_mode: str):
    rank, world_size = _init_dist()
    try:
        eps = 1e-5
        token_num = 5
        head_num = 2
        head_dim = 128
        hidden_size = head_num * head_dim
        dtype = torch.bfloat16

        inputs = _fused_ag_scale_quant_team_inputs(world_size, token_num, head_num, head_dim, dtype)
        quant_scale = torch.ones(hidden_size, dtype=torch.float32, device="cpu")
        norm_weight = (
            torch.ones(head_dim, dtype=torch.float32, device="cpu")
            if norm_mode == "rmsnorm"
            else None
        )

        input = _to_dev(inputs[rank].contiguous())
        quant_scale = _to_dev(quant_scale)
        norm_weight = _to_dev(norm_weight) if norm_weight is not None else None

        op = MojoFusedAGScaleQuant(
            team_size=world_size,
            norm_mode=norm_mode,
            eps=eps,
            max_tokens=token_num,
        )
        op_ref = MojoFusedAGScaleQuant._registry.get("torch")(
            team_size=world_size,
            norm_mode=norm_mode,
            eps=eps,
            max_tokens=token_num,
        )

        scale_atol = 1e-6 if norm_mode == "none" else 5e-4
        scale_rtol = 1e-6 if norm_mode == "none" else 5e-3
        op.forward_diff_with(
            op_ref,
            input,
            quant_scale,
            norm_weight,
            atol=(1, scale_atol),
            rtol=(0, scale_rtol),
        )
    finally:
        _synchronize_device()
        _destroy_pg()


@pytest.mark.parametrize("team_size", [1, 2, 4, 8])
@pytest.mark.parametrize("norm_mode", ["none", "rmsnorm"])
@bypass_not_implemented
def test_fused_ag_scale_quant(team_size, norm_mode):
    if _device_count() < team_size:
        raise NotImplementedError(f"{team_size} {DEVICE} devices are required for this test")

    if _is_dist_env():
        _dist_fused_ag_scale_quant(norm_mode)
    else:
        _run_torchrun_test(
            "_dist_fused_ag_scale_quant",
            norm_mode,
            nproc=team_size,
        )
