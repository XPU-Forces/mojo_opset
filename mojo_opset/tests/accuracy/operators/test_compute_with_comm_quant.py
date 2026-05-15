import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset import MojoAll2AllQuantGemm
from mojo_opset import MojoQuantGemmAll2All


torch.manual_seed(42)


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
