import torch

import triton
import triton.language as tl
from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores

def cfggen1():
    block_m = [1, 2, 4, 6, 8]
    num_stages = [1, 3]
    configs = [
        triton.Config({
            "BLOCK_M": m,
        }, num_stages=s) for m in block_m for s in num_stages
    ]
    return configs


@triton.autotune(configs=cfggen1(), key=["M", "N"])
@triton.heuristics({
    'BLOCK_N': lambda args: args['N'],
    'num_warps': lambda args: 1,
})
@triton.jit(do_not_specialize=["eps"])
def _layer_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)

    n_offset = tl.arange(0, BLOCK_N)
    gamma = tl.load(W + n_offset)
    beta = tl.load(B + n_offset)

    for m_idx in range(pid_m * BLOCK_M, M, pnum * BLOCK_M):
        m_offset = m_idx + tl.arange(0, BLOCK_M)
        mask = m_offset < M
        
        offset = m_offset[:, None] * N + n_offset[None, :]
        inp = tl.load(X + offset, mask=mask[:, None], other=0.0).to(tl.float32)

        alpha = 1.0 / N
        sum = tl.sum(inp, 1)
        mean = sum * alpha
        var = tl.sum(inp * inp, 1) * alpha - mean * mean

        rstd = tl.rsqrt(var + eps)
        tl.store(Mean + m_offset, mean, mask=mask)
        tl.store(Rstd + m_offset, rstd, mask=mask)

        out = (inp - mean[:, None]) * rstd[:, None]
        opt_out = out * gamma + beta
        tl.store(Y + offset, opt_out, mask=mask[:, None])

def cfggen2():
    block_m = [1, 2, 3, 4, 5, 6, 8]
    num_stages = [1, 3]
    configs = [
        triton.Config({
            "BLOCK_M": m,
        }, num_stages=s) for m in block_m for s in num_stages
    ]
    return configs


@triton.autotune(configs=cfggen2(), key=["M", "N"], reset_to_zero=["DW", "DB"])
@triton.heuristics({
    'BLOCK_N': lambda args: args['N'],
    'num_warps': lambda args: 1,
})
@triton.jit
def _layer_norm_bwd(
        DX,  # pointer to the input gradient
        DY,  # pointer to the output gradient
        DW,  # pointer to the partial sum of weights gradient
        DB,  # pointer to the partial sum of biases gradient
        X,  # pointer to the input
        W,  # pointer to the weights
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        M,  # number of rows in X
        N,  # number of columns in X
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)

    row_start = pid * BLOCK_M
    cols = tl.arange(0, BLOCK_N)
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_M

    X += cols[None, :]
    DY += cols[None, :]
    W += cols[None, :]
    DX += cols[None, :]
    w = tl.load(W, mask=cols[None, :] < N, other=0.0).to(tl.float32)
    alpha = 1 / N
    partial_dw = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    partial_db = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_M)
        mask = row_off[:, None] < M
        # Load data to SRAM
        off = row_off[:, None] * N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row_off, mask=row_off < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + row_off, mask=row_off < M)[:, None].to(tl.float32)
        # Compute dx
        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=1)[:, None]
        c2 = tl.sum(wdy, axis=1)[:, None]
        dx = (wdy - (x_hat * c1 + c2) * alpha) * rstd

        # Accumulate partial sums for dw/db
        partial_dw += (dy * x_hat).to(tl.float32)
        partial_db += (dy).to(tl.float32)
        # Write dx
        tl.store(DX + off, dx.to(x.dtype), mask=mask)

    dw = tl.sum(partial_dw, axis=0)
    db = tl.sum(partial_db, axis=0)
    tl.atomic_add(DW + cols, dw, mask=cols < N)
    tl.atomic_add(DB + cols, db, mask=cols < N)

def layernorm_infer_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # allocate output
    y = torch.empty_like(hidden_states)
    # reshape input data into 2D tensor
    x_arg = hidden_states.reshape(-1, hidden_states.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device='mlu')
    rstd = torch.empty((M, ), dtype=torch.float32, device='mlu')
    num_warps = 1
    # enqueue kernel
    _layer_norm_fwd[(get_mlu_total_cores(), )](x_arg,
                                        y,
                                        weight,
                                        bias,
                                        mean,
                                        rstd,
                                        M,
                                        N,
                                        eps,
                                        opt_level="Om")
    return y

def layernorm_fwd_impl(x, w, b, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device='mlu')
    rstd = torch.empty((M, ), dtype=torch.float32, device='mlu')
    num_warps = 1
    # enqueue kernel
    _layer_norm_fwd[(get_mlu_total_cores(), )](x_arg,
                                        y,
                                        w,
                                        b,
                                        mean,
                                        rstd,
                                        M,
                                        N,
                                        eps,
                                        opt_level="Om")
    return y, x, mean, rstd

def layernorm_bwd_impl(dy, x, w, b, m, v):
    N = w.shape[0]
    # allocate output
    dw = torch.zeros((w.shape[0], ), dtype=w.dtype, device=w.device)
    db = torch.zeros((w.shape[0], ), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    # enqueue kernel using forward pass heuristics
    # also compute partial sums for DW and DB
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    _layer_norm_bwd[(get_mlu_total_cores(), )](dx,
                                        dy,
                                        dw,
                                        db,
                                        x,
                                        w,
                                        m,
                                        v,
                                        M,
                                        N,
                                        opt_level="Om")
    return dx, dw, db
