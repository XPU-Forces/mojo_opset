import torch

import triton
import triton.language as tl
from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores

def check_supported_dtype(dtype: torch.dtype) -> None:
    assert dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"dtype must be one of [torch.float16, torch.bfloat16, torch.float32], got {dtype}"


def cfggen():
    block_m = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    num_stages = [4]
    configs = [
        triton.Config({"BLOCK_M": m}, num_stages=s)
        for m in block_m
        for s in num_stages
    ]
    return configs


@triton.autotune(configs=cfggen(), key=["T", "H", "N"])
@triton.jit(do_not_specialize=["eps"])
def _rmsnorm_fwd_impl(
    x_ptr,  # x base ptr, shape [T, H, N]
    w_ptr,  # weight ptr, shape [N]
    y_ptr,  # y base ptr, shape [T, H, N]
    T,  # token 数
    H,  # num_head
    N,  # norm_size
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    M = T * H
    ones = tl.full((1, BLOCK_N), 1.0, dtype=tl.float32)

    for row_start in range(pid * BLOCK_M, M, num_pid * BLOCK_M):
        offs_m = row_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        addr_x = x_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask_2d = mask_m[:, None] & mask_n[None, :]
        x_f32 = tl.load(addr_x, mask=mask_2d, other=0.0).to(tl.float32)

        if USE_DOT:
            sq = x_f32 * x_f32
            acc = tl.reshape(tl.dot(ones, tl.trans(sq), allow_tf32=False), (BLOCK_M,))
        else:
            acc = tl.sum(x_f32 * x_f32, axis=1)
        mean_sq = acc / N
        rstd = tl.rsqrt(mean_sq + eps)

        if HAS_WEIGHT:
            w = tl.load(w_ptr + offs_n, mask=mask_n, other=1.0)
            w_f32 = w.to(tl.float32)
            if USE_DOT:
                y_f32 = tl.dot(tl.reshape(rstd, (BLOCK_M, 1)), ones, allow_tf32=False) * x_f32 * w_f32[None, :]
            else:
                y_f32 = x_f32 * rstd[:, None] * w_f32[None, :]
        else:
            if USE_DOT:
                y_f32 = tl.dot(tl.reshape(rstd, (BLOCK_M, 1)), ones, allow_tf32=False) * x_f32
            else:
                y_f32 = x_f32 * rstd[:, None]

        addr_y = y_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(addr_y, y_f32, mask=mask_2d)


def rmsnorm_fwd_triton(
    x,
    weight=None,
    eps=1e-6,
    output_like_input_stride=True,
):
    check_supported_dtype(x.dtype)
    assert x.ndim == 3, f"x must be 3D [token, num_head, norm_size], got {x.shape}"
    T, H, N = x.shape
    assert x.stride(2) == 1, f"x last dim must be contiguous, got stride={x.stride()}"
    assert N <= 8192, f"x last size must be <= 8192, got token={x.size(-1)}"
    if weight is not None:
        assert weight.dtype == x.dtype
        assert weight.shape == (N,)
        assert weight.is_contiguous()
        has_weight = True
    else:
        has_weight = False
    
    orig_stride = x.stride()
    orig_is_contiguous = x.is_contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    block_n = min(triton.next_power_of_2(N), 8192)

    grid = lambda meta: (
        min(get_mlu_total_cores(), triton.cdiv(T * H, meta['BLOCK_M'])),
    )
    if N > 1024:
        use_dot = False
    else:
        use_dot = True

    _rmsnorm_fwd_impl[grid](
        x,
        weight,
        y,
        T,
        H,
        N,
        eps,
        BLOCK_N=block_n,
        HAS_WEIGHT=has_weight,
        USE_DOT=use_dot,
    )

    if output_like_input_stride and not orig_is_contiguous:
        y_strided = torch.empty_strided(
            size=y.shape,
            stride=orig_stride,
            dtype=y.dtype,
            device=y.device,
        )
        y_strided.copy_(y)
        return y_strided
    else:
        return y

def group_rmsnorm_impl(
    input_groups,
    weight=None,
    eps=1e-6,
    output_like_input_stride=True,
) -> list[torch.tensor]:
    """
    Apply grouped RMSNorm on a list of input tensors.

    This interface performs RMSNorm independently on each tensor in ``input_groups``.
    Every input tensor is expected to have shape ``[token, num_head, norm_size]``,
    and normalization is applied along the last dimension only.

    For each group:
        - Each logical row corresponds to one ``[norm_size]`` slice.
        - RMS statistics are computed independently for every ``[token, num_head]`` location.
        - If ``weight`` is provided, a per-group affine scale is applied after normalization.

    Args:
        - input_groups (list[Tensor]): Grouped input tensors.
            - Shape of each tensor: [token, num_head, norm_size]
            - Supported dtypes: float16 / bfloat16 / float32
            - The last dimension must be contiguous
            - The first two dimensions may be non-contiguous
            - Different groups may have different ``token`` / ``num_head`` sizes
        - weight (Tensor or None): Optional affine scale tensor.
            - If provided:
                - Shape: [num_groups, norm_size]
                - Data type must match the input tensors
                - ``weight[g]`` is applied to ``input_groups[g]``
            - If None:
                - normalization is performed without affine scaling
        - eps (float): Numerical stability term added inside RMS computation.
            - Default: 1e-6
        - output_like_input_stride (bool): Whether each output tensor should preserve
          the same stride layout as its corresponding input tensor.
            - If True: preserve input stride layout
            - If False: allocate output in contiguous layout

    Returns:
        - output_groups (list[Tensor]): Normalized output tensors.
            - Same number of groups as ``input_groups``
            - Each output tensor has the same shape and dtype as its input tensor
            - The last dimension remains contiguous

    Notes:
        - All groups must share the same ``norm_size``.
        - The operation is computed in float32 internally for numerical stability,
          while outputs are written back using the original input dtype.
        - The operator assumes normalization is always performed along the last dimension.
    """
    assert isinstance(input_groups, (list, tuple))
    assert len(input_groups) > 0

    G = len(input_groups)
    N = input_groups[0].shape[-1]

    if weight is not None:
        assert weight.shape == (G, N)
        assert (
            weight.is_contiguous()
        ), f"weight must be contiguous, got stride={weight.stride()}"

    output_groups = []
    for g in range(G):
        xg = input_groups[g]
        assert xg.ndim == 3, f"group {g} input must be [token, num_head, norm_size]"
        assert xg.shape[-1] == N, f"group {g} last dim mismatch: {xg.shape[-1]} vs {N}"
        wg = None if weight is None else weight[g]
        if N != 0:
            yg = rmsnorm_fwd_triton(
                x=xg,
                weight=wg,
                eps=eps,
                output_like_input_stride=output_like_input_stride,
            )
        else:
            if output_like_input_stride:
                yg = torch.empty_strided(
                    size=xg.shape,
                    stride=xg.stride(),
                    dtype=xg.dtype,
                    device=xg.device,
                )
            else:
                yg = torch.empty_like(xg, memory_format=torch.contiguous_format)
        output_groups.append(yg)

    return output_groups
