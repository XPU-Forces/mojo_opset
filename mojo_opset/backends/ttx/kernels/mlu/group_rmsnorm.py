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

@triton.jit
def _rmsnorm_fwd_impl(
    x_ptr,  # x base ptr, shape [T, H, N]
    w_ptr,  # weight ptr, shape [N]
    y_ptr,  # y base ptr, shape [T, H, N]
    T,  # token 数
    H,  # num_head
    N,  # norm_size
    stride_xt,  # x stride for token dim
    stride_xh,  # x stride for head dim
    stride_xn,  # x stride for norm dim，要求=1
    stride_yt,  # y stride for token dim
    stride_yh,  # y stride for head dim
    stride_yn,  # y stride for norm dim，要求=1
    eps,
    BLOCK_N: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    M = T * H

    row = pid
    while row < M:
        token_id = row // H
        head_id = row % H

        # x[token_id, head_id, :]
        #   = x_ptr + token_id * stride_xt + head_id * stride_xh + n * stride_xn
        #
        x_row_base = token_id * stride_xt + head_id * stride_xh
        y_row_base = token_id * stride_yt + head_id * stride_yh

        acc = 0.0
        start_n = 0
        while start_n < N:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            x = tl.load(
                x_ptr + x_row_base + offs_n * stride_xn,
                mask=mask,
                other=0.0,
            )
            x = x.to(tl.float32)
            acc += tl.sum(x * x, axis=0)
            start_n += BLOCK_N
        mean_sq = acc / N
        rstd = tl.rsqrt(mean_sq + eps)

        start_n = 0
        while start_n < N:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            x = tl.load(
                x_ptr + x_row_base + offs_n * stride_xn,
                mask=mask,
                other=0.0,
            )
            x_fp32 = x.to(tl.float32)
            if HAS_WEIGHT:
                w = tl.load(
                    w_ptr + offs_n,
                    mask=mask,
                    other=1.0,
                )
                w_fp32 = w.to(tl.float32)
                y_fp32 = x_fp32 * rstd * w_fp32
            else:
                y_fp32 = x_fp32 * rstd
            tl.store(
                y_ptr + y_row_base + offs_n * stride_yn,
                y_fp32,
                mask=mask,
            )
            start_n += BLOCK_N
        row += num_pid

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
    if weight is not None:
        assert weight.dtype == x.dtype
        assert weight.shape == (N,)
        assert weight.is_contiguous()
        has_weight = True
    else:
        has_weight = False
    if output_like_input_stride:
        y = torch.empty_strided(
            size=x.shape,
            stride=x.stride(),
            dtype=x.dtype,
            device=x.device,
        )
    else:
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
    assert y.stride(2) == 1, f"y last dim must be contiguous, got stride={y.stride()}"

    block_n = min(triton.next_power_of_2(N), 1024)
    grid_size = get_mlu_total_cores()
    total_rows = T * H
    grid = (min(grid_size, total_rows),)
    _rmsnorm_fwd_impl[grid](
        x,
        weight,
        y,
        T,
        H,
        N,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        eps,
        BLOCK_N=block_n,
        HAS_WEIGHT=has_weight,
    )
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
