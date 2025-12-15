import pytest
import torch
import torch.nn.functional as F

from einops import rearrange
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoCausalConv1dFunction
from mojo_opset.backends.ttx.kernels.ascend.utils import input_guard


@input_guard(make_contiguous=True, auto_to_device=True)
def mojo_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    backend: str | None = "triton",
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'` and `'triton'`.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)

    Returns:
        Tuple of (output, final_state).
        If `output_final_state` is `False`, the final state is `None`.
    """

    y, final_state = MojoCausalConv1dFunction.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
    )
    return y, final_state


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_state=None,
    output_final_state=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_state: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_state is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if output_final_state:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in,
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not output_final_state else (out, final_states_out)


@pytest.mark.parametrize(
    ("B", "T", "D", "W", "activation", "has_bias", "has_residual", "dtype", "dummy_tensor"),
    [
        pytest.param(
            *test,
            torch.randn(1),
            id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test),
        )
        for test in [
            (2, 64, 128, 3, "swish", True, True, torch.float32),
            (2, 128, 128, 4, "swish", False, True, torch.float32),
            (2, 64, 128, 3, "swish", True, False, torch.float32),
            (2, 128, 128, 4, "swish", False, False, torch.float32),
            (2, 500, 1024, 3, None, True, True, torch.float32),
            (2, 1024, 1024, 4, None, False, True, torch.float32),
            (2, 64, 128, 3, None, True, False, torch.float16),
            (2, 128, 128, 4, None, False, False, torch.float16),
        ]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_conv(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    dummy_tensor: torch.Tensor,
):
    device = dummy_tensor.device
    print(f"device is {device}")
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(B, T, D).to(device, dtype)

    ref = causal_conv1d_ref(
        x=rearrange(x, "b t d -> b d t"),
        weight=weight,
        bias=bias,
        activation=activation,
    )

    ref = rearrange(ref, "b d t -> b t d")
    if has_residual:
        ref += residual
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, _ = mojo_causal_conv1d(x, weight, bias, residual=residual, activation=activation)
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None
    torch.allclose(ref, tri, atol=1e-3, rtol=1e-3)
    torch.allclose(ref_dx, tri_dx, atol=1e-3, rtol=1e-3)
    torch.allclose(ref_dw, tri_dw, atol=1e-3, rtol=1e-3)
    if has_bias:
        torch.allclose(ref_db, tri_db, atol=1e-3, rtol=1e-3)
    if has_residual:
        torch.allclose(ref_dr, tri_dr, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    ("N", "T", "D", "W", "activation", "has_bias", "has_residual", "dtype", "dummy_tensor"),
    [
        pytest.param(
            *test, torch.randn(1), id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test)
        )
        for test in [
            (4, 500, 128, 3, "silu", True, True, torch.float32),
            (4, 1024, 200, 4, "silu", False, True, torch.float32),
            (4, 500, 128, 3, None, True, False, torch.float16),
            (4, 1024, 1024, 4, None, False, False, torch.float16),
        ]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    dummy_tensor: torch.Tensor,
):
    torch.manual_seed(42)
    device = dummy_tensor.device
    cu_seqlens = (
        torch.cat(
            [
                torch.tensor([0], dtype=torch.long),
                torch.arange(16, T)[torch.randperm(T - 16)[: N - 1]],
                torch.tensor([T], dtype=torch.long),
            ],
            0,
        )
        .to(device)
        .sort()[0]
    )

    x = torch.randn(1, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(1, T, D).to(device, dtype)

    ref = torch.cat(
        [
            rearrange(
                causal_conv1d_ref(
                    x=rearrange(x[:, bos:eos].contiguous(), "b t d -> b d t"),
                    weight=weight,
                    bias=bias,
                    activation=activation,
                ),
                "b t d -> b d t",
            )
            + (residual[:, bos:eos] if has_residual else torch.zeros_like(x[:, bos:eos]))
            for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
        ],
        1,
    )
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, _ = mojo_causal_conv1d(x, weight, bias, residual=residual, activation=activation, cu_seqlens=cu_seqlens)
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None

    torch.allclose(ref, tri, atol=1e-3, rtol=1e-3)
    torch.allclose(ref_dx, tri_dx, atol=1e-3, rtol=1e-3)
    torch.allclose(ref_dw, tri_dw, atol=1e-3, rtol=1e-3)
    if has_bias:
        torch.allclose(ref_db, tri_db, atol=1e-3, rtol=1e-3)
    if has_residual:
        torch.allclose(ref_dr, tri_dr, atol=1e-3, rtol=1e-3)
