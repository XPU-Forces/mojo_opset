import torch

from mojo_opset.backends.ttx.kernels.npu.rmsnorm import rmsnorm_infer_impl


def group_rmsnorm_impl(
    input_groups,
    weight=None,
    eps=1e-6,
    output_like_input_stride=True,
) -> list[torch.Tensor]:
    assert isinstance(input_groups, (list, tuple))
    assert len(input_groups) > 0

    G = len(input_groups)
    N = input_groups[0].shape[-1]

    if weight is not None:
        assert weight.shape == (G, N), (
            f"weight shape must be ({G}, {N}), got {weight.shape}"
        )
        assert weight.is_contiguous(), (
            f"weight must be contiguous, got stride={weight.stride()}"
        )

    output_groups = []
    for g in range(G):
        xg = input_groups[g]
        assert xg.ndim == 3, f"group {g} input must be [token, num_head, norm_size]"
        assert xg.shape[-1] == N, (
            f"group {g} last dim mismatch: {xg.shape[-1]} vs {N}"
        )
        wg = None if weight is None else weight[g]
        if N != 0:
            xg = xg.contiguous()
            yg = rmsnorm_infer_impl(xg, wg, eps)
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
