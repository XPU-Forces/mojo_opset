from typing import Optional

import torch

from ..mojo_function import MojoFuncBase
from ..mojo_function import mojo_func_dispatcher


@mojo_func_dispatcher
class MojoGatedDeltaRuleFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        pass

    @staticmethod
    def forward_ref(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        ctx.save_for_backward(q, k, v, g, beta)

        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        chunk_size = 16
        if use_qk_l2norm_in_kernel:
            q = q / q.norm(dim=-1, keepdim=True)
            k = k / k.norm(dim=-1, keepdim=True)

        _, T, H, K, V = *q.shape, v.shape[-1]
        Hk = k.shape[2]
        if scale is None:
            scale = K**-0.5

        final_states = torch.zeros(len(cu_seqlens) - 1, Hk, K, V, dtype=torch.float32, device="npu")
        o = torch.zeros(1, T, H, V, dtype=torch.float16, device="npu")

        for batch in range(len(cu_seqlens) - 1):
            seq_start = cu_seqlens[batch]
            seq_end = cu_seqlens[batch + 1]
            prev_state = torch.zeros(Hk, K, V, dtype=torch.float32, device="npu")
            for chunk_i in range(seq_start, seq_end, chunk_size):
                chunk_end = min(chunk_i + chunk_size, seq_end)
                b_q = q[0, chunk_i:chunk_end, :, :].transpose(0, 1)
                b_q_g = b_q.view(Hk, H // Hk, b_q.shape[-2], b_q.shape[-1])
                b_k = k[0, chunk_i:chunk_end, :, :].transpose(0, 1)
                b_v = v[0, chunk_i:chunk_end, :, :].transpose(0, 1)
                b_g = g[0, chunk_i:chunk_end, :].transpose(0, 1)
                b_beta = beta[0, chunk_i:chunk_end, :].transpose(0, 1)
                identity = (
                    torch.eye(chunk_end - chunk_i, device=k.device, dtype=torch.float32).unsqueeze(0).expand(Hk, -1, -1)
                )
                b_gamma = b_g.cumsum(dim=-1, dtype=torch.float32)
                b_gamma_diff = torch.tril((b_gamma[:, :, None] - b_gamma[:, None, :]).exp(), diagonal=0)
                b_A = identity + b_gamma_diff * torch.tril(
                    torch.einsum("hcK, hCK->hcC", b_k * b_beta[:, :, None], b_k),
                    diagonal=-1,
                )
                b_A_inv = torch.linalg.inv(b_A)
                b_u = torch.einsum(
                    "hcC, hCV -> hcV",
                    b_A_inv.to(b_k.dtype),
                    (
                        b_v * b_beta[:, :, None]
                        - torch.einsum(
                            "hcK, hKV->hcV",
                            (b_k * (b_beta * b_gamma.exp())[:, :, None]).to(b_k.dtype),
                            prev_state.to(b_k.dtype),
                        )
                    ).to(b_k.dtype),
                )

                b_o = torch.einsum(
                    "hgcK, hKV->hgcV",
                    (b_q_g * scale * b_gamma.exp()[:, None, :, None]).to(b_q.dtype),
                    prev_state.to(b_q.dtype),
                )
                b_o += torch.einsum(
                    "hgcC, hCV -> hgcV",
                    b_gamma_diff.to(b_q.dtype)[:, None, :, :]
                    * torch.tril(
                        torch.einsum("hgcK, hCK->hgcC", (b_q_g * scale).to(b_q.dtype), b_k),
                        diagonal=0,
                    ),
                    b_u,
                )
                prev_state = b_gamma[:, -1].exp()[:, None, None] * prev_state.to(b_q.dtype) + torch.einsum(
                    "hcK, hcV->hKV",
                    ((b_gamma[:, -1][:, None] - b_gamma).exp()[:, :, None] * b_k).to(b_q.dtype),
                    b_u,
                )
                o[0, chunk_i:chunk_end, :, :] = b_o.reshape(H, chunk_end - chunk_i, V).transpose(0, 1)

            final_states[batch] = prev_state

        return o

    @staticmethod
    def backward_dump(ctx, do: torch.Tensor, dht: torch.Tensor):
        pass

    @staticmethod
    def backward_ref(ctx, do: torch.Tensor):
        q, k, v, g, beta = ctx.saved_tensors
        cu_seqlens = ctx.cu_seqlens
        scale = ctx.scale
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel

        q_with_grad = q.detach().clone().requires_grad_(True)
        k_with_grad = k.detach().clone().requires_grad_(True)
        v_with_grad = v.detach().clone().requires_grad_(True)
        g_with_grad = g.detach().clone().requires_grad_(True)
        beta_with_grad = beta.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            o_ref = MojoGatedDeltaRuleFunction.forward_ref(
                ctx,
                q_with_grad,
                k_with_grad,
                v_with_grad,
                g_with_grad,
                beta_with_grad,
                cu_seqlens,
                scale,
                use_qk_l2norm_in_kernel,
            )

        o_ref.backward(gradient=do)

        grad_q = q_with_grad.grad
        grad_k = k_with_grad.grad
        grad_v = v_with_grad.grad
        grad_g = g_with_grad.grad
        grad_beta = beta_with_grad.grad

        return grad_q, grad_k, grad_v, grad_g, grad_beta, None, None, None


def mojo_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    enable_gqa: bool = True,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    assert head_first == False and output_final_state == False, (
        "mojo_chunk_gated_delta_rule does not support head_first=True or output_final_state=True."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    head_dim = 1 if head_first else 2
    num_q_heads = q.shape[head_dim]
    num_kv_heads = k.shape[head_dim]
    num_gbeta_heads = g.shape[head_dim]

    if (not enable_gqa) and (num_q_heads > num_kv_heads):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Number of query heads ({num_q_heads}) must be divisible by "
                f"number of key/value heads ({num_kv_heads}) for GQA"
            )

        num_repeats = num_q_heads // num_kv_heads

        k = k.repeat_interleave(num_repeats, dim=2)
        v = v.repeat_interleave(num_repeats, dim=2)

    if (not enable_gqa) and (num_q_heads > num_gbeta_heads):
        if num_q_heads % num_gbeta_heads != 0:
            raise ValueError(
                f"Number of query heads ({num_q_heads}) must be divisible by "
                f"number of beta/gate heads ({num_gbeta_heads}) for GQA"
            )

        num_repeats = num_q_heads // num_gbeta_heads
        g = g.repeat_interleave(num_repeats, dim=2)
        beta = beta.repeat_interleave(num_repeats, dim=2)

    o = MojoGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
    )
    return o
