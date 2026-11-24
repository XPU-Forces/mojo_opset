"""Reference SDPA formulas, independent of hardware."""

import torch


def sdpa_infer_fwd(query, key, value, attn_mask, scale, enable_gqa):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=enable_gqa
    )


def diffusion_attention_fwd(query, key, value, mask, scale, enable_gqa):
    output = sdpa_infer_fwd(query, key, value, mask, scale, enable_gqa)
    # Reference backward recomputes the graph; these two fields are only saved metadata.
    return output, output.float(), query.new_empty(query.shape[:-1], dtype=torch.float32)


def diffusion_attention_bwd(output_fp32, grad_output, query, key, value, lse, mask, scale, enable_gqa):
    with torch.enable_grad():
        inputs = [x.detach().requires_grad_(True) for x in (query, key, value)]
        output = sdpa_infer_fwd(*inputs, mask, scale, enable_gqa)
        return torch.autograd.grad(output, inputs, grad_output)
