import os
from mojo_opset import MojoRoPE, MojoNorm, MojoSwiGLU, MojoPagedPrefillGQA, MojoPagedDecodeGQA


import torch
import triton
import math
from typing import Optional
import triton.language as tl
import torch.nn as nn


def apply_mojo_op_to_qwen3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    model=None,
) -> None:
    """
    Apply mojo op to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from example_models import mojo_qwen3_dense

    if rope:
        mojo_qwen3_dense.apply_rotary_pos_emb = MojoRoPE()

    if rms_norm:
        mojo_qwen3_dense.Qwen3RMSNorm = MojoNorm

    if swiglu:

        class MojoSwiGLUMLP(nn.Module):
            def __init__(self, config):
                super().__init__()

                self.config = config
                self.hidden_size = config.hidden_size
                self.intermediate_size = config.intermediate_size

                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

                if config.hidden_act != "silu":
                    raise ValueError(f"MojoSwiGLUMLP requires 'silu' activation, but got {config.hidden_act}")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate_output = self.gate_proj(x)
                up_output = self.up_proj(x)

                silu = MojoSwiGLU()
                fused_output = silu(gate_output, up_output)

                return self.down_proj(fused_output)

        mojo_qwen3_dense.Qwen3MLP = MojoSwiGLUMLP

    if attn:
        mojo_qwen3_dense.paged_attention_prefill = MojoPagedPrefillGQA()
        mojo_qwen3_dense.paged_attention_decode = MojoPagedDecodeGQA()

    # NOTE: Currently, only a native decoder layer is implemented as a patch example; the
    # full model is not defined yet, so only static replacement before model instantiation is supported.
    # The following dynamic replacement code is temporarily commented out.

    # if model is not None:
    #     # The model instance already exists, so we need to additionally patch the
    #     # instance variables that reference already-instantiated modules

    #     # get the base model from the model instance
    #     base_model: Qwen3Model = getattr(model, model.base_model_prefix, model)

    #     if rms_norm:
    #         _patch_rms_norm_module(base_model.norm)
    #     for decoder_layer in base_model.layers:
    #         if swiglu:
    #             _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
    #         if rms_norm:
    #             _patch_rms_norm_module(decoder_layer.input_layernorm)
    #             _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
