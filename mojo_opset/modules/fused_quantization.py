"""State-owning inference quantizers; parameters are stored as non-trainable buffers."""

import torch

from mojo_opset.functions import fused_quantization as F


class RMSNormQuant(torch.nn.Module):
    def __init__(
        self,
        norm_size,
        eps=1e-5,
        quant_dtype=torch.int8,
        symmetric=True,
        *,
        implementation=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        F._check_quant_dtype(quant_dtype)
        self.register_buffer("weight", torch.ones(norm_size, device=device, dtype=dtype))
        self.eps, self.quant_dtype, self.symmetric = eps, quant_dtype, symmetric
        self.implementation = implementation

    def forward(self, x, smooth_scale=None):
        return F.rms_norm_quant(
            x,
            self.weight,
            smooth_scale,
            eps=self.eps,
            quant_dtype=self.quant_dtype,
            symmetric=self.symmetric,
            implementation=self.implementation,
        )


class LayerNormQuant(torch.nn.Module):
    def __init__(
        self,
        norm_size,
        eps=1e-5,
        elementwise_affine=True,
        quant_dtype=torch.int8,
        symmetric=True,
        *,
        implementation=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        F._check_quant_dtype(quant_dtype)
        self.register_buffer(
            "weight", torch.ones(norm_size, device=device, dtype=dtype) if elementwise_affine else None
        )
        self.register_buffer("bias", torch.zeros(norm_size, device=device, dtype=dtype) if elementwise_affine else None)
        self.eps, self.quant_dtype, self.symmetric = eps, quant_dtype, symmetric
        self.implementation = implementation

    def forward(self, x, smooth_scale=None):
        return F.layer_norm_quant(
            x,
            self.weight,
            self.bias,
            smooth_scale,
            eps=self.eps,
            quant_dtype=self.quant_dtype,
            symmetric=self.symmetric,
            implementation=self.implementation,
        )


class ResidualAddRMSNormQuant(RMSNormQuant):
    def __init__(
        self,
        norm_size,
        eps=1e-5,
        norm_pos="pre",
        quant_dtype=torch.int8,
        symmetric=True,
        *,
        implementation=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            norm_size, eps, quant_dtype, symmetric, implementation=implementation, device=device, dtype=dtype
        )
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos must be 'pre' or 'post'")
        self.norm_pos = norm_pos

    def forward(self, x, residual, smooth_scale=None):
        return F.residual_add_rms_norm_quant(
            x,
            residual,
            self.weight,
            smooth_scale,
            eps=self.eps,
            norm_pos=self.norm_pos,
            quant_dtype=self.quant_dtype,
            symmetric=self.symmetric,
            implementation=self.implementation,
        )


class ResidualAddLayerNormQuant(LayerNormQuant):
    def __init__(
        self,
        norm_size,
        eps=1e-5,
        elementwise_affine=True,
        norm_pos="pre",
        quant_dtype=torch.int8,
        symmetric=True,
        *,
        implementation=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            norm_size,
            eps,
            elementwise_affine,
            quant_dtype,
            symmetric,
            implementation=implementation,
            device=device,
            dtype=dtype,
        )
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos must be 'pre' or 'post'")
        self.norm_pos = norm_pos

    def forward(self, x, residual, smooth_scale=None):
        return F.residual_add_layer_norm_quant(
            x,
            residual,
            self.weight,
            self.bias,
            smooth_scale,
            eps=self.eps,
            norm_pos=self.norm_pos,
            quant_dtype=self.quant_dtype,
            symmetric=self.symmetric,
            implementation=self.implementation,
        )


class MoEDynamicQuant(torch.nn.Module):
    def __init__(self, expert_num, input_size, quant_dtype=torch.int8, *, implementation=None, device=None, dtype=None):
        super().__init__()
        F._check_quant_dtype(quant_dtype, fp8=False)
        self.register_buffer(
            "inv_smooth_scale", torch.ones((expert_num, input_size), device=device, dtype=torch.float32)
        )
        self.quant_dtype, self.implementation = quant_dtype, implementation

    def forward(self, input, token_count):
        return F.moe_dynamic_quant(
            input, token_count, self.inv_smooth_scale, quant_dtype=self.quant_dtype, implementation=self.implementation
        )


class DequantSwiGLUQuant(torch.nn.Module):
    def __init__(
        self,
        expert_num,
        hidden_size,
        quant_dtype=torch.int8,
        activate_left=False,
        quant_mode=1,
        *,
        implementation=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        F._check_quant_dtype(quant_dtype, fp8=False)
        if quant_mode != 1:
            raise NotImplementedError("only dynamic quant_mode=1 is supported")
        self.register_buffer("weight_scale", torch.ones((expert_num, 2 * hidden_size), device=device, dtype=dtype))
        self.register_buffer("quant_scale", torch.ones((expert_num, hidden_size), device=device, dtype=dtype))
        self.quant_dtype, self.activate_left, self.quant_mode = quant_dtype, activate_left, quant_mode
        self.implementation = implementation

    def forward(self, x, activation_scale=None, bias=None, quant_offset=None, token_count=None):
        return F.dequant_swiglu_quant(
            x,
            self.weight_scale,
            self.quant_scale,
            activation_scale,
            bias,
            quant_offset,
            token_count,
            quant_dtype=self.quant_dtype,
            activate_left=self.activate_left,
            quant_mode=self.quant_mode,
            implementation=self.implementation,
        )
