"""Weight-owning inference GEMMs."""

import torch

from mojo_opset import functions


class GroupGemm(torch.nn.Module):
    def __init__(self, weight, trans_weight=False, *, implementation=None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.trans_weight = trans_weight
        self.implementation = implementation

    def forward(self, input, group_list):
        return functions.group_gemm(
            input, self.weight, group_list, trans_weight=self.trans_weight, implementation=self.implementation
        )


class QuantGemm(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        output_dtype=torch.bfloat16,
        trans_weight=False,
        *,
        implementation=None,
        device=None,
    ):
        super().__init__()
        shape = (out_features, in_features) if trans_weight else (in_features, out_features)
        self.register_buffer("weight", torch.empty(shape, device=device, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, device=device, dtype=torch.bfloat16))
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight
        self.implementation = implementation

    def forward(self, input, input_scale):
        return functions.quant_gemm(
            input,
            self.weight,
            input_scale,
            self.weight_scale,
            output_dtype=self.output_dtype,
            trans_weight=self.trans_weight,
            implementation=self.implementation,
        )


class QuantBatchGemmReduceSum(torch.nn.Module):
    def __init__(self, weight, trans_weight=False, *, implementation=None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.trans_weight = trans_weight
        self.implementation = implementation

    def forward(self, input, x1_scale, x2_scale):
        return functions.quant_batch_gemm_reduce_sum(
            input,
            self.weight,
            x1_scale,
            x2_scale,
            trans_weight=self.trans_weight,
            implementation=self.implementation,
        )
