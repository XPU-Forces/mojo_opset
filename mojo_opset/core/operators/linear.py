import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoLinear(MojoOperator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **self.tensor_factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **self.tensor_factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class MojoQuantLinear(MojoOperator):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_quant_dtype=torch.int8,
        output_dtype=torch.bfloat16,
        symmetric=True,
        group_size=-1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dtype = output_dtype
        if weight_quant_dtype != torch.int8 or not symmetric:
            raise NotImplementedError("Only Symmetric W8A8 linear is supported for now")
        if group_size > 0:
            raise NotImplementedError("Per-Group quantization is not supported yet")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=weight_quant_dtype),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **self.tensor_factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "weight_qscale",
            torch.ones((out_features), dtype=torch.bfloat16, requires_grad=False),
        )

    def forward(self, input: torch.Tensor, input_qscale: torch.Tensor):
        out = torch.matmul(input.float(), self.weight.T.float()) * input_qscale.float() * self.weight_qscale.float()
        if self.bias is not None:
            out += self.bias
        return out.to(self.output_dtype)

