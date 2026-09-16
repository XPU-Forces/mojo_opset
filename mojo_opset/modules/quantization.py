"""State-owning modules for the original Mojo quantization operations."""

import torch

from mojo_opset import functions


class StaticQuant(torch.nn.Module):
    def __init__(self, input_size, quant_dtype=torch.int8, *, implementation=None, device=None, dtype=None):
        super().__init__()
        shape = (input_size,) if isinstance(input_size, int) else tuple(input_size)
        self.scale = torch.nn.Parameter(torch.ones(shape, device=device, dtype=dtype), requires_grad=False)
        self.quant_dtype = quant_dtype
        self.implementation = implementation

    def forward(self, input):
        return functions.static_quant(input, self.scale, quant_dtype=self.quant_dtype,
                                      implementation=self.implementation)


class Dequant(torch.nn.Module):
    def __init__(self, output_dtype=torch.bfloat16, *, implementation=None):
        super().__init__()
        self.output_dtype = output_dtype
        self.implementation = implementation

    def forward(self, input, scale):
        return functions.dequant(input, scale, output_dtype=self.output_dtype,
                                 implementation=self.implementation)


class DynamicQuant(torch.nn.Module):
    def __init__(self, input_size=None, quant_dtype=torch.int8, *, implementation=None, device=None, dtype=None):
        super().__init__()
        self.inv_smooth_scale = (None if input_size is None else
                                torch.nn.Parameter(torch.ones(input_size, device=device, dtype=torch.float32),
                                                   requires_grad=False))
        self.quant_dtype = quant_dtype
        self.implementation = implementation

    def forward(self, input):
        return functions.dynamic_quant(input, self.inv_smooth_scale, quant_dtype=self.quant_dtype,
                                       implementation=self.implementation)
