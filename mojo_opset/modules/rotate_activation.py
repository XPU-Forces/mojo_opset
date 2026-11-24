import torch

from mojo_opset import functions


class RotateActivation(torch.nn.Module):
    def __init__(self, *, implementation=None):
        super().__init__()
        self.implementation = implementation

    def forward(self, x):
        return functions.rotate_activation(x, implementation=self.implementation)
