import torch
import torch.nn as nn

from ..function import MojoFunction


class MojoRMSNormFunction(MojoFunction):
    @staticmethod
    def forward(ctx, input, weight, eps):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class MojoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return MojoRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )
