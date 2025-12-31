from ..function import MojoFunction


class MojoRoPEFunction(MojoFunction):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        pass

    @staticmethod
    def backward(ctx, grad_output_q, grad_output_k):
        pass


def mojo_rope(q, k, cos, sin):
    return MojoRoPEFunction.apply(q, k, cos, sin)
