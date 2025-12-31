from ..function import MojoFunction


class MojoSiluFunction(MojoFunction):
    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


def mojo_silu(input):
    return MojoSiluFunction.apply(input)
