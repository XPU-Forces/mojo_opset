from mojo_opset.utils.logging import get_logger

from ..function import MojoFunction

logger = get_logger(__name__)


class MojoSiluFunction(MojoFunction):
    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


mojo_silu = MojoSiluFunction.apply
