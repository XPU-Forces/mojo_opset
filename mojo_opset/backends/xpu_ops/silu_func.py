from mojo_opset.core import MojoSiluFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class XOpsSiluFunction(MojoSiluFunction):
    """
    Noticed: the reason why default_priority as attribute instead of a parameter to metaclass
             is autograd.function always be used by calling it's static method forward and backward
             without creating any instance object, so we can't pass default_priority as parameter
             to metaclass.
    """

    default_priority = 2

    @staticmethod
    def forward(ctx, input):
        logger.info("XopsSiluFunction forward impl")
        return input

    @staticmethod
    def backward(ctx, grad_output):
        logger.info("XopsSiluFunction backward impl")
        return grad_output
