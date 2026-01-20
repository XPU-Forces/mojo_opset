from torch.autograd import Function

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoFunction(Function):
    def __init__(self, backend: str = ""):
        super().__init__()
        self.backend = backend

    @staticmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, *arg, **kwargs):
        pass
