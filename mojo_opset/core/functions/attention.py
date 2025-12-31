from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoSdpaFunction(MojoFunction):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
        pass

    @staticmethod
    def backward(ctx, do):
        pass

def mojo_sdpa(query, key, value, mask, scale=1.0, enable_gqa=False):
    return MojoSdpaFunction.apply(query, key, value, mask, scale, enable_gqa)