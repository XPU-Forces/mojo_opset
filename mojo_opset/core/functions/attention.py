from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoDiffusionAttentionFunction(MojoFunction):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
        pass

    @staticmethod
    def backward(ctx, do):
        pass


def mojo_diffusion_attention(query, key, value, mask, scale=1.0, enable_gqa=False):
    return MojoDiffusionAttentionFunction.apply(query, key, value, mask, scale, enable_gqa)
