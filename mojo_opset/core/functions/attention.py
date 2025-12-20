from ..mojo_function import MojoFuncBase


class MojoBlockDiffusionAttentionFunction(MojoFuncBase):
    @staticmethod
    def forward_ref(ctx, query, key, value, attn_mask, softmax_scale=None):
        pass

    def backward_ref(ctx, grad_output):
        pass
