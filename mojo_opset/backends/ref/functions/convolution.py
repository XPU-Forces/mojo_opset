from mojo_opset.core import MojoCausalConv1dFunction


class RefCausalConv1dFunction(MojoCausalConv1dFunction):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        initial_state=None,
        output_final_state=False,
        final_states_out=None,
        activation=None,
        cu_seqlens=None,
    ):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass
