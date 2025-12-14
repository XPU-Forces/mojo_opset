from ..mojo_function import MojoFuncBase
from ..mojo_function import mojo_func_dispatcher


@mojo_func_dispatcher
class MojoCausalConv1dFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(
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
    def forward_ref(
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
        raise NotImplementedError

    @staticmethod
    def backward_dump(ctx, *grad_outputs):
        pass

    @staticmethod
    def backward_ref(ctx, *grad_outputs):
        raise NotImplementedError
