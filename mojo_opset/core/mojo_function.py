import os
import torch
import functools
from torch.autograd import Function

from ..mojo_utils import get_mojo_exec_mode
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def mojo_func_dispatcher(cls):
    op_name = cls.__name__

    @functools.lru_cache(maxsize=None)
    def _get_execution_function(op_name_in, direction, layer_idx):
        mode_str = get_mojo_exec_mode(op_name_in, direction, layer_idx)

        func_name = "forward" if direction == "FWD" else "backward"
        ref_func_name = f"{func_name}_ref"
        dump_func_name = f"{func_name}_dump"

        impl_func = cls._registry[0][1].__dict__[func_name] if cls._registry else None
        ref_func = getattr(cls, ref_func_name, None)
        dump_func = getattr(cls, dump_func_name, None)

        if mode_str == "STD":
            chosen_func = impl_func if impl_func else ref_func
            if not chosen_func:
                raise NotImplementedError(f"{op_name_in} has no STD/REF {direction} implementation.")
            return chosen_func

        if mode_str == "REF":
            if not ref_func:
                raise NotImplementedError(f"{op_name_in} must implement {ref_func_name}.")
            return ref_func

        if mode_str == "DUMP":
            if not dump_func:
                raise NotImplementedError(f"{op_name_in} must implement {dump_func_name}.")
            return dump_func

        if mode_str == "DIFF":
            if not impl_func or not ref_func:
                raise NotImplementedError(
                    f"{op_name_in} needs both an implementation and a ref for DIFF mode in {direction} pass."
                )

            if direction == "FWD":

                @functools.wraps(impl_func)
                def fwd_diff_wrapper(diff_ctx, *diff_args, **diff_kwargs):
                    # NOTE(wenshuo.zhao): In DIFF mode, the 'ref' and 'std' implementations share the same 'ctx' object,
                    # which can lead to unexpected errors if their `save_for_backward` calls differ.
                    # We introduce a CtxProxy to hack the context's behavior for the ref impl.
                    class FwdRefCtxProxy:
                        def __init__(self, original_ctx):
                            object.__setattr__(self, "_original_ctx", original_ctx)

                        def save_for_backward(self, *tensors):
                            self._original_ctx._ref_saved_for_bwd = tensors

                        def __getattr__(self, name):
                            return getattr(self._original_ctx, name)

                        # NOTE(wenshuo.zhao): # During the forward pass, we need `__setattr__` to delegate the writing of non-tensor attributes.
                        def __setattr__(self, name, value):
                            setattr(self._original_ctx, name, value)

                    ref_ctx = FwdRefCtxProxy(diff_ctx)
                    ref_outputs = ref_func(ref_ctx, *diff_args, **diff_kwargs)

                    impl_outputs = impl_func(diff_ctx, *diff_args, **diff_kwargs)

                    ref_tuple = ref_outputs if isinstance(ref_outputs, tuple) else (ref_outputs,)
                    impl_tuple = impl_outputs if isinstance(impl_outputs, tuple) else (impl_outputs,)

                    if len(ref_tuple) != len(impl_tuple):
                        raise RuntimeError(f"Forward DIFF for {op_name_in}: Number of outputs mismatch.")

                    for i, (ref_o, impl_o) in enumerate(zip(ref_tuple, impl_tuple)):
                        assert torch.allclose(ref_o, impl_o, atol=0.01, rtol=0.01), (
                            f"Forward output {i} mismatch for {op_name_in}"
                        )

                    return impl_outputs

                return fwd_diff_wrapper

            else:

                @functools.wraps(impl_func)
                def bwd_diff_wrapper(diff_ctx, *diff_args, **diff_kwargs):
                    class BwdRefCtxProxy:
                        def __init__(self, original_ctx):
                            object.__setattr__(self, "_original_ctx", original_ctx)

                        @property
                        def saved_tensors(self):
                            return getattr(self._original_ctx, "_ref_saved_for_bwd", ())

                        def __getattr__(self, name):
                            return getattr(self._original_ctx, name)

                        # NOTE(wenshuo.zhao): # The backward pass proxy theoretically doesn't require `__setattr__` as it's primarily for reading state.
                        # However, we've kept it for robustness.
                        def __setattr__(self, name, value):
                            setattr(self._original_ctx, name, value)

                    ref_ctx = BwdRefCtxProxy(diff_ctx)
                    ref_grads = ref_func(ref_ctx, *diff_args, **diff_kwargs)

                    impl_grads = impl_func(diff_ctx, *diff_args, **diff_kwargs)

                    ref_tuple = ref_grads if isinstance(ref_grads, tuple) else (ref_grads,)
                    impl_tuple = impl_grads if isinstance(impl_grads, tuple) else (impl_grads,)

                    if len(ref_tuple) != len(impl_tuple):
                        raise RuntimeError(f"Backward DIFF for {op_name_in}: Number of gradients mismatch.")

                    for i, (ref_g, impl_g) in enumerate(zip(ref_tuple, impl_tuple)):
                        if ref_g is not None and impl_g is not None:
                            assert torch.allclose(ref_g, impl_g, atol=0.01, rtol=0.01), (
                                f"Backward gradient {i} mismatch for {op_name_in}"
                            )
                        elif ref_g is not None or impl_g is not None:
                            raise AssertionError(
                                f"Backward gradient {i} for {op_name_in}: one is None, the other is not."
                            )

                    return impl_grads

                return bwd_diff_wrapper

        raise ValueError(f"Invalid mode '{mode_str}' for {op_name_in} in {direction} pass.")

    @staticmethod
    def dispatched_forward(ctx, *args, **kwargs):
        layer_idx = getattr(ctx, "layer_idx", -1)

        final_func = _get_execution_function(op_name, "FWD", layer_idx)
        return final_func(ctx, *args, **kwargs)

    @staticmethod
    def dispatched_backward(ctx, *grad_outputs):
        layer_idx = getattr(ctx, "layer_idx", -1)

        final_func = _get_execution_function(op_name, "BWD", layer_idx)
        return final_func(ctx, *grad_outputs)

    cls.forward = dispatched_forward
    cls.backward = dispatched_backward
    return cls


class MojoFuncBase(Function):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        is_direct_child = MojoFuncBase in cls.__bases__

        if is_direct_child:
            cls._registry = []
        else:
            family_head = None
            for base in cls.mro()[1:]:
                if base is MojoFuncBase:
                    break
                if MojoFuncBase in base.__bases__:
                    family_head = base
                    break

            if family_head:
                default_priority = cls.default_priority if hasattr(cls, "default_priority") else 0

                env_var_name = f"{cls.__name__}_PRIORITY".upper()
                env_priority = os.getenv(env_var_name)

                priority = int(env_priority) if env_priority is not None else default_priority

                logger.info(
                    f"Register {cls.__name__} as {family_head.__name__} implementation with priority {priority}"
                )

                if priority in [x[0] for x in family_head._registry]:
                    raise ValueError(f"Operator {cls.__name__} priority {priority} has been registered")

                family_head._registry.append((priority, cls))
                family_head._registry.sort(reverse=False, key=lambda x: x[0])
