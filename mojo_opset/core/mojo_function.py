import os
import functools
from torch.autograd import Function
from ..mojo_utils import get_mojo_exec_mode


import torch


_EXEC_FUNC_CACHE = {}


def mojo_func_dispatcher(cls):
    op_name = cls.__name__

    @staticmethod
    def dispatched_forward(ctx, *args, **kwargs):
        layer_idx = getattr(ctx, "layer_idx", -1)
        cache_key = (op_name, "FWD", layer_idx)

        if cache_key not in _EXEC_FUNC_CACHE:
            mode_str = get_mojo_exec_mode(op_name, "FWD", layer_idx)

            impl_func = cls._registry[0][1].forward if cls._registry else None
            ref_func = getattr(cls, "forward_ref", None)

            if mode_str == "STD":
                chosen_func = impl_func if impl_func else ref_func
                if not chosen_func:
                    raise NotImplementedError(f"{op_name} has no STD/REF forward implementation.")
            elif mode_str == "REF":
                if not ref_func:
                    raise NotImplementedError(f"{op_name} must implement forward_ref.")
                chosen_func = ref_func
            elif mode_str == "DUMP":
                dump_func = getattr(cls, "forward_dump", None)
                if not dump_func:
                    raise NotImplementedError(f"{op_name} must implement forward_dump.")
                chosen_func = dump_func
            elif mode_str == "DIFF":
                if not impl_func or not ref_func:
                    raise NotImplementedError(f"{op_name} needs both an implementation and a ref for DIFF mode.")

                @functools.wraps(impl_func)
                def diff_wrapper(diff_ctx, *diff_args, **diff_kwargs):
                    ref_result = ref_func(diff_ctx, *diff_args, **diff_kwargs)
                    impl_result = impl_func(diff_ctx, *diff_args, **diff_kwargs)
                    torch.testing.assert_close(ref_result, impl_result)
                    return impl_result

                chosen_func = diff_wrapper
            else:
                raise ValueError(f"Invalid forward mode {mode_str} for {op_name}")

            _EXEC_FUNC_CACHE[cache_key] = chosen_func

        final_func = _EXEC_FUNC_CACHE[cache_key]
        return final_func(ctx, *args, **kwargs)

    @staticmethod
    def dispatched_backward(ctx, *grad_outputs):
        layer_idx = -1
        cache_key = (op_name, "BWD", layer_idx)

        if cache_key not in _EXEC_FUNC_CACHE:
            mode_str = get_mojo_exec_mode(op_name, "BWD", layer_idx)

            impl_func = cls._registry[0][1].backward if cls._registry else None
            ref_func = getattr(cls, "backward_ref", None)
            dump_func = getattr(cls, "backward_dump", None)

            if mode_str == "STD":
                chosen_func = impl_func if impl_func else ref_func
                if not chosen_func:
                    raise NotImplementedError(
                        f"{op_name} needs at least a registered implementation or a 'backward_ref' for STD mode."
                    )

            elif mode_str == "REF":
                if not ref_func:
                    raise NotImplementedError(f"{op_name} must implement 'backward_ref' for REF mode.")
                chosen_func = ref_func

            elif mode_str == "DUMP":
                if not dump_func:
                    raise NotImplementedError(f"{op_name} must implement 'backward_dump' for DUMP mode.")
                chosen_func = dump_func

            elif mode_str == "DIFF":
                if not impl_func or not ref_func:
                    raise NotImplementedError(
                        f"{op_name} needs both an implementation and a 'backward_ref' for DIFF mode."
                    )

                @functools.wraps(impl_func)
                def diff_wrapper(diff_ctx, *diff_grad_outputs):
                    ref_grads = ref_func(diff_ctx, *diff_grad_outputs)
                    impl_grads = impl_func(diff_ctx, *diff_grad_outputs)

                    ref_grads_tuple = ref_grads if isinstance(ref_grads, tuple) else (ref_grads,)
                    impl_grads_tuple = impl_grads if isinstance(impl_grads, tuple) else (impl_grads,)

                    if len(ref_grads_tuple) != len(impl_grads_tuple):
                        raise RuntimeError(
                            f"Backward DIFF failed for {op_name}: Implementation and reference have different number of gradients."
                        )

                    for i, (ref_grad, impl_grad) in enumerate(zip(ref_grads_tuple, impl_grads_tuple)):
                        if ref_grad is not None and impl_grad is not None:
                            torch.testing.assert_close(
                                ref_grad, impl_grad, msg=f"Backward gradient {i} mismatch for {op_name}"
                            )
                        elif ref_grad is not None or impl_grad is not None:
                            raise AssertionError(
                                f"Backward gradient {i} mismatch for {op_name}: one is None and the other is not."
                            )

                    return impl_grads

                chosen_func = diff_wrapper
            else:
                raise ValueError(f"Invalid backward mode '{mode_str}' for {op_name}")

            _EXEC_FUNC_CACHE[cache_key] = chosen_func

        final_func = _EXEC_FUNC_CACHE[cache_key]

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

                print(f"Register {cls.__name__} as {family_head.__name__} implementation with priority {priority}")

                if priority in [x[0] for x in family_head._registry]:
                    raise ValueError(f"Operator {cls.__name__} priority {priority} has been registered")

                family_head._registry.append((priority, cls))
                family_head._registry.sort(reverse=False, key=lambda x: x[0])
