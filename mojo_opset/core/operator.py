import os

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Tuple

import torch

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoOperator(ABC, torch.nn.Module):
    supported_platforms_list = ["npu", "mlu", "meta_device"]

    def __init_subclass__(cls, **kwargs):
        # Consume any extra subclass kwargs (e.g., default_priority) without forwarding to ABC
        super().__init_subclass__()

        is_mojo_core_op_cls = MojoOperator in cls.__bases__

        if is_mojo_core_op_cls:
            from mojo_opset.core.backend_registry import MojoBackendRegistry

            # We place a registry for every core mojo op class.
            cls._registry = MojoBackendRegistry(cls)
            # Auto generate fallback dispatch backend "torch", it is registered within its-own __init_subclass__ call
            type("Torch" + cls._registry._operator_name, (cls,), {"__module__": cls.__module__})
        else:
            cls._registry.register(cls)

    def __new__(cls, *args, **kwargs):
        is_mojo_core_op_cls = MojoOperator in cls.__bases__

        if is_mojo_core_op_cls:
            if not hasattr(cls, "_registry") or not cls._registry:
                raise NotImplementedError(f"No {cls.__name__} implementation found, please register at least one.")

            import os

            target_backend = os.environ.get("MOJO_BACKEND")

            target_class = cls._registry.get(target_backend)
            instance = target_class.__new__(target_class, *args, **kwargs)
            return instance
        else:
            return super().__new__(cls)

    def __init__(self):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Any]:
        raise NotImplementedError

    def forward_diff_with(
        self,
        other_op: "MojoOperator",
        *args,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        random_seed: int = 42,
        mixed_tol: bool = False,
        **kwargs,
    ):
        """
        Args:
            *args: The arguments to pass to self.forward.
            other_op: The other operator to compare with.
            atol: The absolute tolerance.
            rtol: The relative tolerance.
            random_seed: The random seed to use.
            **kwargs: The keyword arguments to pass to self.forward.
        """
        # for some cases, we expect std & ref impl share the same random seed init state, i.e. sampling.
        torch.manual_seed(random_seed)
        # maybe inplace, deep copy is needed.
        args_for_std = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs_for_std = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        norm_result = self.forward(*args_for_std, **kwargs_for_std)

        torch.manual_seed(random_seed)
        args_for_ref = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs_for_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        refs_result = other_op.forward(*args_for_ref, **kwargs_for_ref)

        assert norm_result is not None, "forward should return a non-None value."
        assert refs_result is not None, "comparison operator should return a non-None value."

        if isinstance(norm_result, tuple) or isinstance(norm_result, list):
            for norm, ref in zip(norm_result, refs_result):
                if mixed_tol:
                    mask = ref.abs() < 1.0
                    tmpatol = tmprtol = 2**-6
                    torch.testing.assert_close(norm[mask], ref[mask], atol=tmpatol, rtol=0)
                    torch.testing.assert_close(norm[~mask], ref[~mask], atol=0, rtol=tmprtol)
                else:
                    torch.testing.assert_close(norm.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)
        else:
            if mixed_tol:
                mask = refs_result.abs() < 1.0
                tmpatol = tmprtol = 2**-6
                torch.testing.assert_close(norm_result[mask], refs_result[mask], atol=tmpatol, rtol=0)
                torch.testing.assert_close(norm_result[~mask], refs_result[~mask], atol=0, rtol=tmprtol)
            else:
                torch.testing.assert_close(
                    norm_result.to(torch.float32), refs_result.to(torch.float32), atol=atol, rtol=rtol
                )

        return norm_result
