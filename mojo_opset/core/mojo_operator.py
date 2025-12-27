from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Tuple
from typing import Union

import torch

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoOperator(ABC, torch.nn.Module):
    supported_platforms_list = ["npu", "mlu", "meta_device"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        is_direct_child = MojoOperator in cls.__bases__

        if is_direct_child:
            from mojo_opset.core.backend_registry import MojoBackendRegistry

            cls._registry = MojoBackendRegistry(cls)
        else:
            cls._registry.register(cls)

    def __new__(cls, *args, **kwargs):
        is_direct_child = MojoOperator in cls.__bases__

        if is_direct_child:
            if not hasattr(cls, "_registry") or not cls._registry:
                raise NotImplementedError(f"No {cls.__name__} implementation found, please register at least one.")

            import os

            target_backend = os.environ.get("MOJO_BACKEND", None)
            if target_backend is None:
                target_class = cls._registry.get_first_class()
            else:
                target_class = cls._registry.get(target_backend)

            instance = target_class.__new__(target_class, *args, **kwargs)
            return instance
        else:
            print(cls)
            return super().__new__(cls)

    def __init__(self, op_name: str, layer_idx: int):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)

        self.op_name = op_name
        self.layer_idx = layer_idx

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Any]:
        pass

    def forward_diff_with(
        self,
        *args,
        other_op: Union["MojoOperator", str],
        atol: float = 1e-2,
        rtol: float = 1e-2,
        random_seed: int = 42,
        **kwargs,
    ):
        """
        Compare the output of self.forward with other_op.forward.
        If other_op is a backend_name, requires two backend implement share same states.

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

        refs_result = None
        if isinstance(other_op, str):
            refs_result = self._registry.get(other_op).forward(self, *args_for_ref, **kwargs_for_ref)
        else:
            refs_result = other_op.forward(*args_for_ref, **kwargs_for_ref)

        assert norm_result is not None, "forward should return a non-None value."
        assert refs_result is not None, "comparison operator should return a non-None value."

        if isinstance(norm_result, tuple) or isinstance(norm_result, list):
            for norm, ref in zip(norm_result, refs_result):
                torch.testing.assert_close(norm.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(
                norm_result.to(torch.float32), refs_result.to(torch.float32), atol=atol, rtol=rtol
            )

        return norm_result

    def forward_diff_with_ref(
        self,
        *args,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        random_seed: int = 42,
        **kwargs,
    ):
        return self.forward_diff_with(
            *args,
            other_op="ref",
            atol=atol,
            rtol=rtol,
            random_seed=random_seed,
            **kwargs,
        )
