import os

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Tuple

import torch

from mojo_opset.utils.acc import check_tol_diff
from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.misc import get_tensor_factory_kwargs

logger = get_logger(__name__)


class MojoOperator(ABC, torch.nn.Module):
    supported_platforms_list = ["npu", "mlu", "meta_device"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

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

    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)
        self.tensor_factory_kwargs = get_tensor_factory_kwargs(**kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Any]:
        raise NotImplementedError

    def forward_diff_with(
        self,
        other_op: "MojoOperator",
        *args,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        ptol: float = 1.0,
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
            ptol: The percentage tolerance. When match_ratio >= ptol is considered to pass.
            random_seed: The random seed to use.
            mixed_tol: if true, atol, rtol and ptol are ignored.
            **kwargs: The keyword arguments to pass to self.forward.
        """
        # for some cases, we expect std & ref impl share the same random seed init state, i.e. sampling.
        os.environ["PYTHONHASHSEED"] = str(random_seed)
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

        check_tol_diff(norm_result, refs_result, atol, rtol, ptol, mixed_tol)

        return norm_result
