import os

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Tuple

import torch

from mojo_opset.mojo_utils import get_forward_mode
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoOperator(ABC, torch.nn.Module):
    def __init_subclass__(cls, default_priority=0, **kwargs):
        super().__init_subclass__(**kwargs)

        is_direct_child = MojoOperator in cls.__bases__

        if is_direct_child:
            cls._registry = []
        else:
            family_head = None
            for base in cls.mro()[1:]:
                if base is MojoOperator:
                    break
                if MojoOperator in base.__bases__:
                    family_head = base
                    break

            if family_head:
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

    def __new__(cls, *args, **kwargs):
        is_direct_child = MojoOperator in cls.__bases__

        if is_direct_child:
            if not hasattr(cls, "_registry") or not cls._registry:
                raise NotImplementedError(f"No {cls.__name__} implementation found, please register at least one.")

            highest_priority_class = cls._registry[0][1]
            instance = highest_priority_class.__new__(highest_priority_class)
            return instance
        else:
            return super().__new__(cls)

    def __init__(self, op_name: str, layer_idx: int):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)

        self.op_name = op_name
        self.layer_idx = layer_idx

        self._forward_map = {
            "STD": self.forward_std,
            "REFERENCE": self.forward_ref,
            "DUMP": self.forward_dump,
            "DIFF": self.forward_diff,
            "ANALYZE": self.forward_analysis,
        }

        mode, layer_idx = get_forward_mode()
        assert mode in self._forward_map, (
            f"Invalid forward mode {self._forward_mode}, please check the operator implementation."
        )

        if layer_idx == [] or self.layer_idx in layer_idx:
            self._forward_mode = mode
            self._inner_forward = self._forward_map[self._forward_mode]
        else:
            self._forward_mode = "STD"
            self._inner_forward = self._forward_map[self._forward_mode]

    def _set_forward_mode(self, mode: str):
        """
        Set the forward mode of the operator. This function should only be called from subclass.
        """

        assert mode in self._forward_map, f"Invalid forward mode {mode}, please check the operator implementation."
        self._forward_mode = mode
        self._inner_forward = self._forward_map[mode]

    def forward(self, *args, **kwargs) -> Tuple[Any]:
        return self._inner_forward(*args, **kwargs)

    def forward_diff(self, *args, atol: float = None, rtol: float = None, **kwargs) -> Tuple[Any]:
        """
        This function is used to check diff between forward_ref and forward_std which implemented by backend.

        Returns:
            Tuple[Any]: The result of the operator.
        """

        # maybe inplace, deep copy is needed.
        args_for_std = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs_for_std = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        norm_result = self.forward_std(*args_for_std, **kwargs_for_std)

        args_for_ref = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs_for_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        refs_result = self.forward_ref(*args_for_ref, **kwargs_for_ref)

        assert norm_result is not None, "forward_std should return a non-None value."
        assert refs_result is not None, "forward_ref should return a non-None value."

        if isinstance(norm_result, tuple) or isinstance(norm_result, list):
            for norm, ref in zip(norm_result, refs_result):
                torch.testing.assert_close(norm.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(norm_result.to(torch.float32), refs_result.to(torch.float32), atol=atol, rtol=rtol)

        return norm_result

    def forward_dump(self, *args, **kwargs) -> Tuple[Any]:
        """
        This function is used to dump the result of forward_std.

        Returns:
            Tuple[Any]: The result of the operator.
        """

        norm_result = self.forward_std(*args, **kwargs)
        if isinstance(norm_result, tuple):
            for res in norm_result:
                if isinstance(res, torch.Tensor):
                    torch.save(res, f"{self.layer_idx}_{self.op_name}.pt")
        else:
            if isinstance(res, torch.Tensor):
                torch.save(res, f"{self.layer_idx}_{self.op_name}.pt")

        return norm_result

    @abstractmethod
    def forward_ref(self, *args, **kwargs) -> Tuple[Any]:
        """
        Reference forward function, this function supposed to be implemented by Op designer.
        """

        raise NotImplementedError

    @abstractmethod
    def forward_std(self, *args, **kwargs) -> Tuple[Any]:
        """
        Normal forward function, this function supposed to be implemented by backend.
        """

        raise NotImplementedError

    @abstractmethod
    def forward_analysis(self, *args, **kwargs) -> Tuple[Any]:
        """
        This function is used to analyze the operator.

        Returns:
            Tuple[Any]: The result of the operator, IO Bytes / FLOPS.
        """
        raise NotImplementedError
