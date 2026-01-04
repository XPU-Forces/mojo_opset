from typing import Dict
from typing import Union

from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.platform import get_platform

from .function import MojoFunction
from .operator import MojoOperator

logger = get_logger(__name__)


class MojoBackendRegistry:
    def __init__(self, core_op_cls: Union[MojoOperator, MojoFunction]):
        assert core_op_cls.__name__.startswith("Mojo"), (
            f"Operator {core_op_cls.__name__} who is a subclass of MojoOperator, class name must start with Mojo."
        )
        self._core_op_cls = core_op_cls
        self._operator_name = core_op_cls.__name__[4:]
        self._registry: Dict[str, Union[MojoOperator, MojoFunction]] = {}
        # TODO(zhangjihang): support more backend priority list.
        self._backend_priority_list = ["ttx", "ref", "analysis"]  # Default backend implementation priority.

    def get_core_op_cls(self):
        return self._core_op_cls

    def register(self, cls: Union[MojoOperator, MojoFunction]):
        idx = cls.__name__.find(self._operator_name)
        assert idx != -1, (
            f"Operator {cls.__name__} who be a subclass of {self._core_op_cls.__name__} must "
            f"contain {self._operator_name} in its name."
        )
        impl_backend_name = cls.__name__[:idx].lower()
        assert impl_backend_name in self._backend_priority_list, (
            f"Operator {cls.__name__} backend[{impl_backend_name}] is not supported, "
            f"please choose from {self._backend_priority_list}."
        )

        curr_platform = get_platform()
        if curr_platform in cls.supported_platforms_list:
            logger.debug(
                f"Register {cls.__name__} as {self._core_op_cls.__name__} implementation with backend[{impl_backend_name}]"
            )

            if impl_backend_name in [x[0] for x in self._registry]:
                raise ValueError(
                    f"Operator {self._core_op_cls.__name__} backend[{impl_backend_name}] has been registered"
                )

            self._registry[impl_backend_name] = cls
            self.sort()
        else:
            logger.warning(f"Operator {cls.__name__} is not supported on {curr_platform} platform.")

    def get(self, backend_name: str = None) -> Union[MojoOperator, MojoFunction]:
        if (backend_name is None) or (backend_name not in self._registry.keys()):  # get first class
            assert len(self._registry) > 0, f"{self._operator_name} does not implement any backend."
            return list(self._registry.values())[0]

        assert backend_name in self._backend_priority_list, (
            f"{self._operator_name} does not support the target backend {backend_name}."
        )
        return self._registry[backend_name]

    def sort(self):
        self._registry = dict(sorted(self._registry.items(), key=lambda x: self._backend_priority_list.index(x[0])))
