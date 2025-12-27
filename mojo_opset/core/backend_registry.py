from typing import Dict

from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.platform import get_platform

from .mojo_operator import MojoOperator

MOJO_BACKEND_LIST = ["ttx", "ref", "analysis"]

logger = get_logger(__name__)


class MojoBackendRegistry:
    def __init__(self, core_op_cls):
        assert core_op_cls.__name__.startswith("Mojo"), (
            f"Operator {core_op_cls.__name__} who is a subclass of MojoOperator, class name must start with Mojo."
        )
        self._core_op_cls = core_op_cls
        self._operator_name = core_op_cls.__name__[4:]
        self._registry: Dict[str, MojoOperator] = {}

    def register(self, cls: MojoOperator):
        idx = cls.__name__.find(self._operator_name)
        assert idx != -1, (
            f"Operator {cls.__name__} who be a subclass of {self._core_op_cls.__name__} must "
            f"contain {self._operator_name} in its name."
        )
        impl_backend_name = cls.__name__[:idx].lower()
        assert impl_backend_name in MOJO_BACKEND_LIST, (
            f"Operator {cls.__name__} backend[{impl_backend_name}] is not supported, "
            f"please choose from {MOJO_BACKEND_LIST}."
        )

        if get_platform() in cls.supported_platforms_list:
            logger.info(
                f"Register {cls.__name__} as {self._core_op_cls.__name__} implementation with backend[{impl_backend_name}]"
            )

            if impl_backend_name in [x[0] for x in self._registry]:
                raise ValueError(f"Operator {cls.__name__} backend[{impl_backend_name}] has been registered")

            # NOTE(zhangjihang): use a Enum to replace the string backend.
            self._registry[impl_backend_name] = cls
            self._sort()

    def get(self, backend_name: str) -> MojoOperator:
        assert backend_name in self._registry.keys(), (
            f"Operator {self.__name__} does not implement the target backend {backend_name}."
        )
        return self._registry[backend_name]

    def get_first_class(self) -> MojoOperator:
        assert len(self._registry) > 0, f"Operator {self.__name__} does not implement any backend."
        return self._registry[MOJO_BACKEND_LIST[0]]

    def _sort(self):
        self._registry = dict(sorted(self._registry.items(), key=lambda x: MOJO_BACKEND_LIST.index(x[0])))
