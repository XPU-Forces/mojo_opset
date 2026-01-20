import os

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoKernel:
    supported_platforms_list = ["npu", "mlu"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        is_mojo_kernel_cls = MojoKernel in cls.__bases__

        if is_mojo_kernel_cls:
            from mojo_opset.training.backend_registry import MojoBackendRegistry

            cls._registry = MojoBackendRegistry(cls)
        else:
            cls._registry.register(cls)
            cls._registry.sort()

            target_backend = os.environ.get("MOJO_BACKEND", None)
            core_op_cls = cls._registry.get_core_op_cls()

            core_op_cls.forward = cls._registry.get(target_backend).forward
            core_op_cls.backward = cls._registry.get(target_backend).backward

    @staticmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, *arg, **kwargs):
        pass
