"""UC backend stub for UCSWAFunction.

STATUS: NOT IMPLEMENTED — 该算子无 UC kernel 实现.

阻塞原因: unic Pattern A binary.cc:281 双轴 bcast 断言 + split_dim=0 cross-core race
涉及 API : (无 UC kernel)
参考      : docs/project-ops/uc-kernel-fail-todo-2026-06-08.md
            docs/project-ops/mojo-opset-uc-wrapper-update-2026-06-08.md

用户硬约束 (2026-06-08): "wheel 没实现的就直接给报错" — wrapper 在 forward 中
必须直接 raise NotImplementedError, 不允许 silent fallback 到 torch。如需在
该算子上使用其他后端, 请改用 TTX / torch_npu / torch_native backend。
"""

from mojo_opset.core import MojoSWAFunction

class UCSWAFunction(MojoSWAFunction):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCSWAFunction is disabled: (无 UC kernel) not available in current uc-kernel wheel. Reason: unic Pattern A binary.cc:281 双轴 bcast 断言 + split_dim=0 cross-core race. See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md. Per project rule '没实现的就直接报错', this wrapper does not silently fall back to torch. Use TTX / torch_npu / torch_native backend instead."
        )

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        raise NotImplementedError(
            "UC backend for UCSWAFunction is disabled: see forward() for details."
        )
