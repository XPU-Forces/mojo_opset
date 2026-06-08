"""UC backend stub for UCSdpa.

STATUS: DISABLED — kernel 不在当前 uc-kernel wheel 中.

阻塞原因: wheel 不再编译该 specialized 变体 (通用 mojo_sdpa 可用，但 ABI 不兼容)
涉及 API : mojo_sdpa_b1_qh5_kvh1_s4096_d128
参考      : docs/project-ops/uc-kernel-fail-todo-2026-06-08.md
            docs/project-ops/mojo-opset-uc-wrapper-update-2026-06-08.md

用户硬约束 (2026-06-08): "wheel 没实现的就直接给报错" — wrapper 在 forward 中
必须直接 raise NotImplementedError, 不允许 silent fallback 到 torch。如需在
该算子上使用其他后端, 请改用 TTX / torch_npu / torch_native backend。
"""

import math
from typing import Optional
import torch
from mojo_opset.core import MojoSdpa

class UCSdpa(MojoSdpa):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCSdpa is disabled: mojo_sdpa_b1_qh5_kvh1_s4096_d128 not available in current uc-kernel wheel. Reason: wheel 不再编译该 specialized 变体 (通用 mojo_sdpa 可用，但 ABI 不兼容). See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md. Per project rule '没实现的就直接报错', this wrapper does not silently fall back to torch. Use TTX / torch_npu / torch_native backend instead."
        )
