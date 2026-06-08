"""UC backend stub for UCQuantBatchGemmReduceSum.

STATUS: DISABLED — kernel 不在当前 uc-kernel wheel 中.

阻塞原因: ascendc_annotate_visitor.cc:43 crash (F)
涉及 API : mojo_quant_batch_gemm_reduce_sum_bf16
参考      : docs/project-ops/uc-kernel-fail-todo-2026-06-08.md
            docs/project-ops/mojo-opset-uc-wrapper-update-2026-06-08.md

用户硬约束 (2026-06-08): "wheel 没实现的就直接给报错" — wrapper 在 forward 中
必须直接 raise NotImplementedError, 不允许 silent fallback 到 torch。如需在
该算子上使用其他后端, 请改用 TTX / torch_npu / torch_native backend。
"""

import torch
from mojo_opset.experimental import MojoQuantBatchGemmReduceSum

class UCQuantBatchGemmReduceSum(MojoQuantBatchGemmReduceSum):
    supported_platforms_list = ["npu"]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "UC backend for UCQuantBatchGemmReduceSum is disabled: mojo_quant_batch_gemm_reduce_sum_bf16 not available in current uc-kernel wheel. Reason: ascendc_annotate_visitor.cc:43 crash (F). See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md. Per project rule '没实现的就直接报错', this wrapper does not silently fall back to torch. Use TTX / torch_npu / torch_native backend instead."
        )
