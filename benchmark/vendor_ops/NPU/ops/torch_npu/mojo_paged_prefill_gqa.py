"""torch_npu vendor backend for mojo_paged_prefill_gqa.

Wraps ``mojo_opset``'s ``TorchNpuPagedPrefillGQA`` (which dispatches to
``torch_npu.npu_fused_infer_attention_score`` with TND + sparse_mode=3) and
registers it as the ``torch_npu`` vendor implementation.

``register_vendor_impl`` auto-merges this class with the base
``MojoPagedPrefillGQAOp``; only ``_build_mojo_op`` differs.
"""
from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset.core import MojoPagedPrefillGQA




@ProviderRegistry.register_vendor_impl("mojo_paged_prefill_gqa", "torch_npu")
class TORCH_NPU_MojoPagedPrefillGQAOp:
    def vendor_impl(self):
        super().vendor_impl()
        # Force BackendNPU.core_perf to use torch_npu.profiler instead of event timing.
        self.require_profiling = True

    def _build_mojo_op(self, device):
        """Construct the torch_npu backend of MojoPagedPrefillGQA."""
        op_cls = MojoPagedPrefillGQA.get_registry().get("torch_npu")
        op = op_cls(
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
        )
        return op.to(device)
