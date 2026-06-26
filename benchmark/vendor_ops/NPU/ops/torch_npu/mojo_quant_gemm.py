"""torch_npu vendor backend for mojo_quant_gemm.

Wraps ``mojo_opset``'s ``TorchNpuQuantGemm`` (which dispatches to
``torch_npu.npu_quant_matmul``) and registers it as the ``torch_npu`` vendor
implementation of the ``mojo_quant_gemm`` op.

``register_vendor_impl`` auto-merges this class with the base ``MojoQuantGemmOp``;
only ``_build_mojo_op`` differs (selects the ``torch_npu`` backend).
"""
from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset.core import MojoQuantGemm


@ProviderRegistry.register_vendor_impl("mojo_quant_gemm", "torch_npu")
class TORCH_NPU_MojoQuantGemmOp:
    def vendor_impl(self):
        super().vendor_impl()
        # Force BackendNPU.core_perf to use torch_npu.profiler instead of event timing.
        self.require_profiling = True

    def _build_mojo_op(self, device):
        """Construct the torch_npu backend of MojoQuantGemm."""
        op_cls = MojoQuantGemm.get_registry().get("torch_npu")
        op = op_cls(
            in_features=self.in_features,
            out_features=self.out_features,
            output_dtype=self.output_torch_dtype,
            trans_weight=self.trans_weight,
        )
        return op.to(device)
