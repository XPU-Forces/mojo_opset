"""LLM op: mojo_quant_gemm (base implementation).

Base ("torch 小算子") reference for the W8A8 quantized GEMM, wired into an xpu-perf
``BasicOp`` so it can be driven by the micro_perf benchmark engine.

Rather than re-implementing the math inline, this builds the registered
``mojo_opset.core.MojoQuantGemm`` operator and selects its ``torch`` (small-op)
backend, then runs its ``forward`` -- mirroring how the ``torch_npu`` vendor impl
builds the operator from the registry. vendor backends are registered separately.
"""
from ._common import *

from mojo_opset.core import MojoQuantGemm


@ProviderRegistry.register_base_impl("mojo_quant_gemm", "ComputeEngine")
class MojoQuantGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm", "batch_llm"]:
            raise ValueError(
                f"mojo_quant_gemm arg_type must be llm or batch_llm, got {self.arg_type}"
            )

        # Native MojoQuantGemm constructor params (see docs/ops/mojo_quant_gemm.md).
        # in_features = K, out_features = N; num_tokens = M is the runtime batch.
        num_tokens = int(self.args_dict["num_tokens"])
        sp_size = int(self.args_dict["sp_size"])
        self.num_tokens = ((num_tokens + sp_size - 1) // sp_size) * sp_size
        self.sp_size = sp_size

        self.in_features = self.args_dict["in_features"]
        self.out_features = self.args_dict["out_features"]

        self.output_dtype = self.args_dict.get("output_dtype", "bfloat16")
        self.trans_weight = self.args_dict.get("trans_weight", False)
        self.quant_dtype = self.args_dict.get("quant_dtype", "int8")
        self.weight_dtype = self.args_dict.get("weight_dtype", "int8")

    def vendor_parser(self):
        # dtypes are fixed by the operator definition; only int8 W8A8 is supported.
        if self.quant_dtype != "int8" or self.weight_dtype != "int8":
            raise ValueError(
                f"MojoQuantGemm only supports int8 quantization, but got "
                f"quant_dtype={self.quant_dtype}, weight_dtype={self.weight_dtype}"
            )
        if self.output_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(
                f"MojoQuantGemm output_dtype={self.output_dtype} not supported, "
                f"choose from float32 / float16 / bfloat16"
            )

    def vendor_impl(self):
        self.input_torch_dtype = get_torch_dtype(self.quant_dtype)
        self.weight_torch_dtype = get_torch_dtype(self.weight_dtype)
        self.output_torch_dtype = get_torch_dtype(self.output_dtype)

        device = self.backend.get_torch_device_name()

        self.input_tensor_info = {}
        self.output_tensor_info = {}

        # per-token quantized activation: input int8 (M, K), input_scale fp32 (M,)
        self.input_tensor_info["input"] = OpTensorInfo(
            shape=[self.num_tokens, self.in_features],
            dtype=self.input_torch_dtype,
            device=device,
        )
        self.input_tensor_info["input_scale"] = OpTensorInfo(
            shape=[self.num_tokens],
            dtype=torch.float32,
            device=device,
            creator=torch.ones,
        )

        # per-channel quantized weight; logical layout is (K, N), stored as
        # (N, K) when trans_weight is set (matches MojoQuantGemm.weight_shape).
        weight_shape = (
            [self.out_features, self.in_features]
            if self.trans_weight
            else [self.in_features, self.out_features]
        )
        self.input_tensor_info["weight"] = OpTensorInfo(
            shape=weight_shape,
            dtype=self.weight_torch_dtype,
            device=device,
        )
        self.input_tensor_info["weight_scale"] = OpTensorInfo(
            shape=[self.out_features],
            dtype=torch.bfloat16,
            device=device,
            creator=torch.ones,
        )

        # output: (M, N) in output_dtype
        self.output_tensor_info["y"] = OpTensorInfo(
            shape=[self.num_tokens, self.out_features],
            dtype=self.output_torch_dtype,
        )

        # calculator
        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 2 * self.num_tokens * self.in_features * self.out_features

        # build the base (torch small-op) backend operator
        self._op = self._build_mojo_op(device)

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        self._run_func = self.vendor_impl_run

    def _build_mojo_op(self, device):
        """Construct the torch (small-op) backend of MojoQuantGemm."""
        op_cls = MojoQuantGemm.get_registry().get("torch")
        op = op_cls(
            in_features=self.in_features,
            out_features=self.out_features,
            output_dtype=self.output_torch_dtype,
            trans_weight=self.trans_weight,
        )
        return op.to(device)

    def vendor_impl_run(self, tensor_mapping):
        input = tensor_mapping["input"]
        input_scale = tensor_mapping["input_scale"]

        # rebind the registered buffers to the pre-allocated benchmark tensors
        self._op.weight = tensor_mapping["weight"]
        self._op.weight_scale = tensor_mapping["weight_scale"]

        y = self._op(input, input_scale)
        return y
