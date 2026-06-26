"""LLM op: mojo_paged_prefill_gqa (base implementation).

Base ("torch 小算子") reference for paged prefill GQA attention, wired into an
xpu-perf ``BasicOp``. Builds the registered ``mojo_opset.core.MojoPagedPrefillGQA``
operator with its ``torch`` backend and runs ``forward``.
"""
from ._common import *

from mojo_opset.core import MojoPagedPrefillGQA
from xpu_perf.micro_perf.core.profiling_result import KernelExportOptions


def _mojo_paged_prefill_gqa_profiler_parser(to_be_parsed):
    """Exclude aclnnSub prep kernels when computing wall-clock latency."""
    from xpu_perf.micro_perf.core.profiling_result import compute_latency_us

    filtered = [
        ev for ev in to_be_parsed
        if "aclnnSub" not in ev.kernel_name
    ]
    return compute_latency_us(filtered)


def _create_block_tables(size, dtype, device, num_blocks_list):
    batch_size, width = size
    bt = torch.full((batch_size, width), -1, dtype=dtype, device=device)
    block_offset = 0
    for i in range(batch_size):
        row_blocks = num_blocks_list[i]
        for j in range(row_blocks):
            bt[i, j] = block_offset + j
        block_offset += row_blocks
    return bt


def _create_cu_from_lens(size, dtype, device, lens):
    _ = size
    lens_t = torch.tensor(lens, dtype=torch.int32, device=device)
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(lens_t, dim=0, dtype=torch.int32),
        ]
    ).to(dtype=dtype)


@ProviderRegistry.register_base_impl("mojo_paged_prefill_gqa", "ComputeEngine")
class MojoPagedPrefillGQAOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm", "batch_llm"]:
            raise ValueError(
                f"mojo_paged_prefill_gqa arg_type must be llm or batch_llm, got {self.arg_type}"
            )

        # Native MojoPagedPrefillGQA constructor / runtime params
        # (see docs/ops/mojo_paged_prefill_gqa.md).
        self.num_q_heads = self.args_dict["num_q_heads"]
        self.num_kv_heads = self.args_dict["num_kv_heads"]
        self.head_dim = self.args_dict["head_dim"]
        self.block_size = self.args_dict["block_size"]

        if self.arg_type == "llm":
            self.batch_size = int(self.args_dict["batch_size"])
            q_len = int(self.args_dict["q_len"])
            cache_len = int(self.args_dict["cache_len"])
            self.q_lens = [q_len] * self.batch_size
            self.cache_lens = [cache_len] * self.batch_size
        else:
            self.q_lens = [int(v) for v in self.args_dict["q_lens"]]
            self.cache_lens = [int(v) for v in self.args_dict["cache_lens"]]
            if len(self.q_lens) != len(self.cache_lens):
                raise ValueError(
                    f"q_lens/cache_lens length mismatch: "
                    f"{len(self.q_lens)} vs {len(self.cache_lens)}"
                )
            self.batch_size = len(self.q_lens)

        self.is_causal = self.args_dict.get("is_causal", True)
        self.gqa_layout = self.args_dict.get("gqa_layout", "AABB")
        self.dtype = self.args_dict.get("dtype", "bfloat16")

        # kv_len = cache_len + q_len
        self.kv_lens = [q + cache for q, cache in zip(self.q_lens, self.cache_lens)]
        self.max_q_len = max(self.q_lens)
        self.max_total_seq_len = max(self.kv_lens)
        self.num_blocks_list = [
            (kv_len + self.block_size - 1) // self.block_size for kv_len in self.kv_lens
        ]
        self.num_blocks_per_seq = max(self.num_blocks_list)
        self.num_total_blocks = sum(self.num_blocks_list)
        self.total_q_tokens = sum(self.q_lens)
        self.has_cache_prefix = any(cache_len > 0 for cache_len in self.cache_lens)

        self.softmax_scale = self.head_dim ** (-0.5)

    def vendor_parser(self):
        if self.gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(
                f"gqa_layout must be ABAB or AABB, got {self.gqa_layout}"
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(
                f"dtype={self.dtype} not supported, choose from "
                "float32 / float16 / bfloat16"
            )
        if self.num_blocks_per_seq <= 0:
            raise ValueError(
                f"max kv_len={self.max_total_seq_len} and block_size={self.block_size} "
                "produce zero blocks per sequence"
            )

    def flops_calc(self):
        self.calc_flops = 0
        for q_len, cache_len, kv_len in zip(self.q_lens, self.cache_lens, self.kv_lens):
            valid_parts = kv_len * kv_len
            if self.is_causal:
                valid_parts = (cache_len + 1 + kv_len) * q_len / 2
            else:
                valid_parts = q_len * kv_len
            # QK + PV matmuls
            self.calc_flops += 2 * (self.num_q_heads * self.head_dim * valid_parts * 2)

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        device = self.backend.get_torch_device_name()

        self.input_tensor_info = {}
        self.output_tensor_info = {}

        self.input_tensor_info["query"] = OpTensorInfo(
            shape=[self.total_q_tokens, self.num_q_heads, self.head_dim],
            dtype=self.torch_dtype,
            device=device,
        )
        self.input_tensor_info["key_cache"] = OpTensorInfo(
            shape=[
                self.num_total_blocks,
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            dtype=self.torch_dtype,
            device=device,
        )
        self.input_tensor_info["value_cache"] = OpTensorInfo(
            shape=[
                self.num_total_blocks,
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            dtype=self.torch_dtype,
            device=device,
        )
        self.input_tensor_info["cu_q_lens"] = OpTensorInfo(
            shape=[self.batch_size + 1],
            dtype=torch.int32,
            device=device,
            creator=partial(_create_cu_from_lens, lens=self.q_lens),
        )
        self.input_tensor_info["block_tables"] = OpTensorInfo(
            shape=[self.batch_size, self.num_blocks_per_seq],
            dtype=torch.int32,
            device=device,
            creator=partial(_create_block_tables, num_blocks_list=self.num_blocks_list),
        )
        if self.has_cache_prefix:
            self.input_tensor_info["cu_total_seq_lens"] = OpTensorInfo(
                shape=[self.batch_size + 1],
                dtype=torch.int32,
                device=device,
                creator=partial(_create_cu_from_lens, lens=self.kv_lens),
            )

        self.output_tensor_info["out"] = OpTensorInfo(
            shape=[self.total_q_tokens, self.num_q_heads, self.head_dim],
            dtype=self.torch_dtype,
        )

        self.input_tensor_size = sum(
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        )
        self.output_tensor_size = sum(
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.flops_calc()

        self._op = self._build_mojo_op(device)

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        self._run_func = self.vendor_impl_run
        self.profiler_parser = _mojo_paged_prefill_gqa_profiler_parser
        self.kernel_export_options = KernelExportOptions()

    def _build_mojo_op(self, device):
        """Construct the torch (small-op) backend of MojoPagedPrefillGQA."""
        op_cls = MojoPagedPrefillGQA.get_registry().get("torch")
        op = op_cls(
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
        )
        return op.to(device)

    def vendor_impl_run(self, tensor_mapping):
        cu_total_seq_lens = tensor_mapping.get("cu_total_seq_lens")
        return self._op(
            tensor_mapping["query"],
            tensor_mapping["key_cache"],
            tensor_mapping["value_cache"],
            tensor_mapping["cu_q_lens"],
            tensor_mapping["block_tables"],
            softmax_scale=self.softmax_scale,
            cu_total_seq_lens=cu_total_seq_lens,
            max_q_len=self.max_q_len,
            max_total_seq_len=self.max_total_seq_len,
        )
