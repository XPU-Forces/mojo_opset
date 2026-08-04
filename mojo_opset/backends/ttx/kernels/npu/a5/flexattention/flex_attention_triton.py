import importlib.util
import os
import sys
import types
from pathlib import Path

import torch


Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128


def _load_parent_flex_attention_module():
    module_path = Path(__file__).resolve().parents[1] / "flex_attention.py"
    utils_path = module_path.parent / "utils.py"

    package_names = [
        "mojo_opset",
        "mojo_opset.utils",
        "mojo_opset.backends",
        "mojo_opset.backends.ttx",
        "mojo_opset.backends.ttx.kernels",
        "mojo_opset.backends.ttx.kernels.npu",
        "mojo_opset.backends.ttx.kernels.npu.a5",
    ]
    for package_name in package_names:
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

    platform_name = "mojo_opset.utils.platform"
    if platform_name not in sys.modules:
        platform_module = types.ModuleType(platform_name)

        def _get_torch_device():
            if hasattr(torch, "npu") and torch.npu.is_available():
                return torch.device("npu")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        def _get_platform():
            if hasattr(torch, "npu") and torch.npu.is_available():
                return "npu"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"

        platform_module.get_torch_device = _get_torch_device
        platform_module.get_platform = _get_platform
        sys.modules[platform_name] = platform_module

    utils_name = "mojo_opset.backends.ttx.kernels.npu.utils"
    if utils_name not in sys.modules:
        utils_spec = importlib.util.spec_from_file_location(utils_name, utils_path)
        if utils_spec is None or utils_spec.loader is None:
            raise RuntimeError(f"unable to load parent utils module: {utils_path}")
        utils_module = importlib.util.module_from_spec(utils_spec)
        utils_module.__package__ = "mojo_opset.backends.ttx.kernels.npu"
        sys.modules[utils_name] = utils_module
        utils_spec.loader.exec_module(utils_module)

    module_name = "mojo_opset.backends.ttx.kernels.npu.a5.flex_attention"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load parent flex_attention module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mojo_opset.backends.ttx.kernels.npu.a5"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    original_build_task_list = module._build_task_list

    def _safe_build_task_list(w_sparse, Hkv, num_kv_blocks, sparse_kv_multiple, target, num_core, device):
        # Tiny development cases can have total_w < num_core, making int(target) zero in the
        # parent task-list helper. Keep the loaded local copy runnable without touching source.
        return original_build_task_list(
            w_sparse,
            Hkv,
            num_kv_blocks,
            sparse_kv_multiple,
            max(float(target), 1.0),
            num_core,
            device,
        )

    module._build_task_list = _safe_build_task_list

    override_num_aicore = os.environ.get("FLEXATTENTION_TRITON_NUM_AICORE")
    if override_num_aicore:
        num_aicore = max(int(override_num_aicore), 1)
        module._get_num_aicore = lambda: num_aicore
    return module


_PARENT_FLEX_ATTENTION = None


def _parent_flex_attention():
    global _PARENT_FLEX_ATTENTION
    if _PARENT_FLEX_ATTENTION is None:
        _PARENT_FLEX_ATTENTION = _load_parent_flex_attention_module()
    return _PARENT_FLEX_ATTENTION


def triton_create_mask(problem, mask_type, tile_size=Q_BLOCK_SIZE):
    if mask_type == "sparse_doc_image":
        mask_type = "sparse"
    return _parent_flex_attention().triton_create_mask(problem, mask_type, tile_size=tile_size)


def create_block_mask_patched(mask_mod, q_len, kv_len=None, device=None):
    if kv_len is None:
        kv_len = q_len
    return _parent_flex_attention().create_block_mask_patched(
        mask_mod,
        B=1,
        H=1,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
    )


class _TritonFlexAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_mask, sm_scale):
        parent = _parent_flex_attention()
        output, lse = parent.flex_attention_fwd_impl(q, k, v, block_mask, sm_scale)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.block_mask = block_mask
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        parent = _parent_flex_attention()
        dq, dk, dv = parent.flex_attention_bwd_impl(
            grad_output,
            q,
            k,
            v,
            output,
            lse,
            ctx.block_mask,
            ctx.sm_scale,
        )
        return dq, dk, dv, None, None


def flex_attention(q, k, v, block_mask=None, sm_scale=None):
    return _TritonFlexAttentionFunction.apply(q, k, v, block_mask, sm_scale)


def make_triton_flex_attention_runner(block_mask, dense_mask=None):
    def _run(q, k, v):
        active_mask = block_mask
        if dense_mask is not None:
            active_mask.dense_mask = dense_mask
        return flex_attention(q, k, v, block_mask=active_mask)

    return _run


def run_triton_flex_attention(q, k, v, block_mask, dense_mask=None):
    return make_triton_flex_attention_runner(block_mask, dense_mask=dense_mask)(q, k, v)


def make_sdpa_reference_runner(dense_mask):
    def _run(q, k, v):
        group = q.size(1) // k.size(1)
        k_ref = k.repeat_interleave(group, dim=1) if group > 1 else k
        v_ref = v.repeat_interleave(group, dim=1) if group > 1 else v
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k_ref,
            v_ref,
            attn_mask=dense_mask[None, None, :, :],
            dropout_p=0.0,
            scale=q.size(-1) ** -0.5,
        )

    return _run
