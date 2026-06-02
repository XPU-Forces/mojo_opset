import logging

import torch.distributed as dist

from mojo_opset.distributed.parallel.mesh import LLMParallelConfig
from mojo_opset.distributed.parallel.mesh import build_contiguous_submesh
from mojo_opset.distributed.parallel.mojo_parallel import mojo_parallelize_module
from mojo_opset.distributed.parallel.expert_parallel import MojoExpertParallel
from mojo_opset.distributed.parallel.tensor_parallel import MojoColwiseParallel


logger = logging.getLogger(__name__)


def _split_even_range(total_size: int, world_size: int, rank: int) -> tuple[int, int]:
    shard = (total_size + world_size - 1) // world_size
    start = rank * shard
    end = min(start + shard, total_size)
    return start, end


def apply_llm_parallelize_plan(
    model,
    parallel_config: LLMParallelConfig,
    *,
    enable_lmhead_tp: bool = True,
    enable_attention_tp: bool = True,
    enable_moe_ep: bool = False,
    device_type: str = "npu",
):
    """Apply the minimal LLM TP plan through mojo_parallelize_module.

    This is intentionally conservative: LMHead TP is enabled first because the
    model forward already contains the required logits gather path. DeepSeekV4
    MLA Attention TP needs extra head/cache metadata rewrites, so this function
    only logs that a plan hook exists until that rewrite is implemented.
    """

    if not dist.is_initialized():
        return model

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if enable_lmhead_tp and parallel_config.lmhead_tp_size > 1:
        lmhead_mesh = build_contiguous_submesh(
            parallel_config.lmhead_tp_size,
            device_type=device_type,
            mesh_dim_name="lmhead_tp",
            world_size=world_size,
        )
        if lmhead_mesh is None:
            raise RuntimeError("lmhead_tp_size > 1 but lmhead DeviceMesh was not created")
        model = mojo_parallelize_module(
            model,
            device_mesh=lmhead_mesh,
            parallelize_plan={
                "lm_head": MojoColwiseParallel(),
            },
        )
        lmhead_tp_rank = lmhead_mesh.get_local_rank()
        model.lmhead_tp_rank = lmhead_tp_rank
        model.lmhead_vocab_start, model.lmhead_vocab_end = _split_even_range(
            model.config.vocab_size,
            parallel_config.lmhead_tp_size,
            lmhead_tp_rank,
        )
        model.local_vocab_size = model.lmhead_vocab_end - model.lmhead_vocab_start
        logger.info(
            "Applied LMHead TP through mojo_parallelize_module: rank=%s tp_rank=%s tp_size=%s vocab=[%s,%s)",
            rank,
            lmhead_tp_rank,
            parallel_config.lmhead_tp_size,
            model.lmhead_vocab_start,
            model.lmhead_vocab_end,
        )

    if enable_attention_tp and parallel_config.attn_tp_size > 1:
        # DeepSeekV4 uses MLA (wq_a/wq_b/wkv/wo_a/wo_b) rather than a standard
        # QKV/O projection block. The DeviceMesh + plan hook is wired here, but
        # the actual DeepSeekV4 Attention TP tensor rewrite is deferred to avoid
        # producing invalid head/cache metadata in the minimal closure.
        attn_mesh = build_contiguous_submesh(
            parallel_config.attn_tp_size,
            device_type=device_type,
            mesh_dim_name="attn_tp",
            world_size=world_size,
        )
        logger.info(
            "Attention TP DeviceMesh is ready for mojo_parallelize_module: rank=%s tp_rank=%s tp_size=%s. "
            "DeepSeekV4 MLA Attention module rewrite is not enabled in the minimal closure.",
            rank,
            None if attn_mesh is None else attn_mesh.get_local_rank(),
            parallel_config.attn_tp_size,
        )

    if enable_moe_ep and parallel_config.ep_size > 1:
        ep_mesh = build_contiguous_submesh(
            parallel_config.ep_size,
            device_type=device_type,
            mesh_dim_name="moe_ep",
            world_size=world_size,
        )
        if ep_mesh is None:
            raise RuntimeError("ep_size > 1 but moe_ep DeviceMesh was not created")
        model = mojo_parallelize_module(
            model,
            device_mesh=ep_mesh,
            parallelize_plan={
                "model.layers.*.mlp": MojoExpertParallel(
                    hccl_comm_dict=getattr(model, "hccl_comm_dict", None),
                    ep_size=parallel_config.ep_size,
                    ep_rank=parallel_config.ep_rank(rank),
                    global_rank=rank,
                ),
            },
        )
        logger.info(
            "Applied DeepSeekV4 MoE EP through mojo_parallelize_module: rank=%s ep_rank=%s ep_size=%s",
            rank,
            parallel_config.ep_rank(rank),
            parallel_config.ep_size,
        )

    return model
