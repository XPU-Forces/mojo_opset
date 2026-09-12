import csv
import os
import time
import pytest
import torch
import math

from mojo_opset import MojoSWAFunction

from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def compute_attention_area(
    seq_len: int,
    global_window: int,
    local_window: int,
    causal: bool = True,
) -> int:
    r"""
    计算滑动窗口注意力 (SWA) 的有效计算面积（非掩码 q-k 对数量）。

    掩码模式：
      - 全局窗口 (global_window, gw)：前 gw 个 key 对所有 query 可见
      - 局部窗口 (local_window, lw)：每个 query i 可看到距离不超过 lw 的 key

    Causal 模式下，query i 的可见 key 集合为：
        { k : 0 <= k <= i } ∩ ( { k : k < gw } ∪ { k : k >= i - lw } )

    当 i <= gw + lw - 1 时，全局与局部窗口重叠，整行 [0, i] 可见，计数 = i + 1；
    当 i >= gw + lw     时，两窗口间出现间隙，计数 = gw + lw + 1。

    有效面积公式（causal）：
        boundary = gw + lw
        若 seq_len <= boundary:  area = seq_len * (seq_len + 1) / 2
        若 seq_len >  boundary:  area = boundary*(boundary+1)/2
                                        + (seq_len - boundary) * (gw + lw + 1)

    边界情况：
        gw=0, lw=0 → area = seq_len  （仅对角线）
        gw=0       → 纯局部窗口因果注意力
        lw=0       → 仅全局 token + 自身
        gw+lw >= seq_len → 等价于 full causal attention

    Args:
        seq_len:       序列长度（query / key token 数）
        global_window: 全局窗口大小（前 gw 个 key 全局可见）
        local_window:  局部窗口大小（向后覆盖 lw 个 key + 自身）
        causal:        是否因果掩码

    Returns:
        有效 (query, key) 对的数量

    Examples:
        >>> compute_attention_area(1024, 4, 1023)
        523264
        >>> compute_attention_area(1024, 4, 4095)   # gw+lw > seq_len → full causal
        524800
        >>> compute_attention_area(1024, 0, 0)       # 仅对角线
        1024
        >>> compute_attention_area(10, 2, 3)         # 手工验证
        45
    """
    gw = max(0, min(global_window, seq_len))
    lw = max(0, min(local_window, seq_len))

    if seq_len == 0:
        return 0

    if causal:
        boundary = gw + lw
        if seq_len <= boundary:
            return seq_len * (seq_len + 1) // 2
        else:
            full_area = boundary * (boundary + 1) // 2
            sparse_rows = seq_len - boundary
            sparse_area = sparse_rows * (gw + lw + 1)
            return full_area + sparse_area
    else:
        global_area = seq_len * gw
        if seq_len <= lw + 1:
            local_area = seq_len * seq_len
        else:
            local_area = seq_len * (2 * lw + 1) - lw * (lw + 1)
        overlap = 0
        if gw > 0:
            for k in range(gw):
                lo = max(0, k - lw)
                hi = min(seq_len - 1, k + lw)
                overlap += hi - lo + 1
        return global_area + local_area - overlap


def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    torch.manual_seed(43)
    q_lens = torch.randint(max_q_len, max_q_len+1, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0).to(torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0).to(torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    grad_out = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    return query, key, value, grad_out, cu_q_lens, cu_total_seq_lens

test_configs_swa_perf = [
    (1, 12, 4, 128, 1024, 0,torch.bfloat16, "M_BF16_1024"),
    (1, 12, 4, 128, 2048, 0, torch.bfloat16, "M_BF16_2048"),
    (1, 12, 4, 128, 4096, 0, torch.bfloat16, "M_BF16_4096"),
    (1, 12, 4, 128, 8192, 0, torch.bfloat16, "M_BF16_8192"),
    (1, 12, 4, 128, 16384, 0, torch.bfloat16, "M_BF16_16384"),
    (1, 12, 4, 128, 32768, 0, torch.bfloat16, "M_BF16_32768"),
    (1, 12, 4, 128, 65536, 0, torch.bfloat16, "M_BF16_65536"),
    (1, 12, 4, 128, 131072, 0, torch.bfloat16, "M_BF16_131072"),  # 128K
    # (1, 12, 4, 128, 131072*2, 0, torch.bfloat16, "M_BF16_131072_2"), # 256K
    # (1, 12, 4, 128, 131072*4, 0, torch.bfloat16, "M_BF16_131072_4"), # 512K
    # (1, 12, 4, 128, 131072*8, 0, torch.bfloat16, "M_BF16_131072_8"),# 1M
]

@pytest.mark.parametrize(
    "query, key, value, grad_out, cu_q_lens, cu_total_seq_lens",
    [
        pytest.param(
            *generate_sdpa_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype, ID in test_configs_swa_perf
    ],
)
@pytest.mark.parametrize("gqa_interleave, global_window, local_window", [
    (False, 4, 1023),
    # (False, 4, 2048),
    # (False, 4, 4096),
])
@bypass_not_implemented
@auto_switch_platform()
def test_swa_function_perf( 
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    gqa_interleave: bool,
    global_window: int,
    local_window: int,
    ):
    import torch_npu

    # warm up for autotune
    for _ in range(3):        
        swa_func = MojoSWAFunction.apply
        head_dim = query.shape[-1]
        softmax_scale = 1.0 / math.sqrt(head_dim)
        q = query.clone().detach().requires_grad_(True)
        k = key.clone().detach().requires_grad_(True)
        v = value.clone().detach().requires_grad_(True)
        o = swa_func(
            q,
            k,
            v,
            cu_q_lens,
            cu_total_seq_lens,
            True,
            local_window,
            global_window,
            softmax_scale,
            gqa_interleave,
            True,
        )
        o.backward(grad_out) 

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )
    profiling_dir = "./npu_profiling"
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ) as prof:
        for _ in range(10):        
            swa_func = MojoSWAFunction.apply
            head_dim = query.shape[-1]
            softmax_scale = 1.0 / math.sqrt(head_dim)
            q = query.clone().detach().requires_grad_(True)
            k = key.clone().detach().requires_grad_(True)
            v = value.clone().detach().requires_grad_(True)
            o = swa_func(
                q,
                k,
                v,
                cu_q_lens,
                cu_total_seq_lens,
                True,
                local_window,
                global_window,
                softmax_scale,
                gqa_interleave,
                True,
            )
            o.backward(grad_out) 
            prof.step()
            torch.npu.synchronize()
            time.sleep(0.5)     # sleep for NPU calm down

    peak_tflops = 378.0  # for 950-PR
    if os.path.exists(profiling_dir):
        kernel_profiling_path = max(
            [
                os.path.join(profiling_dir, d)
                for d in os.listdir(profiling_dir)
                if os.path.isdir(os.path.join(profiling_dir, d))
            ],
            key=os.path.getmtime,
        )
        csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

        if os.path.exists(csv_file_path):
            kernel_times = {}
            with open(csv_file_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kernel_name = row["OP Type"]
                    for target in [
                        "_swa_fwd_kernel",
                        "_swa_bwd_dkdv_kernel",
                        "_swa_bwd_dkdv_local_kernel",
                        "_swa_bwd_dkdv_global_kernel",
                        "_swa_bwd_dkdv_reduce_kernel",
                        "_swa_bwd_dq_kernel",
                    ]:
                        if target in kernel_name:
                            kernel_times[target] = float(row["Avg Time(us)"])
                            break

            kernel_num_matmuls = {
                "_swa_fwd_kernel": 2,
                "_swa_bwd_dkdv_kernel": 4,
                "_swa_bwd_dkdv_local_kernel": 4,
                "_swa_bwd_dkdv_global_kernel": 4,
                "_swa_bwd_dkdv_reduce_kernel": 0,
                "_swa_bwd_dq_kernel": 3,
            }

            B = cu_q_lens.shape[0] - 1
            seq_len, head_num, head_dim = q.shape
            tot_kv_toks, KV_H, _ = k.shape
            effective_area = compute_attention_area(
                seq_len, global_window, local_window, causal=True
            )
            base_flops = B * head_num * effective_area * head_dim * 2

            print(f"\n{'='*60}")
            print(f"[SWA Perf] B={B}, Q_H={head_num}, KV_H={KV_H}, D={head_dim}, seq_len={seq_len}")
            print(f"[SWA Perf] global_window={global_window}, local_window={local_window}, gqa={gqa_interleave}")
            print(f"[SWA Perf] Peak={peak_tflops} TFLOPs")
            print(f"{'='*60}")

            use_q_split = "_swa_bwd_dkdv_local_kernel" in kernel_times

            if use_q_split:
                dkdv_kernels = [
                    "_swa_bwd_dkdv_local_kernel",
                    "_swa_bwd_dkdv_global_kernel",
                    "_swa_bwd_dkdv_reduce_kernel",
                ]
                dkdv_total_us = sum(kernel_times.get(k, 0.0) for k in dkdv_kernels)
                dkdv_flops = base_flops * 4
                dkdv_mfu = dkdv_flops / (dkdv_total_us / 1e6) / (peak_tflops * 1e12) if dkdv_total_us > 0 else 0.0
                print(f"[SWA Perf] dkdv path: Q-split (3 kernels)")
                for kn in dkdv_kernels:
                    us = kernel_times.get(kn, 0.0)
                    print(f"[SWA Perf]   {kn}: Avg Time={us:.2f} us")
                print(
                    f"[SWA Perf] _swa_bwd_dkdv (combined): "
                    f"Avg Time={dkdv_total_us:.2f} us, "
                    f"num_matmuls=4, "
                    f"FLOPs={dkdv_flops / 1e12:.4f} T, "
                    f"MFU={dkdv_mfu:.4f} ({dkdv_mfu*100:.2f}%)"
                )
            else:
                dkdv_kernels = ["_swa_bwd_dkdv_kernel"]

            for kernel_name in ["_swa_fwd_kernel"] + dkdv_kernels + ["_swa_bwd_dq_kernel"]:
                if kernel_name in ("_swa_bwd_dkdv_local_kernel", "_swa_bwd_dkdv_global_kernel", "_swa_bwd_dkdv_reduce_kernel"):
                    continue
                if kernel_name not in kernel_times:
                    print(f"[SWA Perf] {kernel_name}: not found in op_statistic.csv")
                    continue
                avg_time_us = kernel_times[kernel_name]
                duration_s = avg_time_us / 1e6
                num_matmuls = kernel_num_matmuls[kernel_name]
                effective_flops = base_flops * num_matmuls
                total_flops_t = effective_flops / 1e12
                mfu = total_flops_t / duration_s / peak_tflops
                print(
                    f"[SWA Perf] {kernel_name}: "
                    f"Avg Time={avg_time_us:.2f} us, "
                    f"num_matmuls={num_matmuls}, "
                    f"FLOPs={total_flops_t:.4f} T, "
                    f"MFU={mfu:.4f} ({mfu*100:.2f}%)"
                )
            print(f"{'='*60}\n")
        else:
            print(f"[SWA Perf] op_statistic.csv not found at: {csv_file_path}")
    else:
        print(f"[SWA Perf] Profiling directory not found: {profiling_dir}")