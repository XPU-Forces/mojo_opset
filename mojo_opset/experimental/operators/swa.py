
def flash_attn_sparse_torch(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    gqa_interleave: bool = False,
    softmax_scale=None,
    local_window_size=0,
    global_window_size=0,
):

    T, H, Dq = q.shape
    Hk = k.shape[-2]
    gqa_ratio = H // Hk
    o = torch.zeros_like(q)
    if softmax_scale == None:
        softmax_scale = Dq ** (-0.5)

    bz = len(cu_seqlens_q) - 1

    for b in range(bz):
        for h in range(H):
            q_seq_start = cu_seqlens_q[b].item()
            q_seq_end = cu_seqlens_q[b + 1].item()
            q_seq_len = q_seq_end - q_seq_start
            kv_seq_start = cu_seqlens_kv[b].item()
            kv_seq_end = cu_seqlens_kv[b + 1].item()
            kv_seq_len = kv_seq_end - kv_seq_start
            kv_computed_len = kv_seq_len - q_seq_len

            if gqa_interleave:
                hk = h % Hk
            else:
                hk = h // gqa_ratio

            b_q = q[q_seq_start:q_seq_end, h, :].cpu().double()
            b_k = k[kv_seq_start:kv_seq_end, hk, :].cpu().double()
            b_v = v[kv_seq_start:kv_seq_end, hk, :].cpu().double()
            b_s = b_q @ b_k.T

            casual_mask = (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                >= torch.arange(0, kv_seq_len)[None, :]
            )
            if local_window_size is not None or global_window_size is not None:
                local_window_mask = (
                    (
                        torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                        <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
                    )
                    if local_window_size is not None
                    else False
                )
                global_window_mask = (
                    (torch.arange(0, kv_seq_len) < global_window_size)[None, :]
                    if global_window_size is not None
                    else False
                )
                b_s_mask = casual_mask & (local_window_mask | global_window_mask)
            else:
                b_s_mask = casual_mask

            b_s = torch.where(b_s_mask.to(device=b_s.device), b_s, -float("inf"))
            b_s = b_s * softmax_scale
            b_s = b_s.softmax(dim=-1)

            b_o = b_s @ b_v
            o[q_seq_start:q_seq_end, h, :] = b_o.to(o.dtype).to(o.device)

    return o


def paged_prefill_attn_sparse_torch(
    q,
    k_cache,
    v_cache,
    cu_seqlens,
    kvlens,
    block_table,
    gqa_interleave: bool = False,
    softmax_scale=None,
    local_window_size=0,
    global_window_size=0,
):

    T, H, Dq = q.shape
    _, Hk, P, _ = k_cache.shape
    gqa_ratio = H // Hk
    o = torch.zeros_like(q)
    if softmax_scale == None:
        softmax_scale = Dq ** (-0.5)

    bz = cu_seqlens.shape[0] - 1

    for b in range(bz):
        for h in range(H):
            seq_start = cu_seqlens[b].item()
            seq_end = cu_seqlens[b + 1].item()
            seq_len = seq_end - seq_start
            if gqa_interleave:
                hk = h % Hk
            else:
                hk = h // gqa_ratio

            b_q = q[seq_start:seq_end, h, :].cpu().double()
            kv_len = kvlens[b].item()
            b_pages = block_table[b, : (kv_len + P - 1) // P]
            b_k = k_cache[b_pages, hk].reshape(-1, Dq)[:kv_len].cpu().double()
            b_v = v_cache[b_pages, hk].reshape(-1, Dq)[:kv_len].cpu().double()
            b_s = b_q @ b_k.T

            casual_mask = torch.arange(kv_len - seq_len, kv_len)[:, None] >= torch.arange(0, kv_len)[None, :]
            if local_window_size is not None or global_window_size is not None:
                local_window_mask = (
                    (
                        torch.arange(kv_len - seq_len, kv_len)[:, None]
                        <= torch.arange(0, kv_len)[None, :] + local_window_size
                    )
                    if local_window_size is not None
                    else False
                )
                global_window_mask = (
                    (torch.arange(0, kv_len) < global_window_size)[None, :] if global_window_size is not None else False
                )
                b_s_mask = casual_mask & (local_window_mask | global_window_mask)
            else:
                b_s_mask = casual_mask

            b_s = torch.where(b_s_mask.to(device=b_s.device), b_s, -float("inf"))
            b_s = b_s * softmax_scale
            b_s = b_s.softmax(dim=-1)

            b_o = b_s @ b_v
            o[seq_start:seq_end, h, :] = b_o.to(o.dtype).to(o.device)

    return o


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    gqa_interleave: bool,
    head_dim: int,
    max_q_len: int,
    max_kv_prefix_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
    random_seed: Optional[int] = None,
):
    if random_seed is not None:
        set_seed(random_seed)
    q_lens = torch.randint(max(max_q_len // 2, 1), max_q_len + 1, (bsz,), dtype=torch.int32, device=device)
    if max_kv_prefix_len > 0:
        kv_prefix_lens = torch.randint(
            max_kv_prefix_len // 2, max_kv_prefix_len, (bsz,), dtype=torch.int32, device=device
        )
    else:
        kv_prefix_lens = torch.zeros(bsz, dtype=torch.int32, device=device)
    kv_lens = kv_prefix_lens + q_lens
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0)])
    cu_seqlens_kv = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), kv_lens.cumsum(0)])

    query = torch.randn(cu_seqlens_q[-1].item(), q_head_num, head_dim, dtype=dtype, device=device)
    key = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype, device=device)
    value = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype, device=device)

    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, gqa_interleave, key, value, cu_seqlens_q, cu_seqlens_kv


def generate_paged_prefill_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    gqa_interleave: bool,
    head_dim: int,
    max_q_len: int,
    max_kv_prefix_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
    random_seed: Optional[int] = None,
):
    if random_seed is not None:
        set_seed(random_seed)
    if dtype == torch.float32:
        page_size = 64
    else:
        page_size = 128
    q_lens = torch.randint(max(max_q_len // 2, 1), max_q_len + 1, (bsz,), dtype=torch.int32, device=device)
    if max_kv_prefix_len > 0:
        kv_prefix_lens = torch.randint(
            max_kv_prefix_len // 2, max_kv_prefix_len, (bsz,), dtype=torch.int32, device=device
        )
    else:
        kv_prefix_lens = torch.zeros(bsz, dtype=torch.int32, device=device)
    kv_lens = kv_prefix_lens + q_lens
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0)])

    max_num_pages = (max_kv_prefix_len + max_q_len + page_size - 1) // page_size * bsz * 2

    allocated_pages = (kv_lens + (page_size - 1)) // page_size
    page_idxs = torch.randperm(allocated_pages.sum().item(), device=device)
    cu_alloc_pages = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), allocated_pages.cumsum(0)])
    block_table = torch.zeros(bsz, max_num_pages, dtype=torch.int32, device=device)
    for i in range(bsz):
        block_table[i, : allocated_pages[i]] = page_idxs[cu_alloc_pages[i] : cu_alloc_pages[i + 1]]

    query = torch.randn(cu_seqlens_q[-1].item(), q_head_num, head_dim, dtype=dtype, device=device)
    key_cache = torch.randn(max_num_pages, kv_head_num, page_size, head_dim, dtype=dtype, device=device)
    value_cache = torch.randn(max_num_pages, kv_head_num, page_size, head_dim, dtype=dtype, device=device)

    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, gqa_interleave, key_cache, value_cache, cu_seqlens_q, kv_lens, block_table


def generate_paged_decode_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    gqa_interleave: bool,
    head_dim: int,
    max_kv_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
    random_seed: Optional[int] = None,
):
    query, gqa_interleave, key_cache, value_cache, cu_seqlens_q, kv_lens, block_table = generate_paged_prefill_test_data(
        bsz, q_head_num, kv_head_num, gqa_interleave, head_dim, 1, max_kv_len-1, dtype, device, random_seed,
    )
    torch.testing.assert_close(cu_seqlens_q.to(torch.int32), torch.arange(bsz+1, dtype=torch.int32, device=device))
    return query, gqa_interleave, key_cache, value_cache, kv_lens, block_table


@torch.no_grad
def test_swa_function(query, gqa_interleave, key, value, cu_seqlens_q, cu_seqlens_kv, profiler=None):
    import datetime

    local_window = 1023
    global_window = 4
    head_dim = query.shape[-1]
    scale = 1.0 / head_dim**0.5

    q_mojo = query
    k_mojo = key
    v_mojo = value
    torch.npu.synchronize()
    time = datetime.datetime.now()
    o_mojo = MojoSWA._registry.get("ttx")()(
        q_mojo,
        k_mojo,
        v_mojo,
        cu_seqlens_q,
        cu_seqlens_kv,
        True,
        local_window,
        global_window,
        scale,
        gqa_interleave,
    )

    elapsed_time = datetime.datetime.now() - time

    if profiler is not None:
        profiler.step()
    else:
        q_ref = query.clone()
        k_ref = key.clone()
        v_ref = value.clone()
        o_ref = flash_attn_sparse_torch(
            q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_kv, gqa_interleave, scale, local_window, global_window
        )
        assert_close(o_ref, o_mojo)
    return elapsed_time

@torch.no_grad
def test_paged_prefill_swa_function(query, gqa_interleave, key, value, cu_seqlens_q, kvlens, block_table, profiler=None):
    import datetime

    local_window = 1023
    global_window = 4
    head_dim = query.shape[-1]
    scale = 1.0 / head_dim**0.5
    q_mojo = query
    k_mojo = key
    v_mojo = value
    torch.npu.synchronize()
    time = datetime.datetime.now()
    o_mojo = MojoPagedPrefillSWA._registry.get("ttx")()(
        q_mojo,
        k_mojo,
        v_mojo,
        cu_seqlens_q,
        kvlens,
        block_table,
        True,
        local_window,
        global_window,
        scale,
        gqa_interleave,
    )
    torch.npu.synchronize()
    elapsed_time = datetime.datetime.now() - time

    if profiler is not None:
        profiler.step()
    else:
        q_ref = query.clone()
        k_ref = key.clone()
        v_ref = value.clone()
        o_ref = paged_prefill_attn_sparse_torch(
            q_ref,
            k_ref,
            v_ref,
            cu_seqlens_q,
            kvlens,
            block_table,
            gqa_interleave,
            scale,
            local_window,
            global_window,
        )
        assert_close(o_ref, o_mojo)
    return elapsed_time

@torch.no_grad
def test_paged_decode_swa_function(query, gqa_interleave, key, value, kvlens, block_table, profiler=None):
    import datetime

    head_dim = query.shape[-1]
    local_window = 1023
    global_window = 4
    scale = 1.0 / head_dim**0.5

    q_mojo = query
    k_mojo = key
    v_mojo = value
    torch.npu.synchronize()
    time = datetime.datetime.now()
    o_mojo = MojoPagedDecodeSWA._registry.get("ttx")()(
        q_mojo,
        k_mojo,
        v_mojo,
        kvlens,
        block_table,
        True,
        local_window,
        global_window,
        scale,
        gqa_interleave,
    )
    torch.npu.synchronize()
    elapsed_time = datetime.datetime.now() - time

    if profiler is not None:
        profiler.step()
    else:
        q_ref = query.clone()
        k_ref = key.clone()
        v_ref = value.clone()
        bsz = block_table.shape[0]
        o_ref = paged_prefill_attn_sparse_torch(
            q_ref,
            k_ref,
            v_ref,
            torch.arange(bsz+1, dtype=torch.int32),
            kvlens,
            block_table,
            gqa_interleave,
            scale,
            local_window,
            global_window,
        )
        assert_close(o_ref, o_mojo)
    return elapsed_time

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    import numpy

    numpy.random.seed(seed)

def assert_close(
    results,
    refs,
):
    """
    Asserts that the results are close to the reference tensors within specified tolerances.

    Args:
        results (Union[torch.Tensor, Tuple[Any, ...]]): The calculated result tensor(s).
        refs (Union[torch.Tensor, Tuple[Any, ...]]): The reference/golden tensor(s).

    Raises:
        AssertionError: If shapes, dtypes, or values do not match within tolerance.
    """
    assert type(results) is type(refs)
    if isinstance(results, torch.Tensor) and isinstance(refs, torch.Tensor):
        results = tuple([results])
        refs = tuple([refs])

    for result, ref in zip(results, refs):
        if isinstance(result, torch.Tensor) and isinstance(ref, torch.Tensor):
            assert result.shape == ref.shape
            assert result.dtype == ref.dtype
            dtype = result.dtype
            if dtype == torch.bfloat16:
                max_atol = 0.1
                max_rtol = 0.05
                mean_atol = 0.01
                mean_rtol = 0.01
            elif dtype == torch.float16:
                max_atol = 2e-2
                max_rtol = 2e-2
                mean_atol = 2e-2
                mean_rtol = 2e-2
            elif dtype == torch.float32:
                max_atol = 6e-3
                max_rtol = 6e-3
                mean_atol = 1e-4
                mean_rtol = 1e-4
            else:
                logger.warning(f"dtype {dtype} is not supported.")
                assert False

            torch.testing.assert_close(result.to(torch.float32), ref.to(torch.float32), atol=max_atol, rtol=max_rtol)
            assert (
                torch.mean(torch.abs(ref - result)) < max_atol
                or torch.mean(torch.abs((ref - result) / (ref + mean_atol))) < mean_rtol
            )
        else:
            assert result == ref


if __name__ == "__main__":
    test_func_map = {
        "infer": test_swa_function,
        "prefill": test_paged_prefill_swa_function,
        "decode": test_paged_decode_swa_function,
    }

    generate_test_data_func_map = {
        "infer": generate_test_data,
        "prefill": generate_paged_prefill_test_data,
        "decode": generate_paged_decode_test_data,
    }

    test_configs_map = {
        "infer": [
            # (1, 1, 1, False, 128, 256, 0, torch.float32),
            # (4, 4, 2, True, 128, 512, 0, torch.float32),
            # (4, 4, 2, False, 128, 256, 0, torch.bfloat16),
            (4, 16, 4, False, 128, 1024, 8192, torch.bfloat16),
        ],
        "prefill":[
            # (1, 1, 1, False, 128, 256, 0, torch.float32),
            # (4, 4, 2, True, 128, 512, 1024, torch.float32),
            # (4, 4, 2, False, 128, 256, 0, torch.bfloat16),
            (4, 16, 4, False, 128, 1024, 8192, torch.bfloat16),
        ],
        "decode": [
            # (1, 1, 1, False, 128, 256, torch.float32),
            # (4, 4, 2, True, 128, 1024, torch.float32),
            # (4, 4, 2, False, 128, 256, torch.bfloat16),
            (4, 16, 4, False, 128, 8192, torch.bfloat16),
        ],
    }
    
    import sys
    test_func = test_func_map[sys.argv[1]]
    generate_test_data_func = generate_test_data_func_map[sys.argv[1]]
    test_configs = test_configs_map[sys.argv[1]]

    for test_config in test_configs:
        print(*test_config)

        test_inputs = generate_test_data_func(
            *test_config, random_seed=42
        )

        e2e_times = []
        total_runs = 10
        for i in range(total_runs):
            e2e_times.append(test_func(*test_inputs))
            # torch.distributed.barrier()
        print("Avg E2E time:", sum((t.microseconds / 1000) for t in e2e_times[1:]) / (total_runs - 1), "ms")

        import torch_npu

        profiling_dir = "./npu_profiling"
        active = 5
        # 添加Profiling采集基础配置参数，详细参数介绍可参考下文的参数说明
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            l2_cache=False,
            data_simplification=False,
        )

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=active, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(total_runs):
                # 启动性能数据采集
                test_func(*test_inputs, prof)
        
        import os
        import csv

        try:
            kernel_profiling_path = max(
                [
                    os.path.join(profiling_dir, d)
                    for d in os.listdir(profiling_dir)
                    if os.path.isdir(os.path.join(profiling_dir, d))
                ],
                key=os.path.getmtime,
            )
            csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

            if not os.path.exists(csv_file_path):
                raise ValueError(f"File not found: {csv_file_path}")

        except Exception as e:
            raise ValueError(f"Failed to get Profiling folder name: {e}")

        total_avg_time_us = 0.0

        with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                avg_time = float(row["Total Time(us)"])
                total_avg_time_us += avg_time

        print("Avg device time:", total_avg_time_us / active, "us")