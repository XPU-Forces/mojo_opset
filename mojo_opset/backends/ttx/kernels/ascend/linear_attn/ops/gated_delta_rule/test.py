import torch
import torch_npu
import torch.nn.functional as F
import triton
import triton.language as tl

from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.ops.gated_delta_rule.torch_impl import (
    torch_chunk_gated_delta_rule,
)
import time


def measure_time(func, warmup=3, repeat=10):
    for _ in range(warmup):
        func()
        torch.npu.synchronize()

    start = time.time()
    for _ in range(repeat):
        func()
        torch.npu.synchronize()
    end = time.time()

    avg_time = (end - start) / repeat * 1000
    return avg_time


def npu_run_perf(executor, profiling_dir):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )
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
        mat_a = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_b = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_c = torch.matmul(mat_a, mat_b)
        mat_c.cpu()
        for _ in range(10):
            executor()
            prof.step()
        torch.npu.synchronize()


def main_():
    # Test case with variable lengths
    B, T, H, K, V = 1, 512, 24, 128, 256
    Hk = 4
    # B, T, H, K, V = 1, 128, 1, 128, 256
    # Hk = 1

    # Create input tensors
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="npu", requires_grad=True)
    k = torch.randn(B, T, Hk, K, dtype=torch.bfloat16, device="npu", requires_grad=True)
    v = torch.randn(B, T, Hk, V, dtype=torch.bfloat16, device="npu", requires_grad=True)
    beta = torch.rand(B, T, Hk, dtype=torch.bfloat16, device="npu", requires_grad=True).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, Hk, dtype=torch.bfloat16, device="npu", requires_grad=True))

    # Reshape for variable length input
    cu_seqlens = torch.tensor(
        [0, T - 100, T],
        dtype=torch.long,
        device="npu",
    )

    # Run the function
    o_var, ht_var = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch_o_var, torch_ht_var = torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    # npu_run_perf(
    #     lambda: chunk_gated_delta_rule(
    #         q,
    #         k,
    #         v,
    #         g,
    #         beta,
    #         initial_state=None,
    #         output_final_state=True,
    #         cu_seqlens=cu_seqlens,
    #         head_first=False,
    #         use_qk_l2norm_in_kernel=True,
    #     ),
    #     "chunk_gated_delta_rule",
    # )
    # print(
    #     measure_time(
    #         lambda: chunk_gated_delta_rule(
    #             q,
    #             k,
    #             v,
    #             g,
    #             beta,
    #             initial_state=None,
    #             output_final_state=True,
    #             cu_seqlens=cu_seqlens,
    #             head_first=False,
    #             use_qk_l2norm_in_kernel=True,
    #         )
    #     )
    # )
    # print(
    #     measure_time(
    #         lambda: torch_chunk_gated_delta_rule(
    #             q,
    #             k,
    #             v,
    #             g,
    #             beta,
    #             cu_seqlens=cu_seqlens,
    #             use_qk_l2norm_in_kernel=True,
    #         )
    #     )
    # )
    # # Assert outputs are close
    torch.testing.assert_close(o_var, torch_o_var, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(ht_var, torch_ht_var, rtol=1e-1, atol=1e-1)
    # import ipdb, time, os

    # ipdb.set_trace() if int(os.environ.get("RANK", "0")) == 0 else time.sleep(9999)

    o_var.sum().backward()

    print("Passed")


def main(seed=42):
    """
    Runs a test case to compare the forward and backward pass of Triton
    and PyTorch implementations.
    """
    # Set seed for reproducibility

    torch.set_deterministic_debug_mode(True)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    # 1. Define tensor dimensions
    B, T, H, K, V = 1, 128, 32, 128, 256
    Hk = 32

    # 2. Create identical input tensors for both functions
    q = torch.randn(B, T, H, K, dtype=torch.float16, device="npu", requires_grad=True)
    k = torch.randn(B, T, Hk, K, dtype=torch.float16, device="npu", requires_grad=True)
    v = torch.randn(B, T, Hk, V, dtype=torch.float16, device="npu", requires_grad=True)
    beta = torch.rand(B, T, H, dtype=torch.float32, device="npu", requires_grad=True).sigmoid()
    beta.retain_grad()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device="npu", requires_grad=True))
    g.retain_grad()

    # Create copies for the PyTorch implementation to have separate grad attributes
    q_torch, k_torch, v_torch = (
        q.clone().detach(),
        k.clone().detach(),
        v.clone().detach(),
    )
    beta_torch, g_torch = beta.clone().detach(), g.clone().detach()
    q_torch.requires_grad_()
    k_torch.requires_grad_()
    v_torch.requires_grad_()
    beta_torch.requires_grad_()
    g_torch.requires_grad_()

    # Common arguments
    cu_seqlens = torch.tensor([0, T], dtype=torch.long, device="npu")
    common_kwargs = {
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
    }

    # 3. Run Triton implementation (forward and backward)
    o_triton, _ = chunk_gated_delta_rule(q, k, v, g, beta, **common_kwargs)
    # Use a simple sum for a dummy loss to trigger backward pass
    loss_triton = o_triton.sum()
    loss_triton.backward()

    # # 4. Run PyTorch implementation (forward and backward)
    o_torch, _ = torch_chunk_gated_delta_rule(q_torch, k_torch, v_torch, g_torch, beta_torch, **common_kwargs)
    loss_torch = o_torch.sum()
    loss_torch.backward()

    # 5. Assert that outputs and gradients are close
    print("Comparing forward pass outputs...")
    torch.testing.assert_close(o_triton, o_torch, rtol=1e-1, atol=1e-1)
    print("✅ Forward pass outputs match.")

    print("\nComparing gradients...")
    torch.testing.assert_close(q.grad, q_torch.grad, rtol=1e-1, atol=1e-1)
    print("✅ Gradients for 'q' match.")
    torch.testing.assert_close(k.grad, k_torch.grad, rtol=1e-1, atol=1e-1)
    print("✅ Gradients for 'k' match.")
    torch.testing.assert_close(v.grad, v_torch.grad, rtol=1e-1, atol=1e-1)
    print("✅ Gradients for 'v' match.")
    torch.testing.assert_close(beta.grad, beta_torch.grad, rtol=1e-1, atol=1e-1)
    print("✅ Gradients for 'beta' match.")
    torch.testing.assert_close(g.grad, g_torch.grad, rtol=1e-1, atol=1e-1)
    print("✅ Gradients for 'g' match.")
    print("✅ Gradients for 'q'", q.grad)
    print("✅ Gradients for 'k", k.grad)
    print("✅ Gradients for 'v", v.grad)
    print("✅ Gradients for 'g", g.grad)
    print("✅ Gradients for 'beta", beta.grad)
    return o_triton, q.grad, k.grad, v.grad, g.grad, beta.grad


# ==============================================================================
# Test and Benchmarking Logic
# ==============================================================================


def test_gdn(T, B, HQ, HK, DK, DV, provider, dtype=torch.bfloat16, device="npu"):
    """
    Sets up tensors and  runs a benchmark for a given MGDN implementation.
    """
    torch.manual_seed(42)

    # --- Variable Definitions ---
    cu_seq_lens = torch.tensor([0, T], device=device, dtype=torch.int32)

    # --- Generate Test Data ---
    # Use .clone().detach() to create fresh tensors for each run,
    # ensuring that gradient calculations from one provider don't affect another.
    q = torch.randn(B, T, HQ, DK, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(B, T, HK, DK, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(B, T, HK, DV, device=device, dtype=dtype).requires_grad_(True)
    beta = torch.rand(B, T, HK, dtype=dtype, device=device).sigmoid().requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, T, HK, dtype=dtype, device=device)).requires_grad_(True)
    initial_states = torch.rand(len(cu_seq_lens) - 1, HK, DK, DV, device=device, dtype=dtype)

    # We create clones for each run to ensure fair and isolated benchmarking
    inputs = (
        q.clone().detach().requires_grad_(True),
        k.clone().detach().requires_grad_(True),
        v.clone().detach().requires_grad_(True),
        g.clone().detach().requires_grad_(True),
        beta.clone().detach().requires_grad_(True),
        cu_seq_lens.clone().detach(),
        initial_states.clone().detach(),
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "chunk":

        def run_chunk():
            # Unpack cloned inputs
            q_in, k_in, v_in, g_in, beta_in, cu_in, is_in = inputs
            # Forward pass
            (
                O,
                _,
            ) = chunk_gated_delta_rule(
                q_in,
                k_in,
                v_in,
                g_in,
                beta_in,
                scale=1,
                initial_state=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_in,
            )
            loss = O.sum()
            loss.backward()
            return O

        ms, min_ms, max_ms = triton.testing.do_bench(run_chunk, quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],  # The variable we sweep over on the x-axis (Sequence Length)
        x_vals=[1024, 2048, 4096, 8192],  # Sequence lengths to test
        line_arg="provider",  # The argument that determines the line on the plot
        line_vals=[
            "chunk",
        ],  # The two implementations to compare
        line_names=[
            "GDN triton",
        ],  # Legend names for the plot
        styles=[("blue", "-"), ("green", "--"), ("red", "-.")],  # Line styles
        ylabel="ms",  # Y-axis label
        plot_name="MGDN Forward Performance",  # Plot title
        # --- Fixed arguments passed to the test_mgdn function ---
        args={"B": 1, "HQ": 32, "HK": 32, "DK": 128, "DV": 256},
    )
)
def benchmark_gdn(T, B, HQ, HK, DK, DV, provider):
    """Wrapper function for the Triton benchmark."""
    return test_gdn(T, B, HQ, HK, DK, DV, provider)


def run_test(seed=42):
    """
    Runs a test case to compare the forward and backward pass of Triton
    and PyTorch implementations.
    """
    # Set seed for reproducibility

    torch.set_deterministic_debug_mode(True)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    # 1. Define tensor dimensions
    B, T, H, K, V = 1, 128, 32, 128, 256
    Hk = 32

    # 2. Create identical input tensors for both functions
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="npu", requires_grad=True)
    k = torch.randn(B, T, Hk, K, dtype=torch.bfloat16, device="npu", requires_grad=True)
    v = torch.randn(B, T, Hk, V, dtype=torch.bfloat16, device="npu", requires_grad=True)
    beta = torch.rand(B, T, H, dtype=torch.float32, device="npu", requires_grad=True).sigmoid()
    beta.retain_grad()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device="npu", requires_grad=True))
    g.retain_grad()

    # Create copies for the PyTorch implementation to have separate grad attributes
    q_torch, k_torch, v_torch = (
        q.clone().detach(),
        k.clone().detach(),
        v.clone().detach(),
    )
    beta_torch, g_torch = beta.clone().detach(), g.clone().detach()
    q_torch.requires_grad_()
    k_torch.requires_grad_()
    v_torch.requires_grad_()
    beta_torch.requires_grad_()
    g_torch.requires_grad_()

    # Common arguments
    cu_seqlens = torch.tensor([0, T], dtype=torch.long, device="npu")
    common_kwargs = {
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
    }

    # 3. Run Triton implementation (forward and backward)
    o_triton, _ = chunk_gated_delta_rule(q, k, v, g, beta, **common_kwargs)
    # Use a simple sum for a dummy loss to trigger backward pass
    loss_triton = o_triton.sum()
    loss_triton.backward()

    # # 4. Run PyTorch implementation (forward and backward)
    # o_torch, _ = torch_chunk_gated_delta_rule(q_torch, k_torch, v_torch, g_torch, beta_torch, **common_kwargs)
    # loss_torch = o_torch.sum()
    # loss_torch.backward()

    # # 5. Assert that outputs and gradients are close
    # print("Comparing forward pass outputs...")
    # torch.testing.assert_close(o_triton, o_torch, rtol=1e-1, atol=1e-1)
    # print("✅ Forward pass outputs match.")

    # print("\nComparing gradients...")
    # torch.testing.assert_close(q.grad, q_torch.grad, rtol=1e-1, atol=1e-1)
    # print("✅ Gradients for 'q' match.")
    # torch.testing.assert_close(k.grad, k_torch.grad, rtol=1e-1, atol=1e-1)
    # print("✅ Gradients for 'k' match.")
    # torch.testing.assert_close(v.grad, v_torch.grad, rtol=1e-1, atol=1e-1)
    # print("✅ Gradients for 'v' match.")
    # torch.testing.assert_close(beta.grad, beta_torch.grad, rtol=1e-1, atol=1e-1)
    # print("✅ Gradients for 'beta' match.")
    # torch.testing.assert_close(g.grad, g_torch.grad, rtol=1e-1, atol=1e-1)
    # print("✅ Gradients for 'g' match.")
    print("✅ Gradients for 'q'", q.grad)
    print("✅ Gradients for 'k", k.grad)
    print("✅ Gradients for 'v", v.grad)
    print("✅ Gradients for 'g", g.grad)
    print("✅ Gradients for 'beta", beta.grad)
    return o_triton, q.grad, k.grad, v.grad, g.grad, beta.grad


# ==============================================================================
# Test and Benchmarking Logic
# ==============================================================================


def test_gdn(T, B, HQ, HK, DK, DV, provider, dtype=torch.bfloat16, device="npu"):
    """
    Sets up tensors and  runs a benchmark for a given MGDN implementation.
    """
    torch.manual_seed(42)

    # --- Variable Definitions ---
    cu_seq_lens = torch.tensor([0, T], device=device, dtype=torch.int32)

    # --- Generate Test Data ---
    # Use .clone().detach() to create fresh tensors for each run,
    # ensuring that gradient calculations from one provider don't affect another.
    q = torch.randn(B, T, HQ, DK, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(B, T, HK, DK, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(B, T, HK, DV, device=device, dtype=dtype).requires_grad_(True)
    beta = torch.rand(B, T, HK, dtype=dtype, device=device).sigmoid().requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, T, HK, dtype=dtype, device=device)).requires_grad_(True)
    initial_states = torch.rand(len(cu_seq_lens) - 1, HK, DK, DV, device=device, dtype=dtype)

    # We create clones for each run to ensure fair and isolated benchmarking
    inputs = (
        q.clone().detach().requires_grad_(True),
        k.clone().detach().requires_grad_(True),
        v.clone().detach().requires_grad_(True),
        g.clone().detach().requires_grad_(True),
        beta.clone().detach().requires_grad_(True),
        cu_seq_lens.clone().detach(),
        initial_states.clone().detach(),
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "chunk":

        def run_chunk():
            # Unpack cloned inputs
            q_in, k_in, v_in, g_in, beta_in, cu_in, is_in = inputs
            # Forward pass
            (
                O,
                _,
            ) = chunk_gated_delta_rule(
                q_in,
                k_in,
                v_in,
                g_in,
                beta_in,
                scale=1,
                initial_state=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_in,
            )
            loss = O.sum()
            loss.backward()
            return O

        ms, min_ms, max_ms = triton.testing.do_bench(run_chunk, quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],  # The variable we sweep over on the x-axis (Sequence Length)
        x_vals=[1024, 2048, 4096, 8192],  # Sequence lengths to test
        line_arg="provider",  # The argument that determines the line on the plot
        line_vals=[
            "chunk",
        ],  # The two implementations to compare
        line_names=[
            "GDN triton",
        ],  # Legend names for the plot
        styles=[("blue", "-"), ("green", "--"), ("red", "-.")],  # Line styles
        ylabel="ms",  # Y-axis label
        plot_name="MGDN Forward Performance",  # Plot title
        # --- Fixed arguments passed to the test_mgdn function ---
        args={"B": 1, "HQ": 32, "HK": 32, "DK": 128, "DV": 256},
    )
)
def benchmark_gdn(T, B, HQ, HK, DK, DV, provider):
    """Wrapper function for the Triton benchmark."""
    return test_gdn(T, B, HQ, HK, DK, DV, provider)


if __name__ == "__main__":
    print("Running GDN alignment...")
    main()

    # print("Running GDN benchmark...")
    # benchmark_gdn.run(show_plots=True, print_data=True)
    # print("--- First Run ---")
    # o1, dq1, dk1, dv1, dg1, dbeta1 = run_test(seed=42)
    # print("\n--- Second Run (with same seed) ---")
    # o2, dq2, dk2, dv2, dg2, dbeta2 = run_test(seed=42)
    # print("\nAll tests passed successfully across multiple runs!")
    # import ipdb, time, os; ipdb.set_trace() if int(os.environ.get("RANK", "0")) == 0 else time.sleep(9999)
