import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.SWAInfer", ops=["swa_infer"])
@pytest.mark.parametrize("dim,max_cache", [(128, 0), (96, 0), (128, 8192)])
def test_swa_infer(benchmark, perf_environment, dim, max_cache):
    _, device, _, implementation = perf_environment
    rng = torch.Generator().manual_seed(43)
    qlens = torch.randint(512, 1024, (2,), generator=rng, dtype=torch.int32)
    klens = (
        qlens + torch.randint(max_cache // 2, max_cache, (2,), generator=rng, dtype=torch.int32) if max_cache else qlens
    )

    def factory():
        module = modules.SWAInfer(
            gqa_layout="AABB", global_window_size=4, local_window_size=1023, implementation=implementation
        )
        q = torch.randn(int(qlens.sum()), 16, dim, dtype=torch.bfloat16, device=device)
        k, v = [torch.randn(int(klens.sum()), 4, dim, dtype=torch.bfloat16, device=device) for _ in range(2)]
        cq, ck = [
            torch.cat((torch.zeros(1, dtype=torch.int32), lens)).cumsum(0, dtype=torch.int32).to(device)
            for lens in (qlens, klens)
        ]
        return lambda: module(q, k, v, cq, ck, softmax_scale=dim**-0.5)

    benchmark(
        factory=factory,
        op="swa_infer",
        batch=2,
        q_heads=16,
        kv_heads=4,
        dim=dim,
        max_q=1024,
        max_cache=max_cache,
        dtype="bfloat16",
        phase="forward",
    )


@pytest.mark.api("modules.Sdpa", ops=["sdpa_infer"])
@pytest.mark.parametrize("seq_len", [64, 8192])
def test_sdpa(benchmark, perf_environment, seq_len):
    _, device, _, implementation = perf_environment
    if seq_len * 2 % 512 and implementation != "torch_reference":
        pytest.skip("Inherited perf_new smoke case is outside Triton SDPA's 512-token alignment")

    def factory():
        module = modules.Sdpa(scale=128**-0.5, enable_gqa=True, implementation=implementation)
        q = torch.randn(1, 8, seq_len * 2, 128, dtype=torch.bfloat16, device=device)
        k, v = [torch.randn(1, 2, seq_len * 2, 128, dtype=torch.bfloat16, device=device) for _ in range(2)]
        mask = torch.ones(seq_len * 2, seq_len * 2, dtype=torch.bool, device=device)
        return lambda: module(q, k, v, mask)

    benchmark(
        factory=factory,
        op="sdpa_infer",
        seq_len=seq_len,
        q_heads=8,
        kv_heads=2,
        head_dim=128,
        dtype="bfloat16",
        phase="forward",
    )
