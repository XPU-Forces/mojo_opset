import pytest
import torch

from mojo_opset import MojoMoEGatingTopK
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _make_uniform(shape, *, device: str, dtype: torch.dtype, low: float = -2.0, high: float = 2.0) -> torch.Tensor:
    return (torch.rand(*shape, device=device, dtype=dtype) * (high - low) + low).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm_type", [1, 2])
class TestMoEGatingTopK:
    @bypass_not_implemented
    def test_with_tid2eid_lookup(self, dtype, norm_type):
        device = get_torch_device()
        torch.manual_seed(0)

        batch_size = 16
        expert_count = 256
        k = 6
        n_vocab = 100

        x = _make_uniform((batch_size, expert_count), device=device, dtype=dtype)
        input_ids = torch.randint(0, n_vocab, (batch_size,), device=device, dtype=torch.int64)
        tid2eid = torch.randint(0, expert_count, (n_vocab, k), device=device, dtype=torch.int32)

        mojo_op = MojoMoEGatingTopK()
        ref_op = MojoMoEGatingTopK._registry.get("torch")()

        y_out, expert_idx_out, _ = mojo_op.forward(
            x,
            k,
            bias=None,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-6,
        )

        ref_y_out, ref_expert_idx_out, _ = ref_op.forward(
            x,
            k,
            bias=None,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-6,
        )

        torch.testing.assert_close(y_out.float(), ref_y_out.float(), atol=1e-2, rtol=1e-2)
        assert torch.equal(expert_idx_out, ref_expert_idx_out)

    @bypass_not_implemented
    def test_with_tid2eid_lookup_graph(self, dtype, norm_type):
        device = get_torch_device()
        if device != "npu":
            pytest.skip("Graph-mode test only runs on Ascend NPU.")

        try:
            import torchair  # noqa: F401
            from torchair.configs.compiler_config import CompilerConfig
        except Exception as e:
            pytest.skip(f"torchair is not available: {e}")

        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch build.")

        torch.manual_seed(0)

        batch_size = 16
        expert_count = 256
        k = 6
        n_vocab = 100

        x = _make_uniform((batch_size, expert_count), device=device, dtype=dtype)
        input_ids = torch.randint(0, n_vocab, (batch_size,), device=device, dtype=torch.int64)
        tid2eid = torch.randint(0, expert_count, (n_vocab, k), device=device, dtype=torch.int32)

        try:
            import custom_ops  # noqa: F401
        except Exception as e:
            pytest.skip(f"custom_ops is not available: {e}")

        if not hasattr(torch.ops.custom, "npu_moe_gating_top_k"):
            pytest.skip("torch.ops.custom.npu_moe_gating_top_k is not registered in this environment.")

        ref_op = MojoMoEGatingTopK._registry.get("torch")()

        ref_y_out, ref_expert_idx_out, _ = ref_op.forward(
            x,
            k,
            bias=None,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-6,
        )

        class Network(torch.nn.Module):
            def forward(
                self,
                x_npu,
                k_val,
                bias_val,
                input_ids_val,
                tid2eid_val,
                k_group_val,
                group_count_val,
                routed_scaling_factor_val,
                eps_val,
                group_select_mode_val,
                renorm_val,
                norm_type_val,
                out_flag_val,
            ):
                y_out, expert_idx_out, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu,
                    k_val,
                    bias=bias_val,
                    input_ids=input_ids_val,
                    tid2eid=tid2eid_val,
                    k_group=k_group_val,
                    group_count=group_count_val,
                    routed_scaling_factor=routed_scaling_factor_val,
                    eps=eps_val,
                    group_select_mode=group_select_mode_val,
                    renorm=renorm_val,
                    norm_type=norm_type_val,
                    out_flag=out_flag_val,
                )
                return y_out, expert_idx_out

        # Eager run once (align with upstream warmup behavior).
        _ = torch.ops.custom.npu_moe_gating_top_k(
            x,
            k,
            bias=None,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=1,
            group_count=1,
            routed_scaling_factor=1.0,
            eps=1e-6,
            group_select_mode=0,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
        )

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        import torchair

        npu_backend = torchair.get_npu_backend(compiler_config=config)

        npu_model = Network().to(device)
        try:
            npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)
        except TypeError:
            npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend)
        except Exception as e:
            pytest.skip(f"torch.compile failed on this environment: {e}")

        y_out, expert_idx_out = npu_model(
            x,
            k,
            None,
            input_ids,
            tid2eid,
            1,
            1,
            1.0,
            1e-6,
            0,
            0,
            norm_type,
            False,
        )

        torch.testing.assert_close(y_out.float(), ref_y_out.float(), atol=1e-2, rtol=1e-2)
        assert torch.equal(expert_idx_out, ref_expert_idx_out)

    @bypass_not_implemented
    def test_without_lookup(self, dtype, norm_type):
        if dtype == torch.float32:
            pytest.skip("Keep this case aligned with upstream sample coverage (fp16/bf16 only).")

        device = get_torch_device()
        torch.manual_seed(0)

        batch_size = 16
        expert_count = 384
        k = 6

        x = _make_uniform((batch_size, expert_count), device=device, dtype=dtype)

        mojo_op = MojoMoEGatingTopK()
        ref_op = MojoMoEGatingTopK._registry.get("torch")()

        y_out, expert_idx_out, _ = mojo_op.forward(
            x,
            k,
            bias=None,
            input_ids=None,
            tid2eid=None,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-6,
        )

        ref_y_out, ref_expert_idx_out, _ = ref_op.forward(
            x,
            k,
            bias=None,
            input_ids=None,
            tid2eid=None,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-6,
        )

        torch.testing.assert_close(y_out.float(), ref_y_out.float(), atol=1e-2, rtol=1e-2)
        assert torch.equal(expert_idx_out, ref_expert_idx_out)
