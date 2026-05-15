import pytest
import torch

from mojo_opset import MojoQuantLightningIndexer
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize("dtype", [torch.bfloat16])
class TestQuantLightningIndexer:
    @bypass_not_implemented
    def test_quant_lightning_indexer_basic(self, dtype):
        batch_size = 2
        q_seq_len = 4
        k_seq_len = 8
        head_num = 4
        head_dim = 64

        torch.manual_seed(0)
        query = torch.randn(batch_size, q_seq_len, head_num, head_dim, dtype=dtype)
        key = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
        weights = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
        query_dequant_scale = torch.randn(batch_size, q_seq_len, dtype=dtype).abs() + 0.1
        key_dequant_scale = torch.randn(batch_size, k_seq_len, dtype=dtype).abs() + 0.1
        query_quant_mode = 0
        key_quant_mode = 0

        mojo_op = MojoQuantLightningIndexer()
        ref_op = MojoQuantLightningIndexer._registry.get("torch")()

        out = mojo_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)
        ref_out = ref_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)

        torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_quant_lightning_indexer_with_quant_params(self, dtype):
        batch_size = 2
        q_seq_len = 4
        k_seq_len = 8
        head_num = 4
        head_dim = 64

        torch.manual_seed(0)
        query = torch.randn(batch_size, q_seq_len, head_num, head_dim, dtype=dtype)
        key = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
        weights = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
        query_dequant_scale = torch.randn(batch_size, q_seq_len, dtype=dtype).abs() + 0.1
        key_dequant_scale = torch.randn(batch_size, k_seq_len, dtype=dtype).abs() + 0.1
        query_quant_mode = 1
        key_quant_mode = 1

        mojo_op = MojoQuantLightningIndexer()
        ref_op = MojoQuantLightningIndexer._registry.get("torch")()

        out = mojo_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)
        ref_out = ref_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)

        torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_quant_lightning_indexer_different_shapes(self, dtype):
        head_num = 4
        head_dim = 64
        for batch_size, q_seq_len, k_seq_len in [(2, 4, 8), (4, 8, 16), (1, 2, 4)]:
            torch.manual_seed(0)
            query = torch.randn(batch_size, q_seq_len, head_num, head_dim, dtype=dtype)
            key = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
            weights = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
            query_dequant_scale = torch.randn(batch_size, q_seq_len, dtype=dtype).abs() + 0.1
            key_dequant_scale = torch.randn(batch_size, k_seq_len, dtype=dtype).abs() + 0.1
            query_quant_mode = 0
            key_quant_mode = 0

            mojo_op = MojoQuantLightningIndexer()
            ref_op = MojoQuantLightningIndexer._registry.get("torch")()

            out = mojo_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)
            ref_out = ref_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)

            torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_quant_lightning_indexer_register_call(self, dtype):
        batch_size = 2
        q_seq_len = 4
        k_seq_len = 8
        head_num = 4
        head_dim = 64

        torch.manual_seed(0)
        query = torch.randn(batch_size, q_seq_len, head_num, head_dim, dtype=dtype)
        key = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
        weights = torch.randn(batch_size, k_seq_len, head_dim, dtype=dtype)
        query_dequant_scale = torch.randn(batch_size, q_seq_len, dtype=dtype).abs() + 0.1
        key_dequant_scale = torch.randn(batch_size, k_seq_len, dtype=dtype).abs() + 0.1
        query_quant_mode = 0
        key_quant_mode = 0

        mojo_op = MojoQuantLightningIndexer()
        out = mojo_op.forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode)

        assert out.shape == (batch_size, q_seq_len, k_seq_len)
