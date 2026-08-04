import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import noop_mask
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests


class TestFlexAttention(TestCase):
    _test_shapes = [(2, 4, 128, 64), (2, 8, 256, 64)]
    _test_dtypes = ["float16", "float32", "bfloat16"]

    @staticmethod
    def _generate_tensor(shape, dtype):
        return torch.randn(
            size=shape,
            dtype=getattr(torch, dtype),
            device=torch.device("npu"),
            requires_grad=True,
        )

    @staticmethod
    def _identity_score_mod(score, b, h, m, n):
        return score

    @staticmethod
    def _causal_score_mod(score, b, h, m, n):
        return torch.where(m >= n, score, float("-inf"))

    @staticmethod
    def _rel_bias_score_mod(score, b, h, m, n):
        return score + (m - n)

    @staticmethod
    def _causal_mask_mod(b, h, m, n):
        return m >= n

    def _get_score_mod(self, name):
        score_mods = {
            "identity": self._identity_score_mod,
            "causal": self._causal_score_mod,
            "rel_bias": self._rel_bias_score_mod,
        }
        return score_mods[name]

    def _get_mask_mod(self, name):
        mask_mods = {
            "noop": noop_mask,
            "causal": self._causal_mask_mod,
        }
        return mask_mods[name]

    @staticmethod
    def _get_tolerances(dtype):
        tolerances = {
            "float16": (1e-1, 0.0),
            "float32": (1e-1, 0.0),
            "bfloat16": (1e-1, 0.0),
        }
        return tolerances[dtype]

    def _assert_close(self, actual, expected, dtype):
        atol, rtol = self._get_tolerances(dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def run_flex_attention_case(self, score_mod_name, mask_mod_name, dtype, shape):
        score_mod = self._get_score_mod(score_mod_name)
        mask_mod = self._get_mask_mod(mask_mod_name)

        batch, heads, seq_len, _ = shape
        query = self._generate_tensor(shape, dtype)
        key = self._generate_tensor(shape, dtype)
        value = self._generate_tensor(shape, dtype)

        block_mask = create_block_mask(
            mask_mod,
            batch,
            heads,
            seq_len,
            seq_len,
            device="npu",
        )

        def op_calc(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
            )

        eager_out = op_calc(query, key, value)
        grad_out = torch.randn_like(eager_out)
        eager_out.backward(grad_out)

        eager_query_grad = query.grad.clone()
        eager_key_grad = key.grad.clone()
        eager_value_grad = value.grad.clone()

        query.grad = None
        key.grad = None
        value.grad = None

        compiled_op_calc = torch.compile(op_calc, backend="inductor", dynamic=False)
        inductor_out = compiled_op_calc(query, key, value)
        inductor_out.backward(grad_out)

        self._assert_close(eager_out, inductor_out, dtype)
        self._assert_close(eager_query_grad, query.grad, dtype)
        self._assert_close(eager_key_grad, key.grad, dtype)
        self._assert_close(eager_value_grad, value.grad, dtype)

    @parametrize("score_mod_name", ["identity", "causal", "rel_bias"])
    @parametrize("mask_mod_name", ["noop", "causal"])
    @parametrize("dtype", _test_dtypes)
    @parametrize("shape", _test_shapes)
    def test_flex_attention(self, score_mod_name, mask_mod_name, dtype, shape):
        self.run_flex_attention_case(score_mod_name, mask_mod_name, dtype, shape)


instantiate_parametrized_tests(TestFlexAttention)


if __name__ == "__main__":
    run_tests()
