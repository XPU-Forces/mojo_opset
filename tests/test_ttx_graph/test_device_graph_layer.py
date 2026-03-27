import pytest
import torch
import torch.nn as nn

from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoSwiGLUMLP
from mojo_opset.compile.device_graph import DeviceGraphRunner
from mojo_opset.utils.platform import get_platform


class SimpleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = MojoPagedPrefillGQA(is_causal=True, gqa_layout="AABB")

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.contiguous().view(batch_size * seq_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(batch_size * seq_len, self.num_heads, self.head_dim)
        v = v.contiguous().view(batch_size * seq_len, self.num_heads, self.head_dim)

        block_size = seq_len
        num_blocks = batch_size

        key_cache = k.view(num_blocks, self.num_heads, block_size, self.head_dim)
        value_cache = v.view(num_blocks, self.num_heads, block_size, self.head_dim)

        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=x.device)
        block_tables = torch.arange(0, num_blocks, dtype=torch.int32, device=x.device).view(batch_size, 1)
        seqlens_kv = torch.full((batch_size,), seq_len, dtype=torch.int32, device=x.device)

        attn_out = self.attn(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            block_tables=block_tables,
            seqlens_kv=seqlens_kv,
        )

        attn_out = attn_out.view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_out)


class DummyLayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8, intermediate_size=4096):
        super().__init__()

        self.rms_norm_1 = MojoRMSNorm(norm_size=hidden_size)
        self.rms_norm_2 = MojoRMSNorm(norm_size=hidden_size)
        self.self_attention = SimpleAttention(hidden_size, num_heads)
        self.ffn = MojoSwiGLUMLP(input_size=hidden_size, output_size=hidden_size, hidden_size=intermediate_size)

    def forward(self, hidden_states, session=None):

        residual = hidden_states
        normed_1 = self.rms_norm_1(hidden_states)
        attn_out = self.self_attention(normed_1)
        hidden_states = attn_out + residual

        residual = hidden_states
        normed_2 = self.rms_norm_2(hidden_states)
        ffn_out = self.ffn(normed_2)
        hidden_states = ffn_out + residual

        return hidden_states, session


class DummySession:
    def backup_state(self):
        pass

    def restore_state(self):
        pass


def test_device_graph_m8_dense_layer():
    device = get_platform()
    if device not in ["npu"]:
        pytest.skip(f"DeviceGraphRunner not supported on {device}")

    hidden_size = 256
    seq_len = 128
    batch_size = 2

    model = DummyLayer(hidden_size=hidden_size, num_heads=4, intermediate_size=1024)

    # Initialize weights to avoid NaNs from torch.empty
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 1:
            nn.init.normal_(param, mean=0.0, std=0.02)
        if "rms_norm" in name:
            nn.init.ones_(param)

    model = model.to(device).to(torch.bfloat16)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    test_hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    session = DummySession()

    runner = DeviceGraphRunner(model, device=device)

    with torch.inference_mode():
        runner.capture(hidden_states, session=session)
        graph_test_output, _ = runner.replay(test_hidden_states, session=session)

        eager_test_output, _ = model(test_hidden_states, session=session)

    torch.testing.assert_close(eager_test_output, graph_test_output, rtol=1e-3, atol=1e-3)
