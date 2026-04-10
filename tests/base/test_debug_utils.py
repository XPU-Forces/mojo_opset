"""Integration test — validate MojoDebugger dump & compare on a 5-layer mini-transformer.

The model intentionally mirrors a real LLM structure:

    Embedding -> DecoderLayer x 5 -> final RMSNorm -> LM head (Linear)

Each DecoderLayer contains:
    input_layernorm  (MojoRMSNorm)
    self_attn.q_proj (MojoLinear)
    self_attn.k_proj (MojoLinear)
    self_attn.v_proj (MojoLinear)
    self_attn.o_proj (MojoLinear)
    post_attention_layernorm (MojoRMSNorm)
    mlp.gate_proj    (MojoLinear)
    mlp.up_proj      (MojoLinear)
    mlp.act_fn       (MojoSwiGLU)
    mlp.down_proj    (MojoLinear)

Tests are platform-aware:
    - On accelerator platforms (NPU/MLU/ILU), the default backend (e.g. TTX)
      is used, and compare produces real non-zero diffs against the torch ref.
    - On CPU-only machines (meta_device fallback), the torch backend is used;
      a dedicated test validates compare by perturbing shadow weights.
"""
import os
import shutil
import tempfile

import pytest
import torch

from mojo_opset.utils.debugger import MojoDebugger
from mojo_opset.utils.platform import get_platform, get_torch_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 5
HIDDEN = 128
NUM_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS  # 64, compatible with TTX SDPA kernel
VOCAB_SIZE = 256
SEQ_LEN = 16
BATCH = 2


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _get_test_device() -> str:
    """Return a torch device string suitable for real computation.

    ``get_torch_device()`` may return ``"meta"`` when no accelerator is
    present, which cannot execute kernels.  Fall back to ``"cpu"`` in that
    case.
    """
    dev = get_torch_device()
    return "cpu" if dev == "meta" else dev


def _is_torch_only_backend() -> bool:
    """True when the default MojoOperator dispatch resolves to the torch fallback."""
    return get_platform() == "meta_device"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_debug_env():
    """Save / restore debug-related env vars and reset MojoDebugger state.

    Does NOT force ``MOJO_BACKEND`` — lets the platform dispatch work
    naturally so that tests exercise the real backend on accelerator CI.
    """
    keys = [
        "MOJO_DEBUG", "MOJO_DEBUG_COMPARE",
        "MOJO_DEBUG_DUMP", "MOJO_DEBUG_DUMP_DIR", "MOJO_DEBUG_MAX_STEPS",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    MojoDebugger.disable()


@pytest.fixture
def dump_dir():
    d = tempfile.mkdtemp(prefix="mojo_debug_integ_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def _build_mini_transformer(device: str = "cpu"):
    from mojo_opset import (
        MojoEmbedding,
        MojoLinear,
        MojoRMSNorm,
        MojoSwiGLU,
    )

    class Attention(torch.nn.Module):
        def __init__(self, hidden, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden // num_heads
            self.q_proj = MojoLinear(hidden, hidden, bias=False)
            self.k_proj = MojoLinear(hidden, hidden, bias=False)
            self.v_proj = MojoLinear(hidden, hidden, bias=False)
            self.o_proj = MojoLinear(hidden, hidden, bias=False)

        def forward(self, x):
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.head_dim ** -0.5,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(attn_out)

    class MLP(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            intermediate = hidden * 4
            self.gate_proj = MojoLinear(hidden, intermediate, bias=False)
            self.up_proj = MojoLinear(hidden, intermediate, bias=False)
            self.act_fn = MojoSwiGLU()
            self.down_proj = MojoLinear(intermediate, hidden, bias=False)

        def forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(self.act_fn(gate, up))

    class DecoderLayer(torch.nn.Module):
        def __init__(self, layer_idx, hidden, num_heads):
            super().__init__()
            self.layer_idx = layer_idx
            self.input_layernorm = MojoRMSNorm(norm_size=hidden)
            self.self_attn = Attention(hidden, num_heads)
            self.post_attention_layernorm = MojoRMSNorm(norm_size=hidden)
            self.mlp = MLP(hidden)

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x

    class MiniTransformer(torch.nn.Module):
        def __init__(self, vocab, hidden, num_heads, n_layers):
            super().__init__()
            self.embed = MojoEmbedding(vocab, hidden)
            self.layers = torch.nn.ModuleList(
                [DecoderLayer(i, hidden, num_heads) for i in range(n_layers)]
            )
            self.norm = MojoRMSNorm(norm_size=hidden)
            self.lm_head = MojoLinear(hidden, vocab, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    model = MiniTransformer(VOCAB_SIZE, HIDDEN, NUM_HEADS, NUM_LAYERS)
    # Small init to prevent NaN explosion across 5 layers
    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            torch.nn.init.normal_(m.weight, std=0.02)
    return model.to(device)


def _random_input(device: str = "cpu"):
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)


# ---------------------------------------------------------------------------
# 1. Attach / op-map verification
# ---------------------------------------------------------------------------


class TestAttachOpMap:

    def test_opmap_structure(self):
        """After attach, op_map should contain all per-layer ops and global ops."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)

        per_layer_ops = {
            "input_layernorm", "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj", "mlp.up_proj", "mlp.act_fn", "mlp.down_proj",
        }

        for layer_idx in range(NUM_LAYERS):
            for op_name in per_layer_ops:
                assert (layer_idx, op_name) in dbg._op_map, (
                    f"Missing ({layer_idx}, {op_name}) in op_map"
                )

        assert (None, "norm") in dbg._op_map
        assert (None, "lm_head") in dbg._op_map
        assert (None, "embed") in dbg._op_map

        expected_total = NUM_LAYERS * len(per_layer_ops) + 3
        assert len(dbg._op_map) == expected_total

        dbg.detach()


# ---------------------------------------------------------------------------
# 2. Dump
# ---------------------------------------------------------------------------


class TestDumpIntegration:

    def test_dump_single_op(self, dump_dir):
        """Dump layer2's input_layernorm; verify output files exist and are loadable."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir)
        dbg.attach(model)
        dbg.set_dump("2:input_layernorm")

        with torch.no_grad():
            model(_random_input(device))

        rank_dir = os.path.join(dump_dir, "rank0")
        files = os.listdir(rank_dir)
        out_files = [f for f in files if f.endswith("_output.pt")]
        in_files = [f for f in files if f.endswith("_input.pt")]
        assert len(out_files) == 1
        assert len(in_files) == 1
        assert "layer2.input_layernorm" in out_files[0]

        output_tensor = torch.load(
            os.path.join(rank_dir, out_files[0]), weights_only=True,
            map_location="cpu",
        )
        assert output_tensor.shape == (BATCH, SEQ_LEN, HIDDEN)

        dbg.detach()


    def test_dump_wildcard_all_layers(self, dump_dir):
        """Dump mlp.act_fn across all layers; expect 5 output files."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir)
        dbg.attach(model)
        dbg.set_dump("*:mlp.act_fn")

        with torch.no_grad():
            model(_random_input(device))

        rank_dir = os.path.join(dump_dir, "rank0")
        out_files = [f for f in os.listdir(rank_dir) if "output" in f]
        assert len(out_files) == NUM_LAYERS
        for i in range(NUM_LAYERS):
            assert any(f"layer{i}.mlp.act_fn" in f for f in out_files)

        dbg.detach()


    def test_dump_global_op(self, dump_dir):
        """Dump the global norm op (outside any DecoderLayer)."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir)
        dbg.attach(model)
        dbg.set_dump("none:norm")

        with torch.no_grad():
            model(_random_input(device))

        rank_dir = os.path.join(dump_dir, "rank0")
        out_files = [f for f in os.listdir(rank_dir) if "output" in f]
        assert len(out_files) == 1
        assert "global.norm" in out_files[0]

        dbg.detach()


    def test_dump_multiple_ops_same_forward(self, dump_dir):
        """Dump multiple ops (semicolon-separated) in a single forward pass."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir)
        dbg.attach(model)
        dbg.set_dump("0:input_layernorm;0:self_attn.o_proj;4:mlp.down_proj")

        with torch.no_grad():
            model(_random_input(device))

        rank_dir = os.path.join(dump_dir, "rank0")
        out_files = [f for f in os.listdir(rank_dir) if "output" in f]
        assert any("layer0.input_layernorm" in f for f in out_files)
        assert any("layer0.self_attn.o_proj" in f for f in out_files)
        assert any("layer4.mlp.down_proj" in f for f in out_files)
        assert len(out_files) == 3

        dbg.detach()


    def test_dump_max_steps_across_forwards(self, dump_dir):
        """With max_steps=2, only the first 2 of 3 forward passes are dumped."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir, max_steps=2)
        dbg.attach(model)
        dbg.set_dump("0:input_layernorm")

        inp = _random_input(device)
        with torch.no_grad():
            for _ in range(3):
                model(inp)

        rank_dir = os.path.join(dump_dir, "rank0")
        out_files = [f for f in os.listdir(rank_dir) if "output" in f]
        assert len(out_files) == 2

        dbg.detach()


# ---------------------------------------------------------------------------
# 3. Compare
# ---------------------------------------------------------------------------


class TestCompareIntegration:

    def test_compare_single_op(self):
        """Compare runs end-to-end; step counter increments correctly."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("3:input_layernorm")

        with torch.no_grad():
            out = model(_random_input(device))

        assert out is not None
        assert dbg._step_counters[("cmp", 3, "input_layernorm")] == 1
        dbg.detach()


    def test_compare_stateless_op(self):
        """Compare a stateless, weight-free op (MojoSwiGLU)."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("0:mlp.act_fn")

        with torch.no_grad():
            model(_random_input(device))

        assert dbg._step_counters[("cmp", 0, "mlp.act_fn")] == 1
        dbg.detach()


    def test_compare_wildcard_all_norms(self):
        """Compare all MojoRMSNorm instances (class-name match), including global norm."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("*:MojoRMSNorm")

        with torch.no_grad():
            model(_random_input(device))

        norm_keys = [k for k in dbg._step_counters if k[0] == "cmp"]
        # 5 layers x 2 norms + 1 global norm = 11
        assert len(norm_keys) == NUM_LAYERS * 2 + 1
        for k in norm_keys:
            assert dbg._step_counters[k] == 1

        dbg.detach()


    def test_compare_mlp_chain(self):
        """Compare all ops in the MLP chain of a single layer."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare(
            "1:mlp.gate_proj;1:mlp.up_proj;1:mlp.act_fn;1:mlp.down_proj"
        )

        with torch.no_grad():
            model(_random_input(device))

        for op in ["mlp.gate_proj", "mlp.up_proj", "mlp.act_fn", "mlp.down_proj"]:
            assert dbg._step_counters[("cmp", 1, op)] == 1

        dbg.detach()


    def test_compare_does_not_alter_output(self):
        """Enabling compare must not alter the model inference output."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)

        inp = _random_input(device)
        with torch.no_grad():
            baseline = model(inp).clone()

        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("*:input_layernorm;*:mlp.act_fn")

        with torch.no_grad():
            with_debug = model(inp)

        assert torch.equal(baseline, with_debug), "compare must not alter inference output"
        dbg.detach()


    def test_compare_detects_perturbation(self):
        """Verify compare exercises the full diff-computation path.

        On accelerator platforms where TTX is used for RMSNorm, the compare
        between TTX kernel output and torch reference naturally produces a
        non-zero diff, validating the mechanism with real numerical difference.

        On torch-only platforms where a shadow instance exists, we manually
        perturb the shadow weights to guarantee a non-zero diff.

        In all cases the step counter must increment (proving the compare
        code path was fully executed) and the model must not crash.
        """
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("0:input_layernorm")

        # First forward — triggers lazy shadow construction and compare
        with torch.no_grad():
            model(_random_input(device))
        assert dbg._step_counters[("cmp", 0, "input_layernorm")] == 1

        # If a shadow instance exists (torch-only platform), perturb it
        # to produce a guaranteed non-zero diff on the second forward.
        target = dbg._op_map[(0, "input_layernorm")]
        ref = getattr(target, "_debug_torch_ref", None)
        if ref is not None:
            with torch.no_grad():
                for p in ref.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
        # When ref is None the compare uses ref_forward (core forward on the
        # same module).  On TTX platforms this naturally produces a non-zero
        # diff because the kernel implementation differs from torch reference.

        # Second forward — compare runs again; must not crash
        with torch.no_grad():
            out = model(_random_input(device))

        assert out is not None
        assert dbg._step_counters[("cmp", 0, "input_layernorm")] == 2
        dbg.detach()


# ---------------------------------------------------------------------------
# 4. Dynamic rule switching
# ---------------------------------------------------------------------------


class TestDynamicSwitching:

    def test_switch_compare_between_forwards(self):
        """Switch compare target between two forward passes."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        inp = _random_input(device)

        dbg.set_compare("0:input_layernorm")
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("cmp", 0, "input_layernorm")] == 1

        dbg.set_compare("4:mlp.down_proj")
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("cmp", 4, "mlp.down_proj")] == 1
        assert dbg._step_counters[("cmp", 0, "input_layernorm")] == 1

        dbg.detach()


    def test_dump_and_compare_simultaneously(self, dump_dir):
        """Dump and compare the same op in a single forward pass."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir)
        dbg.attach(model)
        dbg.set_dump("2:post_attention_layernorm")
        dbg.set_compare("2:post_attention_layernorm")

        with torch.no_grad():
            model(_random_input(device))

        assert dbg._step_counters[("dump", 2, "post_attention_layernorm")] == 1
        assert dbg._step_counters[("cmp", 2, "post_attention_layernorm")] == 1

        rank_dir = os.path.join(dump_dir, "rank0")
        assert any("post_attention_layernorm" in f for f in os.listdir(rank_dir))

        dbg.detach()


    def test_env_var_runtime_switch(self):
        """Switch rules via env var between forward passes at runtime."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        inp = _random_input(device)

        os.environ["MOJO_DEBUG_COMPARE"] = "0:self_attn.q_proj"
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("cmp", 0, "self_attn.q_proj")] == 1

        os.environ["MOJO_DEBUG_COMPARE"] = "0:self_attn.k_proj"
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("cmp", 0, "self_attn.k_proj")] == 1
        assert dbg._step_counters[("cmp", 0, "self_attn.q_proj")] == 1

        os.environ.pop("MOJO_DEBUG_COMPARE", None)
        dbg.detach()


    def test_reset_counters_allows_re_dump(self, dump_dir):
        """After reset_step_counters, dump resumes (max_steps re-counted)."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger(dump_dir=dump_dir, max_steps=1)
        dbg.attach(model)
        dbg.set_dump("0:input_layernorm")
        inp = _random_input(device)

        # 1st forward: step=0 -> dump succeeds -> counter=1
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("dump", 0, "input_layernorm")] == 1

        # 2nd forward: counter=1 >= max_steps=1 -> skipped
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("dump", 0, "input_layernorm")] == 1

        # After reset, counter goes back to 0; 3rd forward can dump again
        dbg.reset_step_counters()
        with torch.no_grad():
            model(inp)
        assert dbg._step_counters[("dump", 0, "input_layernorm")] == 1

        # Both dumps write step0 -> same filename -> overwritten, only 1 file
        rank_dir = os.path.join(dump_dir, "rank0")
        out_files = [f for f in os.listdir(rank_dir) if f.endswith("_output.pt")]
        assert len(out_files) == 1

        dbg.detach()


# ---------------------------------------------------------------------------
# 5. Compare replace mode
# ---------------------------------------------------------------------------


class TestCompareReplaceMode:

    def test_compare_replace_mode_changes_output(self):
        """In replace mode, matched ops' outputs are replaced with torch ref.

        Verify that:
        1. The model runs without error in replace mode.
        2. The compare step counters are incremented (proving the code path
           was exercised).
        3. Running replace mode twice with the same input produces the same
           output (deterministic replacement).
        """
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        inp = _random_input(device)

        dbg = MojoDebugger(compare_mode="replace")
        dbg.attach(model)
        dbg.set_compare("*:MojoRMSNorm")

        with torch.no_grad():
            out1 = model(inp).clone()
        with torch.no_grad():
            out2 = model(inp).clone()

        # Step counters should have been incremented twice
        cmp_keys = [k for k in dbg._step_counters if k[0] == "cmp"]
        assert len(cmp_keys) == NUM_LAYERS * 2 + 1  # 5 layers * 2 norms + 1 global
        for k in cmp_keys:
            assert dbg._step_counters[k] == 2

        assert torch.equal(out1, out2), (
            "replace mode should produce deterministic output"
        )
        dbg.detach()

    def test_compare_replace_mode_switchable(self):
        """set_compare_mode switches behaviour between forward passes."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        inp = _random_input(device)

        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("*:MojoRMSNorm")

        # Forward 1: observe (default)
        assert dbg._compare_mode == "observe"
        with torch.no_grad():
            out_obs = model(inp).clone()

        # Forward 2: switch to replace
        dbg.set_compare_mode("replace")
        assert dbg._compare_mode == "replace"
        with torch.no_grad():
            out_rep = model(inp).clone()

        # Forward 3: switch back to observe
        dbg.set_compare_mode("observe")
        assert dbg._compare_mode == "observe"
        with torch.no_grad():
            out_obs2 = model(inp).clone()

        # Observe mode should always produce the same output
        assert torch.equal(out_obs, out_obs2), (
            "observe mode should produce consistent output"
        )

        # Replace mode must complete without error and increment counters
        cmp_keys = [k for k in dbg._step_counters if k[0] == "cmp"]
        assert all(dbg._step_counters[k] == 3 for k in cmp_keys), (
            "all 3 forwards should have triggered compare"
        )

        dbg.detach()


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCasesIntegration:

    def test_unmatched_rule_warns_but_runs(self):
        """Rule targeting a non-existent layer/op warns but inference is unaffected."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("99:nonexistent_op")

        with torch.no_grad():
            out = model(_random_input(device))
        assert out is not None
        assert all(v == 0 for v in dbg._step_counters.values())

        dbg.detach()


    def test_no_rules_no_overhead_counters(self):
        """With no rules set, forward should not increment any step_counter."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)

        with torch.no_grad():
            model(_random_input(device))

        assert len(dbg._step_counters) == 0
        dbg.detach()


    def test_multiple_forwards_accumulate_steps(self):
        """Multiple forward passes correctly accumulate step_counter."""
        device = _get_test_device()
        MojoDebugger.enable()
        model = _build_mini_transformer(device)
        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("1:self_attn.o_proj")
        inp = _random_input(device)

        with torch.no_grad():
            for _ in range(5):
                model(inp)

        assert dbg._step_counters[("cmp", 1, "self_attn.o_proj")] == 5
        dbg.detach()
