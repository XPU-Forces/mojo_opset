def apply_mojo_to_qwen3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model=None,
) -> None:
    """
    Apply mojo op to replace original implementation in HuggingFace Qwen3 models.
    """
    import torch
    import torch.nn as nn

    from mojo_opset import MojoNorm
    from mojo_opset import MojoPagedDecodeGQA
    from mojo_opset import MojoPagedPrefillGQA
    from mojo_opset import MojoRoPE
    from mojo_opset import MojoSwiGLU

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.qwen3 import modeling_qwen3

    if rope:
        modeling_qwen3.apply_rotary_pos_emb = MojoRoPE()

    # TODO(zhangjihang): need to discuss here, patch module seems a little bit tricky.
    # Better ways is that mojo_opset provide a functional api.
    # if rms_norm:
    #     modeling_qwen3.Qwen3RMSNorm.forward = MojoRMSNorm.forward

    if swiglu:

        class MojoSwiGLUMLP(nn.Module):
            def __init__(self, config):
                super().__init__()

                self.config = config
                self.hidden_size = config.hidden_size
                self.intermediate_size = config.intermediate_size

                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
                self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

                if config.hidden_act != "silu":
                    raise ValueError(f"MojoSwiGLUMLP requires 'silu' activation, but got {config.hidden_act}")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate_output = self.gate_proj(x)
                up_output = self.up_proj(x)

                silu = MojoSwiGLU()
                fused_output = silu(gate_output, up_output)

                return self.down_proj(fused_output)

        modeling_qwen3.Qwen3MLP = MojoSwiGLUMLP

    # NOTE: Currently, only a native decoder layer is implemented as a patch example; the
    # full model is not defined yet, so only static replacement before model instantiation is supported.
    # The following dynamic replacement code is temporarily commented out.

    # if model is not None:
    #     # The model instance already exists, so we need to additionally patch the
    #     # instance variables that ref already-instantiated modules

    #     # get the base model from the model instance
    #     base_model: Qwen3Model = getattr(model, model.base_model_prefix, model)

    #     if rms_norm:
    #         _patch_rms_norm_module(base_model.norm)
    #     for decoder_layer in base_model.layers:
    #         if swiglu:
    #             _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
    #         if rms_norm:
    #             _patch_rms_norm_module(decoder_layer.input_layernorm)
    #             _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


from contextlib import contextmanager
@contextmanager
def rewrite_assertion(module_name):
    from mojo_opset.utils.misc import get_bool_env
    disable = get_bool_env("MOJO_DISABLE_ASSERTION_REWRITE", False)
    if disable:
        return

    from .misc import get_bool_env
    from _pytest.stash import Stash
    from _pytest.assertion import install_importhook
    from mojo_opset.utils.logging import get_logger
    import contextlib
    class DummyConfig:
        def __init__(self):
            self.stash = Stash()
            self._cleanup_stack = []
            class DummyTrace:
                def __init__(self, logger):
                    self._logger = logger
                    class Root:
                        @staticmethod
                        def get(name):
                            return DummyTrace(get_logger(name))
                    self.root = Root
                def __call__(self, msg):
                    self._logger.debug(msg)

            self.trace = DummyTrace(get_logger(module_name))
        def getini(self, x):
            return ["*.py"]
        def add_cleanup(self, func):
            self._cleanup_stack.append(func)

    __dummy_cfg = DummyConfig()
    try:
        install_importhook(__dummy_cfg)
        yield
    finally:
        for func in __dummy_cfg._cleanup_stack:
            func()
