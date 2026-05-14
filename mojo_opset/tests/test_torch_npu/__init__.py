# Unit tests for ``mojo_opset.backends.torch_npu`` (Ascend A2 / A5 / 950PR, …).
#
# CPU reference uses ``Mojo*._registry.get("torch")`` so TTX is not selected on NPU hosts.
# See individual test modules for stack-specific skips (e.g. ``npu_silu``, RoPE broadcast).
