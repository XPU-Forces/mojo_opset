from importlib import import_module
from typing import Callable
from typing import Optional

import torch

from mojo_opset.config import _config
from mojo_opset.config._config import normalize_name
from mojo_opset.utils.target import Target
from mojo_opset.utils.target import TargetLike
from mojo_opset.utils.target import detect_target

_PROVIDERS = {
    ("npu.a2", "triton"): "mojo_opset.kernels.npu_a2_triton",
    ("npu.a5", "triton"): "mojo_opset.kernels.npu_a5_triton",
    ("npu.a2", "torch_npu"): "mojo_opset.kernels.npu_torch_npu",
    ("npu.a5", "torch_npu"): "mojo_opset.kernels.npu_torch_npu",
    ("npu.a2", "native"): "mojo_opset.kernels.npu_a2_native",
    ("npu.a5", "native"): "mojo_opset.kernels.npu_a5_native",
    ("npu.a5", "cannbotdsl"): "mojo_opset.kernels.npu_a5_cannbotdsl",
    ("ilu.generic", "triton"): "mojo_opset.kernels.ilu_triton",
    ("ilu.generic", "ixformer"): "mojo_opset.kernels.ilu_ixformer",
    ("mlu.generic", "triton"): "mojo_opset.kernels.mlu_triton",
}


def _provider_package(target: Target, implementation: str) -> str:
    try:
        return _PROVIDERS[(target.arch_key, implementation)]
    except KeyError:
        raise LookupError(
            f"No provider package (no exact binding) for target={target.key!r}, implementation={implementation!r}."
        ) from None


def _load_kernel_binding(
    provider_package: str,
    target_key: Optional[str],
    op_id: str,
) -> tuple[Callable, Optional[Callable]]:
    provider = import_module(provider_package)
    ops = provider.OPS
    if target_key is not None:
        ops = {**ops, **getattr(provider, "OVERRIDES", {}).get(target_key, {})}
    module_name = ops.get(op_id)
    if module_name is None:
        raise LookupError(
            f"Provider {provider_package!r} has no exact binding for target={target_key!r}, op={op_id!r}."
        ) from None

    if not isinstance(module_name, str) or not module_name:
        raise TypeError(f"Provider {provider_package!r} must map op={op_id!r} to a non-empty module name.")
    module = import_module(f"{provider_package}.{module_name}")
    forward_name = f"{op_id}_fwd"
    backward_name = f"{op_id}_bwd"
    try:
        forward = getattr(module, forward_name)
    except AttributeError as error:
        raise LookupError(f"Module {module.__name__!r} does not export required wrapper {forward_name!r}.") from error
    backward = getattr(module, backward_name, None)
    if not callable(forward) or (backward is not None and not callable(backward)):
        raise TypeError(f"Invalid callable binding declared by {module.__name__}.")
    return forward, backward


# Plain dictionary lookups can be guarded by Dynamo. Dynamic imports cannot.
_BINDINGS = {}
# Keep the snapshot and its cache together. In-flight calls may finish against
# the old cache, but cannot populate a replacement cache with stale bindings.
_DISPATCH_STATE = (None, {})


def resolve_implementation(op_id: str, target: Target, snapshot=None) -> str:
    """Look up configuration only on a dispatch cache miss."""
    if snapshot is None:
        snapshot = _config._get_config_snapshot()
    implementation = snapshot.get(target.key, {}).get(op_id)
    if implementation is None:
        implementation = snapshot.get(target.arch_key, {}).get(op_id)
    if implementation is None:
        raise LookupError(f"No implementation configured for op={op_id!r}, target={target.key!r}.")
    return implementation


def load_impl(
    op_id: str,
    implementation: Optional[str] = None,
    *,
    target: Optional[TargetLike] = None,
    require_backward: bool = False,
) -> tuple[Callable, Optional[Callable]]:
    """Select and load an op's forward/backward wrappers, reusing cached bindings.

    This does not execute kernels. Training APIs set require_backward=True;
    this requirement is checked on cached bindings too. Cold imports happen
    only outside capture.
    """
    global _DISPATCH_STATE
    normalized_op_id = normalize_name(op_id, "op id")
    if implementation is not None:
        implementation = normalize_name(implementation, "implementation")
    resolved_target = None
    if implementation != "torch_reference":
        resolved_target = detect_target(target)
    snapshot = _config._get_config_snapshot() if implementation is None else _config._CONFIG
    state = _DISPATCH_STATE
    if state[0] is not snapshot:
        state = (snapshot, {})
        if not torch.compiler.is_compiling():
            _DISPATCH_STATE = state
    cache = state[1]
    target_key = resolved_target.key if resolved_target is not None else None
    selection = (target_key, normalized_op_id, implementation)
    if selection not in cache:
        resolved_implementation = implementation
        if resolved_implementation is None:
            resolved_implementation = resolve_implementation(normalized_op_id, resolved_target, snapshot)
        if resolved_implementation == "torch_reference":
            provider_package = "mojo_opset.kernels.torch_reference"
            provider_target = None
        else:
            provider_package = _provider_package(resolved_target, resolved_implementation)
            provider_target = target_key
        key = (provider_package, provider_target, normalized_op_id)
        if key not in _BINDINGS:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "Preload Mojo wrappers before compilation: call mojo_opset.preload "
                    "with the operator names and selection options, or run an eager warmup."
                )
            _BINDINGS[key] = _load_kernel_binding(*key)
        entry = (resolved_implementation, _BINDINGS[key])
        # AutogradFunction capture cannot mutate global Python dictionaries.
        # A warm wrapper may still be selected through a new config/API choice.
        if not torch.compiler.is_compiling():
            cache[selection] = entry
    else:
        entry = cache[selection]
    resolved_implementation, binding = entry
    if require_backward and binding[1] is None:
        raise RuntimeError(
            f"Selected implementation {resolved_implementation!r} for op={normalized_op_id!r}, "
            f"target={target_key!r} has no backward wrapper."
        )
    return binding


def preload(*op_ids: str, implementation: Optional[str] = None, target: Optional[TargetLike] = None) -> None:
    """Load selected wrappers before graph capture, without executing kernels.

    Eager calls also populate the binding cache. This does not freeze subsequent
    config or API selection; preload every binding a compiled call may select.
    """
    if not op_ids:
        raise ValueError("preload requires at least one operator name")
    for op_id in op_ids:
        load_impl(op_id, implementation, target=target)
