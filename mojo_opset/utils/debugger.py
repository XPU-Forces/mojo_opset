import functools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

_PREFIX = "[MojoDebug]"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    layer_idx: str  # int-string, "*", or "none"
    op_name: str
    raw: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _restore_env(key: str, old_value: Optional[str]):
    if old_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old_value


def _deep_clone(value):
    """Recursively clone tensors; pass through non-tensors unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        cloned = [_deep_clone(v) for v in value]
        if hasattr(type(value), "_fields"):  # namedtuple
            return type(value)(*cloned)
        return type(value)(cloned)
    if isinstance(value, dict):
        return {k: _deep_clone(v) for k, v in value.items()}
    return value


def _to_cpu(value):
    """Recursively move tensors to CPU for serialisation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        converted = [_to_cpu(v) for v in value]
        if hasattr(type(value), "_fields"):  # namedtuple
            return type(value)(*converted)
        return type(value)(converted)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _infer_core_cls_from_mro(module):
    """Walk MRO to find the class whose direct parent is MojoOperator."""
    from mojo_opset.core.operator import MojoOperator

    for cls in type(module).__mro__:
        if MojoOperator in cls.__bases__:
            return cls
    return None


def _infer_device(module) -> torch.device:
    """Determine the device of a module from its parameters or buffers.

    For stateless modules (no parameters/buffers), fall back to the device
    recorded from pre-hook inputs, then to CPU.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    pre_inputs = getattr(module, "_debug_pre_inputs", None)
    if pre_inputs is not None:
        for inp in pre_inputs:
            if isinstance(inp, torch.Tensor):
                return inp.device
    return torch.device("cpu")


def _parse_rules(rule_str: str) -> List[_Rule]:
    """Parse ``'layer_idx:op_name;...'`` into a list of :class:`_Rule`.

    Invalid segments are skipped with a warning.
    """
    if not rule_str or not rule_str.strip():
        return []

    rules: List[_Rule] = []
    for segment in rule_str.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.warning(
                f"{_PREFIX} Ignoring malformed rule: '{segment}'. "
                f"Expected format 'layer_idx:op_name'."
            )
            continue
        rules.append(
            _Rule(
                layer_idx=parts[0].strip(),
                op_name=parts[1].strip(),
                raw=segment,
            )
        )
    return rules


def _match_single(
    layer_idx: Optional[int],
    op_name: str,
    module,
    rule: _Rule,
) -> bool:
    # --- layer_idx ---
    if rule.layer_idx == "*":
        pass
    elif rule.layer_idx == "none":
        if layer_idx is not None:
            return False
    else:
        try:
            if layer_idx != int(rule.layer_idx):
                return False
        except (ValueError, TypeError):
            return False

    # --- op_name ---
    if rule.op_name.startswith("Mojo"):
        core_cls = getattr(module, "_debug_core_cls", None) or _infer_core_cls_from_mro(module)
        if core_cls is None or core_cls.__name__ != rule.op_name:
            return False
    else:
        if op_name != rule.op_name and not op_name.endswith("." + rule.op_name):
            return False

    return True


def _match(
    layer_idx: Optional[int],
    op_name: str,
    module,
    rules: List[_Rule],
) -> Optional[_Rule]:
    for rule in rules:
        if _match_single(layer_idx, op_name, module, rule):
            return rule
    return None


# ---------------------------------------------------------------------------
# MojoDebugger
# ---------------------------------------------------------------------------


class MojoDebugger:
    """Lightweight debug controller for mojo_opset operators.

    Typical usage::

        MojoDebugger.enable()            # before model construction
        model = build_model(config)
        model.load_state_dict(ckpt)

        dbg = MojoDebugger()
        dbg.attach(model)
        dbg.set_compare("5:input_layernorm")
        output = model(input_ids)
        dbg.detach()
    """

    _enabled: bool = False
    _original_new = None

    # ------------------------------------------------------------------
    # Class-level: enable / disable
    # ------------------------------------------------------------------

    @classmethod
    def enable(cls):
        """Patch ``MojoOperator.__new__`` to capture construction args.

        Must be called **before** model construction.  Idempotent.
        """
        if cls._enabled:
            return

        from mojo_opset.core.operator import MojoOperator

        original_new = MojoOperator.__new__

        @functools.wraps(original_new)
        def _debug_new(klass, *args, **kwargs):
            instance = original_new(klass, *args, **kwargs)
            if MojoOperator in klass.__bases__:
                instance._debug_core_cls = klass
                instance._debug_init_args = (args, kwargs)
            return instance

        MojoOperator.__new__ = _debug_new
        cls._original_new = original_new
        cls._enabled = True
        logger.info_rank0(f"{_PREFIX} Debug mode enabled. MojoOperator.__new__ patched.")

    @classmethod
    def disable(cls):
        """Restore original ``MojoOperator.__new__``."""
        if not cls._enabled:
            return
        from mojo_opset.core.operator import MojoOperator

        if cls._original_new is not None:
            MojoOperator.__new__ = cls._original_new
        cls._enabled = False
        cls._original_new = None

    # ------------------------------------------------------------------
    # Instance-level
    # ------------------------------------------------------------------

    _VALID_COMPARE_MODES = ("observe", "replace")

    def __init__(
        self,
        dump_dir: Optional[str] = None,
        max_steps: Optional[int] = None,
        compare_mode: Optional[str] = None,
    ):
        self._dump_dir = (
            dump_dir
            or os.environ.get("MOJO_DEBUG_DUMP_DIR")
            or "./mojo_debug_dump"
        )

        self._max_steps = max_steps
        if self._max_steps is None:
            env_max = os.environ.get("MOJO_DEBUG_MAX_STEPS")
            if env_max is not None:
                try:
                    self._max_steps = int(env_max)
                except ValueError:
                    logger.warning(f"{_PREFIX} Invalid MOJO_DEBUG_MAX_STEPS='{env_max}', ignored.")

        if compare_mode is None:
            compare_mode = os.environ.get("MOJO_DEBUG_COMPARE_MODE", "observe")
        if compare_mode not in self._VALID_COMPARE_MODES:
            logger.warning(
                f"{_PREFIX} Invalid compare_mode='{compare_mode}', falling back to 'observe'."
            )
            compare_mode = "observe"
        self._compare_mode = compare_mode

        self._model: Optional[torch.nn.Module] = None
        self._hook_handles: List = []
        self._attached = False

        # Rule caches (env-var driven)
        self._env_cache: Dict[str, str] = {}
        self._parsed_cache: Dict[str, List[_Rule]] = {}

        # API-set rules take precedence over env vars
        self._api_rules: Dict[str, Optional[List[_Rule]]] = {
            "compare": None,
            "dump": None,
        }

        self._step_counters: Dict[Tuple, int] = defaultdict(int)

        # (layer_idx, op_name) -> module  — built during attach
        self._op_map: Dict[Tuple[Optional[int], str], Any] = {}

    # ------------------------------------------------------------------
    # attach / detach
    # ------------------------------------------------------------------

    def attach(self, model: torch.nn.Module):
        """Register debug hooks on all ``MojoOperator`` modules."""
        from mojo_opset.core.operator import MojoOperator

        if self._attached:
            self.detach()
            logger.warning(f"{_PREFIX} Previous debug session detached before re-attach.")

        self._model = model
        self._hook_handles = []
        self._step_counters.clear()
        self._op_map.clear()

        if not self._enabled:
            logger.warning(
                f"{_PREFIX} MojoDebugger.enable() was not called before model construction. "
                f"Compare may be limited for ops whose backend overrides __init__."
            )

        self._propagate_layer_info(model)

        for name, module in model.named_modules():
            if isinstance(module, MojoOperator):
                pre_h = module.register_forward_pre_hook(self._make_pre_hook())
                post_h = module.register_forward_hook(self._make_hook())
                self._hook_handles.extend([pre_h, post_h])
                layer_idx = getattr(module, "_debug_layer_idx", None)
                op_name = getattr(module, "_debug_op_name", name)
                self._op_map[(layer_idx, op_name)] = module

        self._attached = True
        self._print_op_map()

    def detach(self):
        """Remove hooks, release shadows, clean up debug attributes."""
        from mojo_opset.core.operator import MojoOperator

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        if self._model is not None:
            for _, module in self._model.named_modules():
                if isinstance(module, MojoOperator):
                    for attr in (
                        "_debug_layer_idx",
                        "_debug_op_name",
                        "_debug_torch_ref",
                        "_debug_ref_forward",
                        "_debug_torch_ref_ready",
                        "_debug_pre_inputs",
                    ):
                        try:
                            delattr(module, attr)
                        except AttributeError:
                            pass

        self._model = None
        self._attached = False
        self._op_map.clear()
        self._step_counters.clear()
        self._env_cache.clear()
        self._parsed_cache.clear()
        self._api_rules = {"compare": None, "dump": None}

    # ------------------------------------------------------------------
    # Dynamic rule setters
    # ------------------------------------------------------------------

    def set_compare(self, rules: str):
        """Set compare rules dynamically.  ``""`` clears."""
        parsed = _parse_rules(rules)
        self._api_rules["compare"] = parsed
        if self._attached and parsed:
            self._validate_rules(parsed, "compare")

    def set_dump(self, rules: str):
        """Set dump rules dynamically.  ``""`` clears."""
        parsed = _parse_rules(rules)
        self._api_rules["dump"] = parsed
        if self._attached and parsed:
            self._validate_rules(parsed, "dump")

    def set_compare_mode(self, mode: str):
        """Switch compare mode at runtime.  ``"observe"`` or ``"replace"``."""
        if mode not in self._VALID_COMPARE_MODES:
            raise ValueError(
                f"Unknown compare_mode '{mode}'. Must be one of {self._VALID_COMPARE_MODES}."
            )
        self._compare_mode = mode

    def clear_rules(self):
        self._api_rules = {"compare": None, "dump": None}

    def set_dump_dir(self, path: str):
        self._dump_dir = path

    def set_max_steps(self, n: int):
        self._max_steps = n

    def reset_step_counters(self):
        self._step_counters.clear()

    # ------------------------------------------------------------------
    # Layer-info propagation
    # ------------------------------------------------------------------

    def _propagate_layer_info(self, model: torch.nn.Module):
        from mojo_opset.core.operator import MojoOperator

        found_any_layer_idx = False

        def _walk(module, layer_idx=None, layer_path="", current_path=""):
            nonlocal found_any_layer_idx
            if hasattr(module, "layer_idx"):
                layer_idx = module.layer_idx
                layer_path = current_path
                found_any_layer_idx = True

            if isinstance(module, MojoOperator):
                module._debug_layer_idx = layer_idx
                if layer_path and current_path.startswith(layer_path + "."):
                    module._debug_op_name = current_path[len(layer_path) + 1 :]
                else:
                    module._debug_op_name = (
                        current_path.split(".")[-1] if current_path else ""
                    )

            for name, child in module.named_children():
                child_path = f"{current_path}.{name}" if current_path else name
                _walk(child, layer_idx, layer_path, child_path)

        _walk(model)

        if not found_any_layer_idx:
            logger.warning(
                f"{_PREFIX} No module with 'layer_idx' attribute found. "
                f"All ops have _debug_layer_idx=None. Use 'none:op_name' rules."
            )

    # ------------------------------------------------------------------
    # Op-map printing
    # ------------------------------------------------------------------

    def _print_op_map(self):
        if not self._op_map:
            return

        layer_ops: Dict[Optional[int], Dict[str, str]] = defaultdict(dict)
        for (layer_idx, op_name), module in self._op_map.items():
            core_cls = getattr(module, "_debug_core_cls", None) or _infer_core_cls_from_mro(module)
            cls_name = core_cls.__name__ if core_cls else type(module).__name__
            layer_ops[layer_idx][op_name] = cls_name

        lines = [f"{_PREFIX} Attached. {len(self._op_map)} MojoOperator instances discovered:"]

        layer_indices = sorted(k for k in layer_ops if k is not None)
        if layer_indices:
            # Group contiguous layers with identical op-name sets
            groups: List[Tuple[int, int, Dict[str, str]]] = []
            grp_start = layer_indices[0]
            grp_ops = layer_ops[layer_indices[0]]
            for idx in layer_indices[1:]:
                if layer_ops[idx] == grp_ops:
                    continue
                groups.append((grp_start, idx - 1, grp_ops))
                grp_start = idx
                grp_ops = layer_ops[idx]
            groups.append((grp_start, layer_indices[-1], grp_ops))

            for start, end, ops in groups:
                rng = str(start) if start == end else f"{start}-{end}"
                op_strs = [f"{n} ({c})" for n, c in sorted(ops.items())]
                lines.append(f"  layer {rng}: {', '.join(op_strs)}")

        if None in layer_ops:
            op_strs = [f"{n} ({c})" for n, c in sorted(layer_ops[None].items())]
            lines.append(f"  global: {', '.join(op_strs)}")

        lines.append(
            f'{_PREFIX} Rule format: "layer_idx:op_name", '
            f'e.g. "5:input_layernorm" or "*:self_attn.rope"'
        )
        logger.info_rank0("\n".join(lines))

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------

    def _get_active_rules(self, rule_type: str) -> List[_Rule]:
        api = self._api_rules.get(rule_type)
        if api is not None:
            return api

        env_key = f"MOJO_DEBUG_{rule_type.upper()}"
        current_val = os.environ.get(env_key, "")
        cached_val = self._env_cache.get(env_key)

        if current_val != cached_val:
            self._env_cache[env_key] = current_val
            parsed = _parse_rules(current_val)
            self._parsed_cache[env_key] = parsed
            if parsed and self._attached:
                self._validate_rules(parsed, rule_type)

        return self._parsed_cache.get(env_key, [])

    def _validate_rules(self, rules: List[_Rule], rule_type: str):
        for rule in rules:
            matched_any = False
            for (layer_idx, op_name), module in self._op_map.items():
                if _match_single(layer_idx, op_name, module, rule):
                    matched_any = True
                    break
            if not matched_any:
                logger.warning(
                    f"{_PREFIX} {rule_type} rule '{rule.raw}' did not match any "
                    f"MojoOperator. Check attach() output for available targets."
                )

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_pre_hook(self):
        """Capture a snapshot of inputs *before* forward to handle in-place ops."""
        debugger = self

        def pre_hook(module, inputs):
            try:
                compare_rules = debugger._get_active_rules("compare")
                dump_rules = debugger._get_active_rules("dump")
                if not compare_rules and not dump_rules:
                    return

                layer_idx = getattr(module, "_debug_layer_idx", None)
                op_name = getattr(module, "_debug_op_name", "")

                need_snapshot = (
                    _match(layer_idx, op_name, module, dump_rules) is not None
                    or _match(layer_idx, op_name, module, compare_rules) is not None
                )
                if need_snapshot:
                    module._debug_pre_inputs = _deep_clone(inputs)
            except Exception:
                pass

        return pre_hook

    def _make_hook(self):
        debugger = self

        def hook(module, inputs, output):
            try:
                compare_rules = debugger._get_active_rules("compare")
                dump_rules = debugger._get_active_rules("dump")

                if not compare_rules and not dump_rules:
                    return

                layer_idx = getattr(module, "_debug_layer_idx", None)
                op_name = getattr(module, "_debug_op_name", "")

                matched_dump = _match(layer_idx, op_name, module, dump_rules)
                matched_compare = _match(layer_idx, op_name, module, compare_rules)

                if matched_dump is None and matched_compare is None:
                    return

                safe_inputs = getattr(module, "_debug_pre_inputs", inputs)

                if matched_dump is not None:
                    debugger._do_dump(layer_idx, op_name, module, safe_inputs, output)

                if matched_compare is not None:
                    ref_output = debugger._do_compare(
                        layer_idx, op_name, module, safe_inputs, output,
                    )
                    if debugger._compare_mode == "replace" and ref_output is not None:
                        return ref_output

            except Exception as e:
                logger.warning_once(
                    f"{_PREFIX} Unexpected error in hook for '{op_name}': {e}. "
                    f"Debug skipped, inference unaffected."
                )
            finally:
                module._debug_pre_inputs = None

        return hook

    # ------------------------------------------------------------------
    # Tag helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tag(layer_idx: Optional[int], op_name: str) -> str:
        if layer_idx is not None:
            return f"layer{layer_idx}.{op_name}"
        return f"global.{op_name}"

    # ------------------------------------------------------------------
    # Dump
    # ------------------------------------------------------------------

    def _do_dump(self, layer_idx, op_name, module, inputs, output):
        counter_key = ("dump", layer_idx, op_name)
        step = self._step_counters[counter_key]
        if self._max_steps is not None and step >= self._max_steps:
            return
        self._step_counters[counter_key] = step + 1

        tag = self._make_tag(layer_idx, op_name)
        self._log_tensor_stats(tag, step, output, "output")

        dump_dir = self._get_dump_dir()
        if dump_dir is not None:
            try:
                out_path = os.path.join(dump_dir, f"{tag}_step{step}_output.pt")
                torch.save(_to_cpu(output), out_path)
                in_path = os.path.join(dump_dir, f"{tag}_step{step}_input.pt")
                torch.save(_to_cpu(inputs), in_path)
                logger.info_rank0(f"{_PREFIX} {tag} step={step} | saved to {dump_dir}")
            except (OSError, IOError) as e:
                logger.warning_once(f"{_PREFIX} Failed to save dump for {tag}: {e}")

    def _get_dump_dir(self) -> Optional[str]:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        dump_dir = os.path.join(self._dump_dir, f"rank{rank}")
        try:
            os.makedirs(dump_dir, exist_ok=True)
            return dump_dir
        except OSError as e:
            logger.warning_once(f"{_PREFIX} Cannot create dump directory '{dump_dir}': {e}")
            return None

    def _log_tensor_stats(self, tag: str, step: int, value, prefix: str = "output"):
        if isinstance(value, torch.Tensor):
            t = value.detach().float()
            nan_flag = " HAS_NAN" if torch.isnan(t).any() else ""
            inf_flag = " HAS_INF" if torch.isinf(t).any() else ""
            logger.info_rank0(
                f"{_PREFIX} {tag} step={step} | {prefix} "
                f"shape={tuple(value.shape)} dtype={value.dtype} "
                f"mean={t.mean().item():.4g} std={t.std().item():.4g} "
                f"min={t.min().item():.4g} max={t.max().item():.4g}"
                f"{nan_flag}{inf_flag}"
            )
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                self._log_tensor_stats(tag, step, v, prefix=f"{prefix}[{i}]")
        else:
            logger.debug_rank0(
                f"{_PREFIX} {tag} step={step} | {prefix} "
                f"type={type(value).__name__} (non-tensor, skipped)"
            )

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def _do_compare(self, layer_idx, op_name, module, inputs, output):
        """Run torch reference forward and report diff.

        Returns the reference output so the caller can decide whether to
        replace the accelerator output (``replace`` mode).  Returns ``None``
        on any failure or when the step limit has been reached.
        """
        counter_key = ("cmp", layer_idx, op_name)
        step = self._step_counters[counter_key]
        if self._max_steps is not None and step >= self._max_steps:
            return None
        self._step_counters[counter_key] = step + 1

        tag = self._make_tag(layer_idx, op_name)

        try:
            self._ensure_torch_ref(module)
        except Exception as e:
            logger.warning_once(
                f"{_PREFIX} Cannot build torch ref for {tag}: {e}. Compare skipped."
            )
            return None

        ref_inputs = _deep_clone(inputs)

        try:
            with torch.no_grad():
                if module._debug_torch_ref is not None:
                    ref_output = module._debug_torch_ref(*ref_inputs)
                else:
                    ref_output = module._debug_ref_forward(module, *ref_inputs)
        except Exception as e:
            logger.warning_once(
                f"{_PREFIX} Torch ref forward failed for {tag}: {e}. Compare skipped."
            )
            return None

        self._compare_and_report(tag, step, output, ref_output)
        return ref_output

    def _compare_and_report(
        self,
        tag: str,
        step: int,
        output,
        ref_output,
        prefix: str = "",
    ):
        if isinstance(output, torch.Tensor) and isinstance(ref_output, torch.Tensor):
            if output.shape != ref_output.shape:
                logger.warning_rank0(
                    f"{_PREFIX} {tag} step={step} | {prefix}SHAPE_MISMATCH "
                    f"got {tuple(output.shape)} vs ref {tuple(ref_output.shape)}"
                )
                return

            out_f = output.detach().float()
            ref_f = ref_output.detach().float()
            abs_diff = (out_f - ref_f).abs()
            max_abs_diff = abs_diff.max().item()
            max_rel_diff = (abs_diff / ref_f.abs().clamp(min=1e-12)).max().item()
            cos_sim = F.cosine_similarity(
                out_f.flatten().unsqueeze(0),
                ref_f.flatten().unsqueeze(0),
            ).item()

            logger.info_rank0(
                f"{_PREFIX} {tag} step={step} | {prefix}"
                f"max_abs_diff={max_abs_diff:.4g} "
                f"max_rel_diff={max_rel_diff:.4g} "
                f"cos_sim={cos_sim:.6f}"
            )

        elif isinstance(output, (tuple, list)) and isinstance(ref_output, (tuple, list)):
            if len(output) != len(ref_output):
                logger.warning_rank0(
                    f"{_PREFIX} {tag} step={step} | {prefix}LENGTH_MISMATCH "
                    f"got {len(output)} vs ref {len(ref_output)}"
                )
                return
            for i, (o, r) in enumerate(zip(output, ref_output)):
                self._compare_and_report(tag, step, o, r, prefix=f"[{i}] ")

        else:
            logger.debug_rank0(
                f"{_PREFIX} {tag} step={step} | {prefix}non-tensor output, skipped"
            )

    # ------------------------------------------------------------------
    # Lazy shadow construction
    # ------------------------------------------------------------------

    def _ensure_torch_ref(self, module):
        if getattr(module, "_debug_torch_ref_ready", False):
            return

        from mojo_opset.core.operator import MojoOperator  # noqa: F811

        core_cls = getattr(module, "_debug_core_cls", None)
        backend_cls = type(module)

        if core_cls is None:
            core_cls = _infer_core_cls_from_mro(module)
            if core_cls is None:
                raise RuntimeError(
                    f"Cannot determine core op class for {backend_cls.__name__}."
                )

        if "__init__" not in backend_cls.__dict__:
            # Backend shares __init__ with core — weights are in standard format.
            module._debug_torch_ref = None
            module._debug_ref_forward = core_cls.forward
        else:
            init_args = getattr(module, "_debug_init_args", None)
            if init_args is None:
                logger.warning_once(
                    f"{_PREFIX} No init args recorded for {backend_cls.__name__}. "
                    f"Falling back to direct core forward "
                    f"(may be inaccurate if backend transforms weights)."
                )
                module._debug_torch_ref = None
                module._debug_ref_forward = core_cls.forward
            else:
                args, kwargs = init_args
                old_backend = os.environ.get("MOJO_BACKEND")
                os.environ["MOJO_BACKEND"] = "torch"
                try:
                    shadow = core_cls(*args, **kwargs)
                finally:
                    _restore_env("MOJO_BACKEND", old_backend)

                result = shadow.load_state_dict(module.state_dict(), strict=False)
                if result.missing_keys:
                    logger.warning(
                        f"{_PREFIX} Shadow missing keys: {result.missing_keys}"
                    )
                if result.unexpected_keys:
                    logger.warning(
                        f"{_PREFIX} Shadow unexpected keys: {result.unexpected_keys}"
                    )

                device = _infer_device(module)
                shadow = shadow.to(device)
                shadow.eval()

                object.__setattr__(module, "_debug_torch_ref", shadow)
                module._debug_ref_forward = None

        module._debug_torch_ref_ready = True
