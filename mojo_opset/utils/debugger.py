import functools
import math
import os

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

_PREFIX = "[MojoDebug]"
_DEFAULT_IPD_PROBES = 64
_DEFAULT_IPD_SEED = 20260423
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    layer_idx: str  # int-string, "*", or "none"
    op_name: str
    raw: str


@dataclass
class DebugAnalysisContext:
    """Context passed to compare analysis hooks.

    ``ref_output`` is the torch reference output. ``output`` is the backend
    output observed by the forward hook. Hooks should treat tensors as
    read-only; the debugger's ``replace`` mode remains the only path that
    intentionally changes model execution.
    """

    tag: str
    step: int
    layer_idx: Optional[int]
    op_name: str
    module: Any
    inputs: Any
    output: Any
    ref_output: Any
    prefix: str = ""


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
# Precision guard metrics
# ---------------------------------------------------------------------------


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float()


def _flatten_last_dim(value: torch.Tensor) -> torch.Tensor:
    """Flatten all leading dimensions and keep the last dimension as vector dim."""
    value = _as_float_tensor(value)
    if value.dim() == 0:
        return value.reshape(1, 1)
    if value.shape[-1] == 0:
        return value.reshape(-1, 0)
    return value.reshape(-1, value.shape[-1])


def _safe_quantile(value: torch.Tensor, q: float) -> float:
    value = value.detach().float().reshape(-1)
    if value.numel() == 0:
        return float("nan")
    if value.numel() == 1:
        return value.item()
    return value.quantile(q).item()


def _make_generator(device: torch.device, seed: Optional[int] = None) -> torch.Generator:
    if seed is None:
        seed = _DEFAULT_IPD_SEED
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _normalise_rows(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp(min=_EPS)


def _random_probes(
    n_probes: int,
    dim: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if dim <= 0 or n_probes <= 0:
        return torch.empty((0, max(dim, 0)), device=device, dtype=torch.float32)
    generator = _make_generator(device, seed)
    probes = torch.randn(
        n_probes,
        dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return _normalise_rows(probes)


def _sample_probe_rows(
    probes: torch.Tensor,
    n_probes: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if probes.shape[0] <= n_probes:
        return probes
    generator = _make_generator(probes.device, seed)
    idx = torch.randperm(probes.shape[0], device=probes.device, generator=generator)
    return probes.index_select(0, idx[:n_probes])


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning_once(f"{_PREFIX} Invalid {name}='{raw}', using {default}.")
        return default
    return value if value > 0 else default


def compute_snr_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    """D-class numeric-signal metrics: SNR, RelL2, MaxAbsErr, CosSim.

    This mirrors the D-class metric used in ``xperf_gpt_triton`` operator
    reports. ``reference`` is the BF16/torch value, and ``candidate`` is the
    backend or quantized value being checked.
    """
    ref = _as_float_tensor(reference)
    cand = _as_float_tensor(candidate)
    diff = cand - ref

    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    diff_flat = diff.reshape(-1)

    signal_power = (ref_flat ** 2).sum().item()
    noise_power = (diff_flat ** 2).sum().item()
    snr_db = 10 * math.log10(max(signal_power / max(noise_power, 1e-30), 1e-30))

    ref_norm = ref_flat.norm().clamp(min=_EPS)
    cand_norm = cand_flat.norm().clamp(min=_EPS)
    rel_l2 = (diff_flat.norm() / ref_norm).item()
    cos = ((ref_flat * cand_flat).sum() / (ref_norm * cand_norm)).clamp(-1.0, 1.0).item()
    if math.isnan(cos):
        cos = 1.0 if noise_power < 1e-30 else 0.0

    if ref.dim() > 1 and ref.shape[-1] > 0:
        ref_2d = _flatten_last_dim(ref)
        cand_2d = _flatten_last_dim(cand)
        per_vec_cos = F.cosine_similarity(ref_2d, cand_2d, dim=-1)
        cos_mean = per_vec_cos.mean().item()
        cos_min = per_vec_cos.min().item()
    else:
        cos_mean = cos
        cos_min = cos

    abs_diff = diff_flat.abs()
    rel_diff = abs_diff / ref_flat.abs().clamp(min=_EPS)
    max_abs = abs_diff.max().item() if abs_diff.numel() else 0.0
    max_rel = rel_diff.max().item() if rel_diff.numel() else 0.0

    return {
        "snr_db": snr_db,
        "rel_l2": rel_l2,
        "max_abs_err": max_abs,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "cos_sim": cos,
        "cos_mean": cos_mean,
        "cos_min": cos_min,
        "num_elements": int(ref.numel()),
    }


def compute_probability_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    top_k: int = 8,
    inputs_are_logits: bool = True,
) -> Dict[str, float]:
    """A-class probability metrics: KL/JS divergence and Top-K match.

    Use ``inputs_are_logits=False`` when both inputs are already probability
    distributions. With the default, the last dimension is softmaxed first.
    """
    ref = _flatten_last_dim(reference)
    cand = _flatten_last_dim(candidate)
    if ref.shape[-1] == 0:
        return {
            "kl_mean": float("nan"),
            "kl_max": float("nan"),
            "js_mean": float("nan"),
            "js_max": float("nan"),
            "topk_match_mean": float("nan"),
            "num_vectors": int(ref.shape[0]),
        }

    if inputs_are_logits:
        p = F.softmax(ref, dim=-1).clamp(min=_EPS)
        q = F.softmax(cand, dim=-1).clamp(min=_EPS)
    else:
        p = ref.clamp(min=_EPS)
        q = cand.clamp(min=_EPS)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        q = q / q.sum(dim=-1, keepdim=True).clamp(min=_EPS)

    m = 0.5 * (p + q)
    kl_pq = (p * (p / q).log()).sum(dim=-1)
    kl_qp = (q * (q / p).log()).sum(dim=-1)
    js = 0.5 * (p * (p / m).log()).sum(dim=-1) + 0.5 * (q * (q / m).log()).sum(dim=-1)

    k = min(max(top_k, 1), ref.shape[-1])
    ref_topk = ref.topk(k=k, dim=-1).indices
    cand_topk = cand.topk(k=k, dim=-1).indices
    match = (ref_topk.unsqueeze(-1) == cand_topk.unsqueeze(-2)).any(dim=-1).float()
    match_rate = match.mean(dim=-1)

    return {
        "kl_mean": kl_pq.mean().item(),
        "kl_max": kl_pq.max().item(),
        "kl_reverse_mean": kl_qp.mean().item(),
        "js_mean": js.mean().item(),
        "js_max": js.max().item(),
        "topk_match_mean": match_rate.mean().item(),
        "num_vectors": int(ref.shape[0]),
    }


def compute_angular_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    """B-class direction metrics: angular error, cosine and relative norm error."""
    ref = _flatten_last_dim(reference)
    cand = _flatten_last_dim(candidate)
    if ref.shape[-1] == 0:
        return {
            "cos_mean": float("nan"),
            "cos_min": float("nan"),
            "angular_err_deg_mean": float("nan"),
            "angular_err_deg_max": float("nan"),
            "angular_err_deg_p95": float("nan"),
            "rel_norm_err_mean": float("nan"),
            "rel_norm_err_max": float("nan"),
            "num_vectors": int(ref.shape[0]),
        }

    ref_norm = ref.norm(dim=-1).clamp(min=_EPS)
    cand_norm = cand.norm(dim=-1).clamp(min=_EPS)
    cos = ((ref * cand).sum(dim=-1) / (ref_norm * cand_norm)).clamp(-1.0, 1.0)
    angular = torch.acos(cos)
    rel_norm = (cand_norm - ref_norm).abs() / ref_norm

    return {
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "cos_p05": _safe_quantile(cos, 0.05),
        "angular_err_deg_mean": math.degrees(angular.mean().item()),
        "angular_err_deg_max": math.degrees(angular.max().item()),
        "angular_err_deg_p95": math.degrees(_safe_quantile(angular, 0.95)),
        "rel_norm_err_mean": rel_norm.mean().item(),
        "rel_norm_err_max": rel_norm.max().item(),
        "rel_norm_err_p95": _safe_quantile(rel_norm, 0.95),
        "num_vectors": int(ref.shape[0]),
    }


def compute_ipd_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    probes: Optional[torch.Tensor] = None,
    n_probes: int = _DEFAULT_IPD_PROBES,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """C-class Inner Product Distortion along the last dimension.

    IPD is the average relative drift of probe inner products:

    ``abs(<probe, candidate> - <probe, reference>) / (abs(<probe, reference>) + eps)``.

    If ``probes`` is omitted, deterministic Gaussian probes are generated with
    a local ``torch.Generator`` so metrics do not perturb the model RNG state.
    """
    ref = _flatten_last_dim(reference)
    cand = _flatten_last_dim(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"IPD shape mismatch: {tuple(cand.shape)} vs {tuple(ref.shape)}")
    dim = ref.shape[-1]
    if dim == 0 or ref.shape[0] == 0:
        return {
            "ipd_mean": float("nan"),
            "ipd_p95": float("nan"),
            "ipd_max": float("nan"),
            "cos_mean": float("nan"),
            "num_vectors": int(ref.shape[0]),
            "num_probes": 0,
        }

    if probes is None:
        probe_2d = _random_probes(n_probes, dim, ref.device, seed=seed)
    else:
        probe_2d = _flatten_last_dim(probes).to(device=ref.device, dtype=torch.float32)
        if probe_2d.shape[-1] != dim:
            raise ValueError(
                f"Probe dim mismatch: got {probe_2d.shape[-1]}, expected {dim}."
            )
        probe_2d = _normalise_rows(probe_2d)
        probe_2d = _sample_probe_rows(probe_2d, n_probes, seed=seed)

    ip_ref = ref @ probe_2d.T
    ip_cand = cand @ probe_2d.T
    rel_err = (ip_cand - ip_ref).abs() / ip_ref.abs().clamp(min=_EPS)
    ipd_per_vec = rel_err.mean(dim=-1)
    cos = F.cosine_similarity(ref, cand, dim=-1)

    return {
        "ipd_mean": ipd_per_vec.mean().item(),
        "ipd_p95": _safe_quantile(ipd_per_vec, 0.95),
        "ipd_max": ipd_per_vec.max().item(),
        "cos_mean": cos.mean().item(),
        "num_vectors": int(ref.shape[0]),
        "num_probes": int(probe_2d.shape[0]),
    }


def compute_ipd_consumer_aware(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    consumer: Optional[torch.Tensor] = None,
    n_probes: int = _DEFAULT_IPD_PROBES,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """IPD along the last dimension, using real consumer vectors as probes.

    Examples:
    - Q uses real K rows as probes.
    - K uses real Q rows as probes.
    - O-proj input or FC intermediate can use real weight rows as probes.
    Falls back to deterministic random probes if the consumer shape is absent
    or incompatible.
    """
    ref = _flatten_last_dim(reference)
    probes = None
    if consumer is not None:
        consumer_2d = _flatten_last_dim(consumer)
        if consumer_2d.shape[-1] == ref.shape[-1] and consumer_2d.shape[0] > 0:
            probes = consumer_2d
    return compute_ipd_metrics(
        reference,
        candidate,
        probes=probes,
        n_probes=n_probes,
        seed=seed,
    )


def compute_ipd_v_seqdim(
    reference_v: torch.Tensor,
    candidate_v: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    n_probes: int = _DEFAULT_IPD_PROBES,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Consumer-aware IPD for V along the sequence dimension.

    For ``P @ V``, the GEMM K-axis is ``seq_len`` rather than ``head_dim``.
    This function transposes V from ``[seq_len, heads, head_dim]`` into
    vectors of length ``seq_len`` and uses rows from the real probability
    matrix P as probes when available.
    """
    ref = _as_float_tensor(reference_v)
    cand = _as_float_tensor(candidate_v)
    if ref.dim() == 3:
        seq_len, heads, head_dim = ref.shape
        ref_vec = ref.permute(1, 2, 0).reshape(heads * head_dim, seq_len)
        cand_vec = cand.permute(1, 2, 0).reshape(heads * head_dim, seq_len)
    else:
        ref_vec = _flatten_last_dim(ref)
        cand_vec = _flatten_last_dim(cand)
        seq_len = ref_vec.shape[-1]

    probes = None
    if probs is not None:
        p_rows = _flatten_last_dim(probs)
        if p_rows.shape[-1] == seq_len and p_rows.shape[0] > 0:
            probes = p_rows

    return compute_ipd_metrics(
        ref_vec,
        cand_vec,
        probes=probes,
        n_probes=n_probes,
        seed=seed,
    )


def infer_debug_metric_type(name: str) -> str:
    """Infer metric semantic type from an operator/tag name.

    Returns ``A`` probability/routing, ``B`` direction/norm, ``C`` inner-product
    sensitive, or ``D`` generic numeric signal.
    """
    lowered = (name or "").lower()
    if any(token in lowered for token in ("gate", "router", "softmax", "prob")):
        return "A"
    if any(
        token in lowered
        for token in (
            "rmsnorm",
            "rms_norm",
            "layernorm",
            "layer_norm",
            "query_norm",
            "key_norm",
            "norm",
        )
    ):
        return "B"
    if any(
        token in lowered
        for token in (
            "q_proj",
            "k_proj",
            "v_proj",
            "qkv_proj",
            ".q",
            ".k",
            ".v",
            "query",
            "key",
            "value",
            "kv_cache",
        )
    ):
        return "C"
    return "D"


def _guard_status(metric_type: str, metrics: Dict[str, float]) -> Tuple[str, str, float]:
    """Classify metrics with thresholds from the quant-metrics report."""
    if metric_type == "A":
        value = metrics.get("kl_mean", float("nan"))
        if value < 0.001:
            return "safe", "kl_mean", value
        if value < 0.01:
            return "warn", "kl_mean", value
        return "red", "kl_mean", value

    if metric_type == "B":
        value = metrics.get("angular_err_deg_mean", float("nan"))
        if value < 5.5:
            return "safe", "angular_err_deg_mean", value
        if value < 6.5:
            return "warn", "angular_err_deg_mean", value
        return "red", "angular_err_deg_mean", value

    if metric_type == "C":
        value = metrics.get("ipd_p95", metrics.get("ipd_mean", float("nan")))
        if value < 0.50:
            return "safe", "ipd_p95", value
        if value < 1.00:
            return "warn", "ipd_p95", value
        return "red", "ipd_p95", value

    value = metrics.get("snr_db", float("nan"))
    if value > 25.0:
        return "safe", "snr_db", value
    if value >= 18.0:
        return "warn", "snr_db", value
    return "red", "snr_db", value


def compute_debug_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    metric_type: Optional[str] = None,
    name: str = "",
    n_probes: int = _DEFAULT_IPD_PROBES,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the semantic precision-guard metrics used by ``MojoDebugger``."""
    metric_type = metric_type or infer_debug_metric_type(name)
    if metric_type == "A":
        metrics = compute_probability_metrics(reference, candidate)
    elif metric_type == "B":
        metrics = compute_angular_metrics(reference, candidate)
    elif metric_type == "C":
        metrics = compute_ipd_metrics(reference, candidate, n_probes=n_probes, seed=seed)
    else:
        metric_type = "D"
        metrics = compute_snr_metrics(reference, candidate)

    if metric_type != "D":
        metrics.update(compute_snr_metrics(reference, candidate))

    guard, guard_metric, guard_value = _guard_status(metric_type, metrics)
    metrics["semantic_type"] = metric_type
    metrics["guard_status"] = guard
    metrics["guard_metric"] = guard_metric
    metrics["guard_value"] = guard_value
    return metrics


def _format_float(value: float, precision: int = 6) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{precision}g}"


def _format_compare_metrics(metrics: Dict[str, float]) -> str:
    sem_type = metrics.get("semantic_type", "D")
    common = (
        f"sem={sem_type} guard={metrics.get('guard_status', 'unknown')} "
        f"{metrics.get('guard_metric', 'metric')}={_format_float(metrics.get('guard_value'))} "
        f"max_abs_diff={_format_float(metrics.get('max_abs_diff'))} "
        f"max_rel_diff={_format_float(metrics.get('max_rel_diff'))}"
    )

    if sem_type == "A":
        detail = (
            f"kl_mean={_format_float(metrics.get('kl_mean'))} "
            f"kl_max={_format_float(metrics.get('kl_max'))} "
            f"js_mean={_format_float(metrics.get('js_mean'))} "
            f"topk_match={_format_float(metrics.get('topk_match_mean'))} "
            f"snr_db={_format_float(metrics.get('snr_db'))}"
        )
    elif sem_type == "B":
        detail = (
            f"cos_mean={_format_float(metrics.get('cos_mean'))} "
            f"cos_min={_format_float(metrics.get('cos_min'))} "
            f"angular_err_deg_mean={_format_float(metrics.get('angular_err_deg_mean'))} "
            f"angular_err_deg_p95={_format_float(metrics.get('angular_err_deg_p95'))} "
            f"rel_norm_err_mean={_format_float(metrics.get('rel_norm_err_mean'))}"
        )
    elif sem_type == "C":
        detail = (
            f"ipd_mean={_format_float(metrics.get('ipd_mean'))} "
            f"ipd_p95={_format_float(metrics.get('ipd_p95'))} "
            f"ipd_max={_format_float(metrics.get('ipd_max'))} "
            f"cos_mean={_format_float(metrics.get('cos_mean'))} "
            f"snr_db={_format_float(metrics.get('snr_db'))}"
        )
    else:
        detail = (
            f"snr_db={_format_float(metrics.get('snr_db'))} "
            f"rel_l2={_format_float(metrics.get('rel_l2'))} "
            f"cos_mean={_format_float(metrics.get('cos_mean'))} "
            f"cos_min={_format_float(metrics.get('cos_min'))}"
        )

    return f"{common} {detail}"


def default_precision_analysis_hook(context: DebugAnalysisContext) -> Dict[str, float]:
    """Default analysis hook used by ``MojoDebugger`` compare.

    The hook infers an operator semantic type and logs the corresponding
    precision guard metrics. Users can replace or extend this behavior by
    passing ``analysis_func`` to ``MojoDebugger`` or by calling
    ``add_analysis_hook``.
    """
    if not isinstance(context.output, torch.Tensor) or not isinstance(context.ref_output, torch.Tensor):
        return {}
    if context.output.shape != context.ref_output.shape:
        return {}

    n_probes = _parse_int_env("MOJO_DEBUG_IPD_PROBES", _DEFAULT_IPD_PROBES)
    seed = _parse_int_env("MOJO_DEBUG_IPD_SEED", _DEFAULT_IPD_SEED)
    metric_name = f"{context.tag} {context.prefix}".strip()
    metrics = compute_debug_metrics(
        context.ref_output,
        context.output,
        name=metric_name,
        n_probes=n_probes,
        seed=seed,
    )
    logger.info_rank0(
        f"{_PREFIX} {context.tag} step={context.step} | {context.prefix}"
        f"{_format_compare_metrics(metrics)}"
    )
    return metrics


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
        analysis_func: Optional[Callable[[DebugAnalysisContext], Any]] = None,
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
        self._analysis_hooks: List[Callable[[DebugAnalysisContext], Any]] = [
            analysis_func or default_precision_analysis_hook
        ]

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

    def set_analysis_func(
        self,
        func: Optional[Callable[[DebugAnalysisContext], Any]],
    ):
        """Replace compare analysis hooks.

        Passing ``None`` restores the default precision analysis hook. The hook
        is called once for every tensor output matched by compare rules.
        """
        self._analysis_hooks = [func or default_precision_analysis_hook]

    def add_analysis_hook(self, func: Callable[[DebugAnalysisContext], Any]):
        """Append an analysis hook after the current hooks."""
        self._analysis_hooks.append(func)

    def clear_analysis_hooks(self):
        """Disable compare analysis hooks while keeping compare execution active."""
        self._analysis_hooks.clear()

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

        self._compare_and_report(
            tag,
            step,
            output,
            ref_output,
            module=module,
            inputs=inputs,
            layer_idx=layer_idx,
            op_name=op_name,
        )
        return ref_output

    def _compare_and_report(
        self,
        tag: str,
        step: int,
        output,
        ref_output,
        module=None,
        inputs=None,
        layer_idx=None,
        op_name: str = "",
        prefix: str = "",
    ):
        if isinstance(output, torch.Tensor) and isinstance(ref_output, torch.Tensor):
            if output.shape != ref_output.shape:
                logger.warning_rank0(
                    f"{_PREFIX} {tag} step={step} | {prefix}SHAPE_MISMATCH "
                    f"got {tuple(output.shape)} vs ref {tuple(ref_output.shape)}"
                )
                return

            context = DebugAnalysisContext(
                tag=tag,
                step=step,
                layer_idx=layer_idx,
                op_name=op_name,
                module=module,
                inputs=inputs,
                output=output,
                ref_output=ref_output,
                prefix=prefix,
            )
            for analysis_hook in self._analysis_hooks:
                try:
                    analysis_hook(context)
                except Exception as e:
                    logger.warning_once(
                        f"{_PREFIX} Analysis hook failed for {tag}: {e}. "
                        f"Analysis skipped, inference unaffected."
                    )

        elif isinstance(output, (tuple, list)) and isinstance(ref_output, (tuple, list)):
            if len(output) != len(ref_output):
                logger.warning_rank0(
                    f"{_PREFIX} {tag} step={step} | {prefix}LENGTH_MISMATCH "
                    f"got {len(output)} vs ref {len(ref_output)}"
                )
                return
            for i, (o, r) in enumerate(zip(output, ref_output)):
                self._compare_and_report(
                    tag,
                    step,
                    o,
                    r,
                    module=module,
                    inputs=inputs,
                    layer_idx=layer_idx,
                    op_name=op_name,
                    prefix=f"[{i}] ",
                )

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
