"""Accuracy comparison helpers for benchmark op_defs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union

import torch

from mojo_opset.utils.acc import check_tol_diff

TensorLike = Union[torch.Tensor, Tuple[Any, ...], list]


@dataclass
class AccuracyMetrics:
    max_abs_diff: Optional[float]
    mean_abs_diff: Optional[float]
    max_rel_diff: Optional[float]
    mean_rel_diff: Optional[float]
    rmse: Optional[float]
    match_ratio: Optional[float]
    status: str
    note: str = ""


def clone_tensor_mapping(mapping: dict) -> dict:
    """Deep-clone tensor values in a ``tensor_mapping`` dict."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in mapping.items()
    }


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _tensor_metrics(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    rel_eps: float,
) -> AccuracyMetrics:
    out_f = _to_cpu_float(out)
    ref_f = _to_cpu_float(ref)
    if out_f.shape != ref_f.shape:
        return AccuracyMetrics(
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            mean_rel_diff=float("inf"),
            rmse=float("inf"),
            match_ratio=0.0,
            status="FAIL",
            note=f"shape mismatch: {tuple(out_f.shape)} vs {tuple(ref_f.shape)}",
        )

    abs_diff = (out_f - ref_f).abs()
    denom = ref_f.abs().clamp_min(rel_eps)
    rel_diff = abs_diff / denom

    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    max_rel = float(rel_diff.max().item())
    mean_rel = float(rel_diff.mean().item())
    rmse = float(torch.sqrt((abs_diff * abs_diff).mean()).item())
    matches = torch.isclose(out_f, ref_f, rtol=rtol, atol=atol)
    match_ratio = float(matches.float().mean().item())

    status = "PASS"
    note = ""
    try:
        check_tol_diff(out_f, ref_f, atol=atol, rtol=rtol)
    except AssertionError as err:
        status = "FAIL"
        note = str(err)

    return AccuracyMetrics(
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        max_rel_diff=max_rel,
        mean_rel_diff=mean_rel,
        rmse=rmse,
        match_ratio=match_ratio,
        status=status,
        note=note,
    )


def _pick_worst(metrics: Iterable[AccuracyMetrics]) -> AccuracyMetrics:
    ranked = sorted(
        metrics,
        key=lambda item: (
            0 if item.status == "FAIL" else 1,
            -item.max_abs_diff,
            -item.match_ratio,
        ),
    )
    return ranked[0]


def compare_outputs(
    out: TensorLike,
    ref: TensorLike,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    rel_eps: float = 1e-12,
) -> AccuracyMetrics:
    """Compare vendor output against reference with preset accuracy metrics."""
    if isinstance(out, torch.Tensor) and isinstance(ref, torch.Tensor):
        return _tensor_metrics(out, ref, atol=atol, rtol=rtol, rel_eps=rel_eps)

    if isinstance(out, (tuple, list)) and isinstance(ref, (tuple, list)):
        if len(out) != len(ref):
            return AccuracyMetrics(
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                mean_rel_diff=float("inf"),
                rmse=float("inf"),
                match_ratio=0.0,
                status="FAIL",
                note=f"output length mismatch: {len(out)} vs {len(ref)}",
            )
        if not out:
            return AccuracyMetrics(
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                max_rel_diff=0.0,
                mean_rel_diff=0.0,
                rmse=0.0,
                match_ratio=1.0,
                status="PASS",
            )
        return _pick_worst(
            compare_outputs(out_i, ref_i, atol=atol, rtol=rtol, rel_eps=rel_eps)
            for out_i, ref_i in zip(out, ref)
        )

    return AccuracyMetrics(
        max_abs_diff=float("inf"),
        mean_abs_diff=float("inf"),
        max_rel_diff=float("inf"),
        mean_rel_diff=float("inf"),
        rmse=float("inf"),
        match_ratio=0.0,
        status="FAIL",
        note=f"unsupported output types: {type(out).__name__} vs {type(ref).__name__}",
    )
