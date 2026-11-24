import os

from importlib.resources import files
from pathlib import Path
from threading import RLock
from typing import Any
from typing import Mapping
from typing import Optional

import yaml

from mojo_opset.utils.target import Target

_CONFIG_PATH_ENV = "MOJO_OPS_CONFIG_PATH"


def normalize_name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} must be a non-empty string.")
    return value.strip().lower()


def _normalize_config(raw_config: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw_config, Mapping) or not raw_config:
        raise ValueError("Mojo config must contain a non-empty target-to-operators mapping.")

    result: dict[str, dict[str, str]] = {}
    for raw_target, raw_ops in raw_config.items():
        target = Target.parse(raw_target).key
        if target in result:
            raise ValueError(f"Duplicate configuration for target={target!r}.")
        if not isinstance(raw_ops, Mapping):
            raise ValueError(f"Implementations for target={target!r} must be an operator mapping.")
        ops = {}
        for raw_op_id, raw_implementation in raw_ops.items():
            op_id = normalize_name(raw_op_id, "op id")
            if op_id in ops:
                raise ValueError(f"Duplicate configuration for target={target!r}, op={op_id!r}.")
            ops[op_id] = normalize_name(raw_implementation, "implementation")
        result[target] = ops
    return result


def _merge_configs(
    configs: list[tuple[str, dict[str, dict[str, str]]]],
) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    sources: dict[tuple[str, str], str] = {}
    for source, config in configs:
        for target, ops in config.items():
            merged_ops = merged.setdefault(target, {})
            for op_id, implementation in ops.items():
                key = (op_id, target)
                if key in sources:
                    raise ValueError(
                        f"Duplicate configuration for op={op_id!r}, target={target!r} "
                        f"in {sources[key]!r} and {source!r}."
                    )
                merged_ops[op_id] = implementation
                sources[key] = source
    return merged


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Mojo config keys must be strings.")
            if key in keys:
                raise ValueError(f"Duplicate configuration key {key!r} at {key_node.start_mark}.")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _load_file(path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as config_file:
        return _normalize_config(yaml.load(config_file, Loader=_UniqueKeyLoader))


def _load_directory(path) -> dict[str, dict[str, str]]:
    # Path and package resources share this directory/file interface.
    config_paths = sorted(
        (entry for entry in path.iterdir() if entry.name.endswith((".yaml", ".yml")) and entry.is_file()),
        key=lambda entry: entry.name,
    )
    if not config_paths:
        raise ValueError(f"Config directory {str(path)!r} contains no YAML files.")
    return _merge_configs([(str(config_path), _load_file(config_path)) for config_path in config_paths])


def _load_config(path: Optional[str] = None) -> dict[str, dict[str, str]]:
    config = _load_directory(files(__package__))
    configured_path = path or os.getenv(_CONFIG_PATH_ENV)
    if configured_path:
        config_path = Path(configured_path)
        overrides = _load_directory(config_path) if config_path.is_dir() else _load_file(config_path)
        for target, ops in overrides.items():
            config.setdefault(target, {}).update(ops)
    return config


_CONFIG_LOCK = RLock()
_CONFIG = None


def _get_config_snapshot() -> dict[str, dict[str, str]]:
    """Return the cached configuration; internal callers must not mutate it."""
    global _CONFIG
    snapshot = _CONFIG
    if snapshot is not None:
        return snapshot

    import torch

    if torch.compiler.is_compiling():
        raise RuntimeError("Load Mojo configuration before compilation: call mojo_opset.preload or use eager warmup.")
    with _CONFIG_LOCK:
        if _CONFIG is None:
            _CONFIG = _load_config()
        return _CONFIG


def get_config() -> dict[str, dict[str, str]]:
    return {target: dict(ops) for target, ops in _get_config_snapshot().items()}


def configure(config: Mapping[str, Any]) -> None:
    """Atomically replace dispatch configuration using the YAML-equivalent schema."""

    global _CONFIG
    normalized = _normalize_config(config)
    with _CONFIG_LOCK:
        _CONFIG = normalized


def reload_config(path: Optional[str] = None) -> None:
    """Reload packaged defaults, overlaid by a path or MOJO_OPS_CONFIG_PATH."""

    global _CONFIG
    normalized = _load_config(path)
    with _CONFIG_LOCK:
        _CONFIG = normalized
