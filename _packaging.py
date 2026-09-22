"""Build configuration helpers that do not import the operator package."""

import json
import os
from pathlib import Path
import re


def native_provider(root):
    """Select exactly one lib distribution for the current build environment."""
    root = Path(root)
    provider = os.environ.get("MOJO_NATIVE_PROVIDER", "")
    if (not re.fullmatch(r"[a-z][a-z0-9]*_[a-z0-9_]+(?:/sku_[a-z0-9_]+)?", provider)
            or not (root / "native/mojo_opset_lib" / provider / "__init__.py").is_file()
            or not (root / "native/src" / provider / "bindings.cpp").is_file()):
        raise ValueError(f"MOJO_NATIVE_PROVIDER must select one supported provider: {provider!r}")
    return provider


_STRINGS = {"name", "version", "description", "requires_python", "readme"}
_FIELDS = _STRINGS | {"urls", "dependencies", "optional_dependencies"}


def _read(path, *, override=False):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read packaging configuration {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    allowed = _FIELDS - {"version"} if override else _FIELDS
    if data.keys() - allowed:
        raise ValueError(f"Unsupported fields in {path.name}: {sorted(data.keys() - allowed)}")
    if not override and _FIELDS - data.keys():
        raise ValueError(f"Missing fields in {path.name}: {sorted(_FIELDS - data.keys())}")
    for key in data.keys() & _STRINGS:
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{path.name}: {key} must be a nonempty string")
    if "urls" in data and (
        not isinstance(data["urls"], dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in data["urls"].items())
    ):
        raise ValueError(f"{path.name}: urls must map strings to strings")
    extras = data.get("optional_dependencies", {})
    if not isinstance(extras, dict) or any(not isinstance(k, str) or not k for k in extras):
        raise ValueError(f"{path.name}: optional_dependencies must be an object with nonempty group names")
    lists = {"dependencies": data.get("dependencies", []), **{f"extra {k}": v for k, v in extras.items()}}
    for key, values in lists.items():
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{path.name}: {key} must be a list of requirement strings")
    return data


def load_metadata(root):
    """Load defaults, then apply packaging_*.json overrides in filename order."""
    root = Path(root)
    metadata = _read(root / "packaging.json")
    for path in sorted(root.glob("packaging_*.json")):
        overrides = _read(path, override=True)
        extras = {**metadata["optional_dependencies"], **overrides.pop("optional_dependencies", {})}
        metadata.update(overrides)
        metadata["optional_dependencies"] = extras
    return metadata


def _lib_metadata(root):
    path = Path(root) / "native/packaging.json"
    try:
        versions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read native/packaging.json") from error
    if not isinstance(versions, dict) or not versions:
        raise ValueError("native/packaging.json: expected provider versions")
    providers = {}
    names = set()
    for provider, version in versions.items():
        if (not re.fullmatch(r"[a-z][a-z0-9]*_[a-z0-9_]+(?:/sku_[a-z0-9_]+)?", provider)
                or not isinstance(version, str) or not version.strip()):
            raise ValueError(f"native/packaging.json: expected a version for {provider!r}")
        name = provider.replace("/sku_", "-").replace("_", "-")
        if name in names:
            raise ValueError(f"native/packaging.json: duplicate lib name {name!r}")
        names.add(name)
        providers[provider] = {"name": name, "version": version}
    return providers


def setup_metadata(root):
    metadata = load_metadata(root)
    extras = {group: list(requirements) for group, requirements in metadata["optional_dependencies"].items()}
    for provider, lib in _lib_metadata(root).items():
        platform = provider.split("_", 1)[0]
        extras.setdefault(platform, []).append(f"{metadata['name']}-lib-{lib['name']}=={lib['version']}")
    return {
        "name": metadata["name"],
        "version": metadata["version"],
        "description": metadata["description"],
        "python_requires": metadata["requires_python"],
        "project_urls": metadata["urls"],
        "long_description": (Path(root) / metadata["readme"]).read_text(encoding="utf-8"),
        "long_description_content_type": "text/markdown",
        "install_requires": metadata["dependencies"],
        "extras_require": extras,
    }


def lib_setup_metadata(root, provider):
    """Lib releases have their own versions; the main extra pins compatible releases."""
    root = Path(root)
    main = load_metadata(root)
    providers = _lib_metadata(root)
    if provider not in providers:
        raise ValueError(f"native/packaging.json: missing provider {provider!r}")
    lib = providers[provider]
    return {
        "name": main["name"] + "-lib-" + lib["name"],
        "version": lib["version"],
        "description": f"Prebuilt Mojo operator libraries for {lib['name']}",
        "python_requires": main["requires_python"],
        "project_urls": main["urls"],
        "long_description": (root / "native/README.md").read_text(encoding="utf-8"),
        "long_description_content_type": "text/markdown",
        "install_requires": [],
    }
