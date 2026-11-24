"""Load optional package additions without masking broken dependencies."""
from importlib import import_module
from importlib.util import find_spec


def additions(package):
    name = package + "._private"
    return import_module(name) if find_spec(name) is not None else None


def extend_exports(namespace):
    module = additions(namespace["__name__"])
    if module is None:
        return
    exports = {name: getattr(module, name) for name in module.__all__}
    duplicates = set(exports).intersection(namespace["__all__"])
    if duplicates:
        raise ValueError(f"Duplicate optional exports: {sorted(duplicates)}")
    namespace.update(exports)
    namespace["__all__"].extend(exports)


def extend_bindings(package, bindings):
    module = additions(package)
    if module is None:
        return
    duplicates = bindings.keys() & module.OPS.keys()
    if duplicates:
        raise ValueError(f"Duplicate optional bindings: {sorted(duplicates)}")
    bindings.update(module.OPS)
