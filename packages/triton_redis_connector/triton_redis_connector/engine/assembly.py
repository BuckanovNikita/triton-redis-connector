"""Config-driven node assembly: import and type resolution utilities.

Provides ``import_class`` for dynamic class loading and ``resolve_type_params``
for resolving string class paths in constructor parameters annotated as ``type[X]``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


def import_class(class_path: str) -> type:
    """Import a class from a fully-qualified dotted path.

    Raises ``ImportError`` if the module cannot be found and ``AttributeError``
    if the class does not exist in the module.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        msg = f"class_path must be fully-qualified (got {class_path!r})"
        raise ImportError(msg)
    module = importlib.import_module(module_path)
    result: type = getattr(module, class_name)
    return result


def resolve_type_params(cls: type, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Resolve string class paths in params that map to ``type[X]`` hints.

    Inspects the ``__init__`` signature of *cls*. For each parameter annotated
    as ``type[X]``, if the raw value is a string, import it as a class.
    Other parameters pass through unchanged.
    """
    type_param_names = _get_init_type_params(cls)
    resolved: dict[str, Any] = {}
    for key, value in raw_params.items():
        if isinstance(value, str) and key in type_param_names:
            resolved[key] = import_class(value)
        else:
            resolved[key] = value
    return resolved


def _get_init_type_params(cls: type) -> set[str]:
    """Return parameter names whose annotation looks like ``type[...]``.

    Works with both evaluated and string annotations (PEP 695 + future
    annotations). For string annotations, matches the pattern ``type[X]``.
    For evaluated annotations, checks ``typing.get_origin(...) is type``.
    """
    raw = inspect.get_annotations(cls.__init__)  # type: ignore[misc]
    result: set[str] = set()
    for name, annotation in raw.items():
        if name == "return":
            continue
        if isinstance(annotation, str) and annotation.startswith("type["):
            result.add(name)
    return result
