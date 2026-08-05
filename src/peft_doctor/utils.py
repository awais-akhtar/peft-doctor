"""Small helpers for dependency-light introspection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union


MISSING = object()


def optional_import(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except Exception:
        return None


def get_value(obj: Any, name: str, default: Any = None) -> Any:
    """Read a value from dicts, dataclasses, or ordinary Python objects."""

    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj).get(name, default)
    return getattr(obj, name, default)


def first_value(obj: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        value = get_value(obj, name, MISSING)
        if value is not MISSING and value is not None:
            return value
    return default


def bool_value(obj: Any, names: list[str], default: bool = False) -> bool:
    value = first_value(obj, names, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def read_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")


def coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
