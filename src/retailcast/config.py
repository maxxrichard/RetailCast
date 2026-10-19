"""YAML configuration with simple inheritance via an ``extends`` key."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_raw(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file, resolving ``extends`` relative to the file's directory."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular `extends` detected at {path}")
    _seen.add(path)
    raw = _load_yaml(path)
    parent = raw.pop("extends", None)
    if parent:
        base = load_raw(path.parent / parent, _seen)
        raw = _deep_merge(base, raw)
    return raw


@dataclass
class Config:
    """A thin attribute-style wrapper over the merged configuration dict."""

    raw: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        raw = object.__getattribute__(self, "raw")
        if name in raw:
            v = raw[name]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(name)

    def __getitem__(self, name: str) -> Any:
        return self.raw[name]

    def get(self, name: str, default: Any = None) -> Any:
        v = self.raw.get(name, default)
        return Config(v) if isinstance(v, dict) else v

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    def override(self, **kwargs: Any) -> Config:
        """Return a new Config with dotted keys overridden, e.g. ``horizon=14``."""
        raw = copy.deepcopy(self.raw)
        for key, value in kwargs.items():
            parts = key.split("__")
            node = raw
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return Config(raw)


def load_config(path: str | Path = "default.yaml", **overrides: Any) -> Config:
    cfg = Config(load_raw(path))
    return cfg.override(**overrides) if overrides else cfg
