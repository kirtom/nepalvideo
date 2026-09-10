"""Configuration loading.

One YAML file is the single source of truth for every tunable in the spec.
Values are reachable by dotted path so call sites read like the spec section
they implement: cfg.get("probe.fov.fallback_deg").
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


class Config:
    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else Path(os.environ.get("NEPAL_CONFIG", DEFAULT_CONFIG))
        with open(p) as fh:
            return cls(yaml.safe_load(fh), p)

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise KeyError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    # -- derived paths -------------------------------------------------
    @property
    def data_root(self) -> Path:
        return Path(self.get("project.data_root")).expanduser()

    @property
    def work_root(self) -> Path:
        return Path(self.get("project.work_root")).expanduser()

    @property
    def db_path(self) -> Path:
        return Path(self.get("project.db_path")).expanduser()

    def work(self, *parts: str) -> Path:
        """Path under work_root, with parent directories created."""
        p = self.work_root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def workdir(self, *parts: str) -> Path:
        """Directory under work_root, created."""
        p = self.work_root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def quality_curve(self, name: str) -> dict[str, float]:
        curves = self.get("quality_curves")
        return curves.get(name, curves["camera"])

    def act_targets(self) -> list[dict[str, Any]]:
        return list(self.get("acts"))
