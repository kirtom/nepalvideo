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
    """Loaded pipeline configuration.

    Relative paths in the config resolve against the config file's own
    directory, NOT the process working directory. The config is found relative
    to the installed package, so resolving its contents against the cwd instead
    would mean running ``nepal`` from two different directories produced two
    different work folders and two different databases -- with the second run
    silently redoing everything into a new place.
    """

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = Path(path) if path else None

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths are resolved against.

        The project root: the parent of a ``config/`` directory when the file
        lives in one, otherwise the file's own directory. That way a config
        kept beside the media, or anywhere else, resolves against a location
        the person who put it there would predict.
        """
        if self.path is None:
            return Path.cwd()
        parent = self.path.resolve().parent
        return parent.parent if parent.name == "config" else parent

    def resolve(self, value: str | os.PathLike) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

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
        return self.resolve(self.get("project.data_root"))

    @property
    def work_root(self) -> Path:
        return self.resolve(self.get("project.work_root"))

    @property
    def db_path(self) -> Path:
        return self.resolve(self.get("project.db_path"))

    @property
    def srtm_dir(self) -> Path:
        return self.resolve(self.get("spine.srtm_dir", "./data/srtm"))

    @property
    def geonames_path(self) -> Path:
        return self.resolve(self.get("spine.geonames_path", "./data/geonames/NP.txt"))

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
