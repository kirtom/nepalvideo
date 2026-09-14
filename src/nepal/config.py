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


class DuplicateKeyError(ValueError):
    """A mapping in the config declares the same key twice."""


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys instead of silently keeping one.

    YAML's rule is last-one-wins, applied without a word. A second ``process:``
    block appended to the file -- the natural way to add a section for a new
    stage -- therefore deletes the first one, and every tunable in it reverts
    to whatever default the call site happened to pass. That is not a
    hypothetical: it cost a 3.5-hour S03.1 run, which built every proxy at the
    source frame rate because ``process.proxy_fps`` had been shadowed away.
    A config file is the one place in this pipeline where a silent default is
    indistinguishable from a decision, so the loader fails loudly instead.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                    f"of {key_node.start_mark.name}: the later block would "
                    f"silently replace the earlier one")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


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
            return cls(yaml.load(fh, Loader=_StrictLoader), p)

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
