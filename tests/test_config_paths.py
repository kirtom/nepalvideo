import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
import pytest

from nepal.config import Config


def _cfg(tmp_path, **project):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True, exist_ok=True)
    data = {"project": {"data_root": "./nepal_data", "work_root": "./work",
                        "db_path": "./work/db/nepal.sqlite", **project},
            "spine": {"srtm_dir": "./data/srtm",
                      "geonames_path": "./data/geonames/NP.txt"}}
    p = root / "config" / "pipeline.yaml"
    p.write_text(yaml.safe_dump(data))
    return Config.load(p), root


def test_relative_paths_anchor_to_the_project_root(tmp_path):
    cfg, root = _cfg(tmp_path)
    assert cfg.data_root == root / "nepal_data"
    assert cfg.work_root == root / "work"
    assert cfg.db_path == root / "work" / "db" / "nepal.sqlite"
    assert cfg.srtm_dir == root / "data" / "srtm"


def test_resolution_is_independent_of_the_working_directory(tmp_path, monkeypatch):
    """The bug this guards: running `nepal` from two directories produced two
    different work folders and two different databases, and the second run
    silently redid everything into the new one."""
    cfg, root = _cfg(tmp_path)
    first = cfg.work_root
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg2, _ = _cfg(tmp_path)     # same config file, different cwd
    assert cfg2.work_root == first


def test_absolute_paths_are_left_alone(tmp_path):
    cfg, _ = _cfg(tmp_path, data_root="/mnt/big/nepal_data")
    assert str(cfg.data_root) == "/mnt/big/nepal_data"


def test_home_relative_paths_expand(tmp_path):
    cfg, _ = _cfg(tmp_path, data_root="~/nepal_data")
    assert str(cfg.data_root).startswith(str(pathlib.Path.home()))
    assert "~" not in str(cfg.data_root)


def test_parent_relative_paths_resolve(tmp_path):
    """A data_root beside the repo rather than inside it -- the sensible place
    for 65 GB of media."""
    cfg, root = _cfg(tmp_path, data_root="../nepal_data")
    assert cfg.data_root == (root.parent / "nepal_data").resolve()


def test_config_outside_a_config_dir_anchors_to_its_own_folder(tmp_path):
    """A config kept beside the media rather than in <root>/config/."""
    import yaml
    here = tmp_path / "beside_the_media"
    here.mkdir()
    p = here / "pipeline.yaml"
    p.write_text(yaml.safe_dump({"project": {"data_root": "./nepal_data",
                                             "work_root": "./work",
                                             "db_path": "./work/db/nepal.sqlite"}}))
    cfg = Config.load(p)
    assert cfg.base_dir == here
    assert cfg.data_root == here / "nepal_data"


def test_config_in_a_config_dir_anchors_to_the_project_root(tmp_path):
    cfg, root = _cfg(tmp_path)
    assert cfg.base_dir == root
