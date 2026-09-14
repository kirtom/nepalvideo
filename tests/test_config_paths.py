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


# -- duplicate keys ---------------------------------------------------
def test_a_repeated_section_is_an_error_not_a_silent_replacement(tmp_path):
    """YAML's rule is last-one-wins, applied without a word. A second
    ``process:`` block appended to the config -- the natural way to add a
    section for a new stage -- deleted the first one and reverted every tunable
    in it to a call-site default. It cost a 3.5-hour run of S03.1, which built
    every proxy at the source frame rate because ``proxy_fps`` was shadowed."""
    import pytest
    from nepal.config import Config, DuplicateKeyError

    p = tmp_path / "dup.yaml"
    p.write_text("process:\n  proxy_fps: 15\nprocess:\n  photo_workers: 0\n")
    with pytest.raises(DuplicateKeyError) as exc:
        Config.load(p)
    assert "process" in str(exc.value)


def test_a_repeated_key_inside_a_section_is_an_error_too(tmp_path):
    import pytest
    from nepal.config import Config, DuplicateKeyError

    p = tmp_path / "dup.yaml"
    p.write_text("process:\n  proxy_fps: 15\n  proxy_fps: 30\n")
    with pytest.raises(DuplicateKeyError):
        Config.load(p)


def test_the_shipped_config_has_no_duplicate_keys():
    """The guard is worth nothing if the file it guards is not loaded by it."""
    from nepal.config import Config, DEFAULT_CONFIG
    cfg = Config.load(DEFAULT_CONFIG)
    assert cfg.get("process.proxy_fps") == 15
    assert cfg.get("process.photo_workers") == 0


# -- finding the config ----------------------------------------------
def test_an_installed_package_still_finds_the_projects_config(tmp_path, monkeypatch):
    """Three directories up from config.py is the repository root only in a
    source checkout. Installed into a virtualenv the same arithmetic points at
    .venv/lib/python3.14/config/pipeline.yaml -- a path nobody chose, whose
    absence reads as a missing file rather than a wrong guess."""
    from nepal.config import find_config

    project = tmp_path / "nepalvideo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "pipeline.yaml").write_text("project: {name: t}\n")
    venv = project / ".venv" / "lib" / "python3.14" / "site-packages" / "nepal"
    venv.mkdir(parents=True)
    monkeypatch.delenv("NEPAL_CONFIG", raising=False)
    assert find_config(venv) == project / "config" / "pipeline.yaml"


def test_the_environment_beats_every_other_location(tmp_path, monkeypatch):
    from nepal.config import find_config

    chosen = tmp_path / "mine.yaml"
    chosen.write_text("project: {name: t}\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "pipeline.yaml").write_text("project: {name: other}\n")
    monkeypatch.setenv("NEPAL_CONFIG", str(chosen))
    assert find_config(tmp_path) == chosen


def test_the_nearest_project_wins_over_one_further_up(tmp_path, monkeypatch):
    from nepal.config import find_config

    for name in ("outer", "outer/inner"):
        (tmp_path / name / "config").mkdir(parents=True)
        (tmp_path / name / "config" / "pipeline.yaml").write_text("project: {}\n")
    monkeypatch.delenv("NEPAL_CONFIG", raising=False)
    assert find_config(tmp_path / "outer" / "inner") == \
        tmp_path / "outer" / "inner" / "config" / "pipeline.yaml"


def test_a_missing_config_says_where_it_looked(tmp_path, monkeypatch):
    """A FileNotFoundError on a path the user never typed is not a diagnosis."""
    from nepal.config import candidate_configs, ConfigNotFound

    monkeypatch.delenv("NEPAL_CONFIG", raising=False)
    root = tmp_path / "a" / "b"
    root.mkdir(parents=True)
    tried = candidate_configs(root)
    assert root / "config" / "pipeline.yaml" in tried
    message = str(ConfigNotFound(tried))
    assert "Looked in:" in message
    assert str(root / "config" / "pipeline.yaml") in message
    assert "--config" in message and "NEPAL_CONFIG" in message


def test_every_candidate_is_listed_once(tmp_path, monkeypatch):
    """The working directory walk and the package walk overlap whenever the
    virtualenv lives inside the project, which is the ordinary case."""
    from nepal.config import candidate_configs

    monkeypatch.delenv("NEPAL_CONFIG", raising=False)
    tried = candidate_configs(tmp_path)
    assert len(tried) == len(set(tried))


def test_an_explicit_config_that_does_not_exist_is_reported_clearly(tmp_path):
    import pytest
    from nepal.config import Config, ConfigNotFound

    with pytest.raises(ConfigNotFound):
        Config.load(tmp_path / "nope.yaml")
