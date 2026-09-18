"""What travels, and the rsync commands that move it."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.cloud import sync
from nepal.config import Config


def _cfg(tmp_path):
    return Config({"project": {"data_root": str(tmp_path / "data"),
                               "work_root": str(tmp_path / "work"),
                               "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
                   "cloud": {"gcp": {"bucket": "gs://b"}}},
                  path=tmp_path / "config" / "pipeline.yaml")


def test_rsync_args_are_recursive_and_exclude_junk():
    args = sync.rsync_args("/x", "gs://b/y")
    assert args[:3] == ["storage", "rsync", "--recursive"]
    assert any(a.startswith("--exclude=") and "lock" in a for a in args)
    assert args[-2:] == ["/x", "gs://b/y"]


def test_push_plan_covers_the_working_set_and_not_the_originals(tmp_path):
    plan = sync.push_plan(_cfg(tmp_path))
    dsts = {d for _, d in plan}
    assert "gs://b/raw/media_from_phones" in dsts and "gs://b/raw/strava" in dsts
    assert "gs://b/work/proxies" in dsts and "gs://b/work/db" in dsts
    assert "gs://b/ref/data" in dsts
    assert not any("media_from_camera" in d for d in dsts)
    ref = [s for s, d in plan if d == "gs://b/ref/data"][0]
    assert ref == tmp_path / "data"                       # <project root>/data


def test_pull_plan_is_what_a_run_can_change(tmp_path):
    plan = sync.pull_plan(_cfg(tmp_path))
    srcs = {s for s, _ in plan}
    assert "gs://b/work/db" in srcs and "gs://b/work/reports" in srcs
    assert "gs://b/work/semantic" in srcs and "gs://b/work/gates" in srcs
    assert "gs://b/work/proxies" not in srcs                    # never changes after S03.1
    assert all(str(d).startswith(str(tmp_path / "work")) for _, d in plan)
