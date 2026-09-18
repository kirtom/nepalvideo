"""A host that holds part of the corpus must not delete the rest."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal import db
from nepal.config import Config
from nepal.probe import manifest
from nepal.stages import s01_probe


def _cfg(tmp_path):
    data = tmp_path / "data"
    (data / "media_from_phones" / "keller").mkdir(parents=True)
    (data / "media_from_phones" / "keller" / "IMG_1.jpg").write_bytes(b"\xff\xd8jpeg-ish")
    return Config({"project": {"data_root": str(data), "work_root": str(tmp_path / "work"),
                               "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
                   "probe": {"max_chapter_gap_s": 1.0, "exclude_dirs": [], "ignore_globs": [],
                             "capture_time_overrides": {}}}), data


def test_absent_sources_names_what_this_host_does_not_hold(tmp_path):
    _, data = _cfg(tmp_path)
    assert manifest.absent_sources(data) == {"camera", "phone_kulikov", "telegram", "music"}


def test_rows_of_an_absent_source_survive_a_re_probe(tmp_path):
    cfg, data = _cfg(tmp_path)
    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind) VALUES "
                 "('cam1', 'raw/media_from_camera/VID_1.insv', 'camera', 'video360')")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind) VALUES "
                 "('gone', 'raw/media_from_phones/keller/IMG_9.jpg', 'phone_keller', 'photo')")
    conn.commit()
    rep = s01_probe.build_manifest(cfg, conn)
    ids = {r[0] for r in conn.execute("SELECT asset_id FROM assets")}
    assert "cam1" in ids                       # its source is absent here: untouched
    assert "gone" not in ids                   # its source is present and the file is not
    assert rep["sources_absent"] == sorted({"camera", "phone_kulikov", "telegram", "music"})


def test_an_unchanged_file_is_not_hashed_again(tmp_path, monkeypatch):
    cfg, data = _cfg(tmp_path)
    conn = db.init(cfg.db_path)
    first = s01_probe.build_manifest(cfg, conn)
    assert first["n_hash_reused"] == 0
    calls = []
    real = s01_probe.sha256_file
    monkeypatch.setattr(s01_probe, "sha256_file", lambda p: (calls.append(p), real(p))[1])
    second = s01_probe.build_manifest(cfg, conn)
    assert second["n_hash_reused"] == 1 and calls == []
    (data / "media_from_phones" / "keller" / "IMG_1.jpg").write_bytes(b"\xff\xd8changed!")
    third = s01_probe.build_manifest(cfg, conn)
    assert third["n_hash_reused"] == 0 and len(calls) == 1
