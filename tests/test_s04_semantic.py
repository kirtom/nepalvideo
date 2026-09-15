"""S04.1 -- which shots get an embedding.

The bug this guards: S04.1 asked for ``status = 'candidate'``, but S05
promotes the shots it picks to 'shortlisted'. Once a cut existed, the 400
shots the film is actually made of no longer matched -- so a later S04.1 run
would embed 1232 rows and skip exactly the ones MMR needs to compare.
"""
import json
import sys, pathlib

import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db
from nepal.stages import s04_semantic as s04


@pytest.fixture
def conn(tmp_path):
    c = db.init(tmp_path / "t.sqlite")
    c.execute("INSERT INTO recordings(recording_id, source, is_360) "
              "VALUES ('rec1', 'camera', 1)")
    c.execute("INSERT INTO assets(asset_id, s3_key, source, kind) "
              "VALUES ('ast1', 'raw/p.jpg', 'phone_keller', 'photo')")
    for sid, status, kind in (
            ("s#cand", "candidate", "video"),
            ("s#short", "shortlisted", "video"),
            ("s#rej", "rejected", "video"),
            ("p#short", "shortlisted", "photo"),
    ):
        if kind == "photo":
            c.execute("INSERT INTO shots(shot_id, asset_id, media_kind, start_s,"
                      " end_s, status) VALUES (?,?,?,0,3,?)", (sid, "ast1", kind, status))
        else:
            c.execute("INSERT INTO shots(shot_id, recording_id, media_kind, start_s,"
                      " end_s, status) VALUES (?,?,?,0,3,?)", (sid, "rec1", kind, status))
    return c


def test_shortlisted_shots_are_embedded_not_skipped(conn):
    """The shots in the cut are 'shortlisted'. They are the whole point."""
    ids = {r["shot_id"] for r in s04.surviving_shots(conn)}
    assert "s#short" in ids and "p#short" in ids


def test_the_gate_s_rejections_are_not_paid_for(conn):
    ids = {r["shot_id"] for r in s04.surviving_shots(conn)}
    assert "s#rej" not in ids
    assert ids == {"s#cand", "s#short", "p#short"}


def test_a_photo_carries_its_key_and_a_video_its_recording(conn):
    """The join is what tells the stage where to read a frame from; a missing
    s3_key sends a photograph to a proxy that was never built for it."""
    by_id = {r["shot_id"]: r for r in s04.surviving_shots(conn)}
    assert by_id["p#short"]["s3_key"] == "raw/p.jpg"
    assert by_id["s#short"]["recording_id"] == "rec1"


# -- a six-hour run must survive being killed --------------------------

class _Cfg:
    """The slice of Config that embed_shots touches."""
    def __init__(self, root, values):
        self._root, self._v = root, values
    def get(self, dotted, default=None):
        return self._v.get(dotted, default)
    @property
    def work_root(self):
        return self._root
    @property
    def data_root(self):
        return self._root / "data"
    def workdir(self, *parts):
        p = self._root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _fake_embedding_world(monkeypatch, *, boom_after=None):
    """Replace the model and the decoder; keep every real path and file write."""
    calls = {"n": 0}

    def fake_load(name, pretrained, *, gpu=False, threads=0):
        return object(), object(), "cpu"

    def fake_encode(model, pre, dev, images):
        calls["n"] += len(images)
        if boom_after is not None and calls["n"] > boom_after:
            raise KeyboardInterrupt("killed mid-run")
        return np.ones((len(images), 4), dtype=np.float32)

    monkeypatch.setattr(s04.embed_mod, "load_model", fake_load)
    monkeypatch.setattr(s04.embed_mod, "encode", fake_encode)
    monkeypatch.setattr(s04.embed_mod, "sharpest", lambda frames: 0)
    monkeypatch.setattr(s04.metrics_mod, "sample_times", lambda a, b, n: [0.0])
    monkeypatch.setattr(s04.metrics_mod, "read_samples",
                        lambda px, times, frames_per_sample=1:
                        [[np.zeros((8, 8, 3), dtype=np.uint8)]])
    return calls


@pytest.fixture
def many_shots(tmp_path):
    c = db.init(tmp_path / "t.sqlite")
    c.execute("INSERT INTO recordings(recording_id, source, is_360) "
              "VALUES ('rec1', 'camera', 1)")
    for i in range(40):
        c.execute("INSERT INTO shots(shot_id, recording_id, media_kind, start_s,"
                  " end_s, status) VALUES (?,'rec1','video',0,3,'candidate')",
                  (f"s#{i:03d}",))
    return c


def test_progress_survives_a_kill_and_the_next_run_resumes(tmp_path, monkeypatch,
                                                           many_shots):
    """The bug this guards: the stage held every vector in memory and wrote
    once at the end, so a kill at hour five of six threw away five hours."""
    (tmp_path / "proxies").mkdir(exist_ok=True)
    (tmp_path / "proxies" / "rec1_eq.mp4").write_bytes(b"x")
    cfg = _Cfg(tmp_path, {"semantic.clip_checkpoint_every": 8,
                          "semantic.clip_batch": 4})

    _fake_embedding_world(monkeypatch, boom_after=16)
    with pytest.raises(KeyboardInterrupt):
        s04.embed_shots(cfg, many_shots)

    idx = tmp_path / "semantic" / "clip_index.json"
    assert idx.exists(), "a kill must not lose everything"
    part = json.loads(idx.read_text())["shot_ids"]
    assert len(part) == 16, f"expected the last checkpoint, got {len(part)}"
    assert np.load(tmp_path / "semantic" / "clip.npy").shape == (16, 4)

    # resume: only the remaining 24 are encoded, and nothing is lost
    calls = _fake_embedding_world(monkeypatch)
    out = s04.embed_shots(cfg, many_shots)
    assert calls["n"] == 24, "a resume must not redo finished work"
    assert out["n_embedded"] == 40
    ids = json.loads(idx.read_text())["shot_ids"]
    assert len(ids) == len(set(ids)) == 40


def test_the_checkpoint_is_atomic_so_a_kill_cannot_truncate_it(tmp_path,
                                                               monkeypatch,
                                                               many_shots):
    """np.save straight onto the real path leaves a half-written array if the
    process dies inside it, which reads back as corrupt rather than as short."""
    (tmp_path / "proxies").mkdir(exist_ok=True)
    (tmp_path / "proxies" / "rec1_eq.mp4").write_bytes(b"x")
    cfg = _Cfg(tmp_path, {"semantic.clip_checkpoint_every": 8,
                          "semantic.clip_batch": 4})
    _fake_embedding_world(monkeypatch)
    s04.embed_shots(cfg, many_shots)
    sem = tmp_path / "semantic"
    assert not list(sem.glob("*.tmp")), "temporaries must not survive"
    np.load(sem / "clip.npy")            # raises if truncated
