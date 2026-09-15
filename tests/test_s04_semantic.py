"""S04.1 -- which shots get an embedding.

The bug this guards: S04.1 asked for ``status = 'candidate'``, but S05
promotes the shots it picks to 'shortlisted'. Once a cut existed, the 400
shots the film is actually made of no longer matched -- so a later S04.1 run
would embed 1232 rows and skip exactly the ones MMR needs to compare.
"""
import sys, pathlib
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
