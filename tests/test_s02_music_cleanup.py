"""S02.7 leaves no orphan music rows behind, and does not trip foreign keys."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db
from nepal.stages.s02_spine import remove_stale_music


def seed(conn, track_id: str) -> None:
    conn.execute("INSERT INTO music_tracks(track_id, title) VALUES (?,?)",
                 (track_id, track_id))
    conn.execute("INSERT INTO music_sections(section_id, track_id, start_s, end_s,"
                 " energy, is_swell) VALUES (?,?,?,?,?,?)",
                 (f"{track_id}_s0", track_id, 0.0, 60.0, 0.5, 1))
    conn.executemany("INSERT INTO beats(track_id, t_s, is_downbeat) VALUES (?,?,?)",
                     [(track_id, float(i), int(i % 4 == 0)) for i in range(8)])
    conn.commit()


@pytest.fixture()
def conn(tmp_path):
    c = db.init(tmp_path / "nepal.sqlite")
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1, \
        "the regression this guards only exists with foreign keys enforced"
    yield c
    c.close()


def test_stale_tracks_and_their_children_are_removed(conn):
    seed(conn, "keep_me")
    seed(conn, "left_over")

    assert remove_stale_music(conn, {"keep_me"}) == 1

    assert [r[0] for r in conn.execute("SELECT track_id FROM music_tracks")] == ["keep_me"]
    assert conn.execute("SELECT COUNT(*) FROM music_sections "
                        "WHERE track_id='left_over'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM beats "
                        "WHERE track_id='left_over'").fetchone()[0] == 0
    # the surviving track keeps everything that belongs to it
    assert conn.execute("SELECT COUNT(*) FROM music_sections "
                        "WHERE track_id='keep_me'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM beats "
                        "WHERE track_id='keep_me'").fetchone()[0] == 8


def test_no_section_is_left_pointing_at_a_deleted_track(conn):
    """Deleting the parent first raised IntegrityError and aborted the stage."""
    for tid in ("a", "b", "c"):
        seed(conn, tid)

    remove_stale_music(conn, {"c"})

    orphans = conn.execute(
        "SELECT COUNT(*) FROM music_sections s "
        "LEFT JOIN music_tracks t ON t.track_id = s.track_id "
        "WHERE t.track_id IS NULL").fetchone()[0]
    assert orphans == 0
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_nothing_stale_is_a_no_op(conn):
    seed(conn, "only")
    assert remove_stale_music(conn, {"only"}) == 0
    assert conn.execute("SELECT COUNT(*) FROM beats").fetchone()[0] == 8
