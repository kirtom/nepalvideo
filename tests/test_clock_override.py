"""Gate 1's word on a clock: the operator's offset outranks the solve, and
a moved clock moves the shots that were cut from it."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

from nepal import db
from nepal.config import Config
from nepal.spine import acts as acts_mod
from nepal.stages import s01_probe, s03_process


def test_an_override_is_read_by_source_or_label_and_absent_otherwise():
    cfg = Config({"probe": {"clock": {"offset_overrides_s": {"camera": 1209600,
                                                             "kulikov": -5}}}})
    assert s01_probe.override_offset(cfg, "camera") == 1209600.0
    assert s01_probe.override_offset(cfg, "phone_kulikov") == -5.0
    assert s01_probe.override_offset(cfg, "phone_keller") is None
    assert s01_probe.override_offset(Config({"probe": {"clock": {}}}), "camera") is None
    assert s01_probe.override_offset(Config({}), "camera") is None


def test_shots_follow_their_recording_when_its_clock_moves(tmp_path):
    """A corrected offset reaches the assets and the recordings through
    apply_offsets; the place step carries it on to the shots and their act."""
    conn = db.init(tmp_path / "n.sqlite")
    conn.execute("INSERT INTO recordings(recording_id, source, is_360, start_utc) "
                 "VALUES ('camera_x', 'camera', 0, '2024-05-01T11:25:00+00:00')")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc) "
                 "VALUES ('p1', 'k', 'phone_keller', 'photo', '2024-04-28T04:00:00+00:00')")
    conn.executemany(
        "INSERT INTO shots(shot_id, recording_id, asset_id, media_kind, start_s, end_s, "
        "start_utc, act) VALUES (?,?,?,?,?,?,?,?)",
        [("camera_x#0001", "camera_x", None, "video", 30.0, 40.0,
          "2024-05-01T11:25:30+00:00", 3),
         ("photo_p1", None, "p1", "photo", 0.0, 5.0, "2024-04-28T04:00:00+00:00", 2)])
    bounds = [acts_mod.ActBoundary(2, datetime(2024, 4, 27, tzinfo=timezone.utc),
                                   datetime(2024, 5, 1, 1, tzinfo=timezone.utc), "t"),
              acts_mod.ActBoundary(3, datetime(2024, 5, 1, 1, tzinfo=timezone.utc),
                                   datetime(2024, 5, 6, tzinfo=timezone.utc), "t")]
    assert s03_process.refresh_shot_times(conn, bounds) == 0       # nothing moved yet
    # the camera turns out to be four days early: the recording moves
    conn.execute("UPDATE recordings SET start_utc='2024-04-27T11:25:00+00:00'")
    conn.commit()
    assert s03_process.refresh_shot_times(conn, bounds) == 1
    row = conn.execute("SELECT start_utc, act FROM shots WHERE shot_id='camera_x#0001'").fetchone()
    assert row["start_utc"] == "2024-04-27T11:25:30+00:00" and row["act"] == 2
    photo = conn.execute("SELECT start_utc, act FROM shots WHERE shot_id='photo_p1'").fetchone()
    assert photo["act"] == 2                                          # untouched
