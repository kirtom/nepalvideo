"""The checkpoint table must name material the film cannot reach.

A clip outside every act window is a silent deletion unless it is counted, and
on real material one camera with a reset clock held nearly half the footage.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal import db
from nepal.spine.acts import ActBoundary
from nepal.stages.s02_spine import print_unreachable

UTC = timezone.utc
TREK = datetime(2024, 4, 28, tzinfo=UTC)

BOUNDS = [
    ActBoundary(1, TREK - timedelta(days=180), TREK, "message-phase:planning"),
    ActBoundary(2, TREK, TREK + timedelta(days=3), "altitude-changepoint"),
    ActBoundary(3, TREK + timedelta(days=3), TREK + timedelta(days=8), "altitude-changepoint"),
    ActBoundary(4, TREK + timedelta(days=8), TREK + timedelta(days=9), "altitude-changepoint"),
    ActBoundary(5, TREK + timedelta(days=9), TREK + timedelta(days=43), "altitude-changepoint"),
]


@pytest.fixture()
def conn(tmp_path):
    c = db.init(tmp_path / "nepal.sqlite")
    yield c
    c.close()


def add(conn, asset_id, when, *, kind="video360", source="camera",
        dur=30.0, lat=None):
    conn.execute(
        "INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc, "
        "duration_s, lat) VALUES (?,?,?,?,?,?,?)",
        (asset_id, f"raw/{asset_id}", source, kind, when.isoformat(), dur, lat))
    conn.commit()


def test_material_inside_the_acts_is_not_reported(conn, capsys):
    for i in range(5):
        add(conn, f"in{i}", TREK + timedelta(days=4, hours=i))
    assert print_unreachable(conn, BOUNDS) == {"n": 0}
    assert capsys.readouterr().out == ""


def test_a_camera_whose_clock_reset_is_counted_in_minutes_and_share(conn, capsys):
    """Eighteen months late, and holding more footage than the trek itself."""
    for i in range(10):
        add(conn, f"in{i}", TREK + timedelta(days=4, hours=i), dur=60.0)
    later = TREK + timedelta(days=574)
    for i in range(20):
        add(conn, f"lost{i}", later + timedelta(minutes=i), dur=60.0,
            lat=28.55)

    out = print_unreachable(conn, BOUNDS)
    assert out["n"] == 20
    assert out["n_clips"] == 20
    assert out["lost_video_s"] == pytest.approx(1200.0)
    # 20 of 30 minutes of video is unreachable
    assert out["share_pct"] == pytest.approx(66.7, abs=0.2)

    text = capsys.readouterr().out
    assert "20 media asset(s) fall outside every act window" in text
    assert "67% of all" in text
    # a position with a bad date is a recoverable stamp, and the report says so
    assert "20 of them DO carry a GPS position" in text
    assert "nepal diagnose --timestamps" in text


def test_photos_and_clips_are_reported_separately(conn, capsys):
    add(conn, "in0", TREK + timedelta(days=4), dur=60.0)
    later = TREK + timedelta(days=200)
    add(conn, "lost_clip", later, dur=45.0)
    add(conn, "lost_photo", later + timedelta(minutes=1), kind="photo",
        source="phone_a", dur=0.0)

    out = print_unreachable(conn, BOUNDS)
    assert (out["n_clips"], out["n_photos"]) == (1, 1)
    text = capsys.readouterr().out
    assert "1 clip(s), 1 photo(s)" in text
    assert "camera=1" in text and "phone_a=1" in text
    # neither carries a position here, so no recovery hint
    assert "DO carry a GPS position" not in text


def test_planning_material_inside_the_window_still_reaches_act_one(conn):
    add(conn, "planning", TREK - timedelta(days=90), kind="photo", dur=0.0)
    assert print_unreachable(conn, BOUNDS) == {"n": 0}
