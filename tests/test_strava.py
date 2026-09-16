"""Strava: the trek at one fix per second, with heart rate."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal.spine import strava
from nepal.spine.gps import GpsPoint, merge_points

UTC = timezone.utc
T0 = datetime(2024, 5, 6, 4, 2, 34, tzinfo=UTC)

CSV = (
    "Activity ID,Activity Date,Activity Name,Activity Type,Activity Description,"
    "Elapsed Time,Distance,Max Heart Rate,Relative Effort,Commute,Activity Private Note,"
    "Activity Gear,Filename,Athlete Weight,Bike Weight,Elapsed Time,Moving Time,Distance,"
    "Max Speed,Average Speed,Elevation Gain,Elevation Loss,Elevation Low,Elevation High,"
    "Max Grade,Average Grade,Average Positive Grade,Average Negative Grade,Max Cadence,"
    "Average Cadence,Max Heart Rate,Average Heart Rate\n"
    '11343979900,"May 6, 2024, 4:02:34 AM",Larkya Pass to Bimtang,Hike,,18485,12.36,128,'
    ',false,,,activities/12116271787.fit.gz,,,18485,15099,12360.4,1.781,0.819,718.8,'
    ',3705,5154.2,,,,,,,128,111\n'
)


def test_activities_csv_takes_the_metric_columns_not_the_display_ones(tmp_path):
    """Strava writes 'Distance' twice: 12.36 (km, for people) and 12360.4 (m).
    The last occurrence is the one in base units."""
    p = tmp_path / "activities.csv"
    p.write_text(CSV, encoding="utf-8")
    acts = strava.read_activities_csv(p)
    assert len(acts) == 1
    a = acts[0]
    assert a.activity_id == "11343979900"
    assert a.name == "Larkya Pass to Bimtang" and a.kind == "Hike"
    assert a.start_utc == T0
    assert a.elapsed_s == 18485 and a.moving_s == 15099
    assert a.distance_m == pytest.approx(12360.4)
    assert a.gain_m == pytest.approx(718.8)
    assert a.hr_max == 128 and a.hr_avg == 111
    assert a.filename == "activities/12116271787.fit.gz"
    assert a.end_utc == T0 + timedelta(seconds=18485)


def _rec(seconds, lat_deg, lon_deg, ele=4000.0, hr=120):
    semi = 2 ** 31 / 180.0
    return {"timestamp": T0 + timedelta(seconds=seconds),
            "position_lat": int(lat_deg * semi), "position_long": int(lon_deg * semi),
            "enhanced_altitude": ele, "heart_rate": hr}


def test_points_convert_semicircles_and_carry_hr_and_baro():
    pts = strava.points_from_records([_rec(0, 28.65, 84.62, 5106.0, 131)], activity_id="A")
    assert len(pts) == 1
    p = pts[0]
    assert p.lat == pytest.approx(28.65, abs=1e-6) and p.lon == pytest.approx(84.62, abs=1e-6)
    assert p.ele == 5106.0 and p.hr == 131.0
    assert p.source == "strava" and p.activity_id == "A" and p.ts == T0


def test_points_skip_records_without_a_fix_and_downsample():
    recs = [_rec(i, 28.65 + i * 1e-5, 84.62) for i in range(10)]
    recs.insert(3, {"timestamp": T0 + timedelta(seconds=3), "heart_rate": 100})   # indoor gap
    pts = strava.points_from_records(recs, activity_id="A", sample_s=5.0)
    assert [int((p.ts - T0).total_seconds()) for p in pts] == [0, 5]


def test_points_accept_degrees_if_a_decoder_already_converted():
    pts = strava.points_from_records([{"timestamp": T0, "position_lat": 28.65,
                                       "position_long": 84.62}], activity_id="A")
    assert pts[0].lat == pytest.approx(28.65)


def test_naive_timestamps_are_utc():
    pts = strava.points_from_records([{"timestamp": T0.replace(tzinfo=None),
                                       "position_lat": 28.65, "position_long": 84.62}],
                                     activity_id="A")
    assert pts[0].ts.tzinfo is not None and pts[0].ts == T0


def test_strava_outranks_a_photo_fix_at_the_same_second():
    photo = GpsPoint(T0, 28.0, 84.0, None, "phone_keller")
    watch = GpsPoint(T0, 28.65, 84.62, 5106.0, "strava", None, 130.0, "A")
    merged = merge_points([[photo], [watch]])
    assert merged[0].source == "strava"


REAL = os.environ.get("NEPAL_STRAVA_DIR")


@pytest.mark.skipif(not REAL, reason="set NEPAL_STRAVA_DIR to the strava/ export to run")
def test_real_export_decodes_every_activity():
    """Verify at the real boundary: fitdecode against the actual files."""
    acts, pts, rep = strava.load_strava(pathlib.Path(REAL), sample_s=5.0)
    assert len(acts) >= 10
    assert len(pts) > 10_000
    assert all(a.activity_id in rep["points_per_activity"] for a in acts)
    hrs = [p.hr for p in pts if p.hr is not None]
    assert hrs and 40 < min(hrs) and max(hrs) < 220
    assert max(p.ele for p in pts if p.ele is not None) > 5000
