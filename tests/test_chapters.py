import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.probe.chapters import parse_chapter, group_recordings, check_continuity, Recording


def test_parse_insta360():
    ck = parse_chapter("VID_20231015_143022_00_001.insv")
    assert ck.recording_key == "20231015_143022"
    assert ck.chapter_index == 1
    assert ck.method == "insta360"


def test_parse_lrv_matches_same_key_as_insv():
    a = parse_chapter("VID_20231015_143022_00_002.insv")
    b = parse_chapter("LRV_20231015_143022_01_002.lrv")
    assert a.recording_key == b.recording_key
    assert a.chapter_index == b.chapter_index == 2


def test_parse_gopro():
    ck = parse_chapter("GH010123.MP4")
    assert ck.recording_key == "gopro_0123"
    assert ck.chapter_index == 1


def test_parse_unstructured_returns_none():
    assert parse_chapter("IMG_sunset.jpg") is None


def _asset(aid, name, kind="video360", dur=600.0, created=None, container="insv"):
    return dict(asset_id=aid, s3_key=f"raw/camera/{name}", filename=name,
                source="camera", kind=kind, duration_s=dur,
                created_at=created, created_at_utc=created, container=container)


def test_group_chapters_into_one_recording():
    assets = [
        _asset("a1", "VID_20231015_143022_00_001.insv", created="2023-10-15T14:30:22+00:00"),
        _asset("a2", "VID_20231015_143022_00_002.insv", created="2023-10-15T14:40:22+00:00"),
        _asset("a3", "LRV_20231015_143022_01_001.lrv", container="lrv"),
        _asset("a4", "LRV_20231015_143022_01_002.lrv", container="lrv"),
    ]
    recs = group_recordings(assets)
    assert len(recs) == 1, "chapters + their proxies must form one recording"
    r = recs[0]
    assert r.asset_count == 4
    assert r.is_360
    # duration must not double-count the .lrv proxies
    assert r.duration_s == 1200.0


def test_separate_takes_stay_separate():
    assets = [
        _asset("a1", "VID_20231015_143022_00_001.insv"),
        _asset("a2", "VID_20231015_150000_00_001.insv"),
    ]
    assert len(group_recordings(assets)) == 2


def test_photos_are_not_recordings():
    assets = [_asset("p1", "IMG_0001.jpg", kind="photo", container="jpg")]
    assert group_recordings(assets) == []


def test_continuity_accepts_contiguous_chapters():
    assets = [
        _asset("a1", "VID_20231015_143022_00_001.insv", dur=600.0, created="2023-10-15T14:30:22+00:00"),
        _asset("a2", "VID_20231015_143022_00_002.insv", dur=600.0, created="2023-10-15T14:40:22+00:00"),
    ]
    rec = group_recordings(assets)[0]
    assert check_continuity(rec, {a["asset_id"]: a for a in assets}) == []


def test_continuity_flags_a_real_gap():
    assets = [
        _asset("a1", "VID_20231015_143022_00_001.insv", dur=600.0, created="2023-10-15T14:30:22+00:00"),
        _asset("a2", "VID_20231015_143022_00_002.insv", dur=600.0, created="2023-10-15T14:45:22+00:00"),
    ]
    rec = group_recordings(assets)[0]
    problems = check_continuity(rec, {a["asset_id"]: a for a in assets})
    assert len(problems) == 1 and "300" in problems[0]


def test_continuity_reports_identical_timestamps_separately():
    ts = "2023-10-15T14:30:22+00:00"
    assets = [
        _asset("a1", "VID_20231015_143022_00_001.insv", dur=600.0, created=ts),
        _asset("a2", "VID_20231015_143022_00_002.insv", dur=600.0, created=ts),
    ]
    rec = group_recordings(assets)[0]
    problems = check_continuity(rec, {a["asset_id"]: a for a in assets})
    assert "unverifiable" in problems[0]
