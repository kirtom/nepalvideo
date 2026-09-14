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


# -- only cameras chapter a take across files ---------------------------

def _vid(source, name, kind="video_flat", dur=10.0):
    return {"asset_id": name, "s3_key": f"raw/x/{name}", "source": source,
            "kind": kind, "duration_s": dur,
            "created_at": "2024-05-06T08:12:00+00:00"}


def test_phone_clips_are_not_chapters_of_one_recording():
    """A phone names every clip IMG_1234.MOV. The generic rule read that as
    base 'IMG', chapter 1234, and collapsed every video on the phone into one
    recording -- a single 230 MB proxy of hundreds of unrelated clips, 52
    minutes to build, with timestamps measured from the wrong start."""
    from nepal.probe.chapters import group_recordings
    assets = [_vid("phone_kulikov", f"IMG_{1000 + i}.MOV") for i in range(6)]
    assert len(group_recordings(assets)) == 6


def test_telegram_exports_are_separate_messages_not_chapters():
    from nepal.probe.chapters import group_recordings
    assets = [_vid("telegram", f"video ({i}).mp4") for i in range(4)]
    assert len(group_recordings(assets)) == 4


def test_a_camera_lens_pair_is_still_one_recording():
    """The narrowing must not lose the case chaptering exists for."""
    from nepal.probe.chapters import group_recordings
    assets = [_vid("camera", f"VID_20240418_034346_{s}_039.insv", "video360", 7.7)
              for s in ("00", "10")]
    recs = group_recordings(assets)
    assert len(recs) == 1
    assert recs[0].recording_id == "camera_20240418_034346"


def test_real_camera_chapters_still_group():
    from nepal.probe.chapters import group_recordings
    assets = [_vid("camera", f"VID_20240419_022300_00_{i:03d}.insv", "video360")
              for i in (1, 2, 3)]
    assert len(group_recordings(assets)) == 1
