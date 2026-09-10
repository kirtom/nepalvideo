import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone
import pytest

from nepal.probe.manifest import (classify, telegram_subkind, exif_get, parse_exif_datetime,
                                  parse_gps, parse_duration, NEPAL_TZ, walk_media)


@pytest.mark.parametrize("path,source,kind,curve", [
    ("media_from_camera/VID_20231015_143022_00_001.insv", "camera", "video360", "camera"),
    ("media_from_camera/LRV_20231015_143022_01_001.lrv", "camera", "video360", "camera"),
    ("media_from_camera/VID_20231015_150000.mp4", "camera", "video_flat", "camera"),
    ("media_from_phones/keller/IMG_0001.jpg", "phone_keller", "photo", "phone"),
    ("media_from_phones/keller/IMG_0002.HEIC", "phone_keller", "photo", "phone"),
    ("media_from_phones/kulikov/VID_0003.mp4", "phone_kulikov", "video_flat", "phone"),
    ("chat_export/photos/photo_1@15-10-2023.jpg", "telegram", "photo", "telegram"),
    ("chat_export/files/route.gpx", "telegram", "document", "telegram"),
    ("chat_export/result.json", "telegram", "document", "telegram"),
    ("music/nils_frahm_says.mp3", "music", "audio", "camera"),
])
def test_classification(path, source, kind, curve):
    c = classify(path)
    assert (c["source"], c["kind"], c["quality_curve"]) == (source, kind, curve)


def test_telegram_media_never_classed_360_even_with_360_extension():
    """A .lrv that Telegram passed through is compressed flat video, not a
    dual-fisheye source -- routing it into v360 would produce garbage."""
    c = classify("chat_export/video_files/clip.lrv")
    assert c["kind"] == "video_flat"
    assert c["quality_curve"] == "telegram"


def test_telegram_curve_beats_extension_for_photos():
    assert classify("chat_export/photos/x.jpg")["quality_curve"] == "telegram"
    assert classify("media_from_phones/keller/x.jpg")["quality_curve"] == "phone"


def test_round_video_messages_stay_identifiable():
    assert telegram_subkind("chat_export/round_video_messages/v_1.mp4") == "round_video_messages"
    assert telegram_subkind("chat_export/photos/p.jpg") == "photos"
    assert telegram_subkind("media_from_camera/a.insv") is None


def test_windows_separators_are_handled():
    assert classify(r"media_from_phones\keller\IMG_1.jpg")["source"] == "phone_keller"


# -- exif access -------------------------------------------------------

def test_exif_get_finds_tag_in_any_group():
    row = {"ExifIFD:DateTimeOriginal": "2023:10:15 14:30:22"}
    assert exif_get(row, "DateTimeOriginal") == "2023:10:15 14:30:22"
    row2 = {"QuickTime:CreateDate": "2023:10:15 14:30:22"}
    assert exif_get(row2, "DateTimeOriginal", "CreateDate") is not None


def test_exif_get_prefers_first_named_tag():
    row = {"ExifIFD:DateTimeOriginal": "2023:10:15 01:00:00",
           "QuickTime:CreateDate": "2023:10:15 02:00:00"}
    assert "01:00:00" in exif_get(row, "DateTimeOriginal", "CreateDate")


def test_exif_get_skips_null_timestamps():
    row = {"QuickTime:CreateDate": "0000:00:00 00:00:00",
           "ExifIFD:DateTimeOriginal": "2023:10:15 14:30:22"}
    assert exif_get(row, "CreateDate", "DateTimeOriginal") == "2023:10:15 14:30:22"


def test_exif_get_missing_returns_none():
    assert exif_get({"a": 1}, "DateTimeOriginal") is None


# -- datetimes ---------------------------------------------------------

def test_parse_exif_colon_format():
    dt = parse_exif_datetime("2023:10:15 14:30:22")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2023, 10, 15, 14, 30, 22)
    assert dt.tzinfo == timezone.utc


def test_parse_handles_nepal_quarter_hour_offset():
    """+05:45 is the offset that breaks parsers assuming whole hours."""
    dt = parse_exif_datetime("2023:10:15 14:30:22+05:45")
    assert dt.utcoffset() == timedelta(hours=5, minutes=45)
    assert dt.astimezone(timezone.utc).hour == 8
    assert dt.astimezone(timezone.utc).minute == 45


def test_parse_assume_tz_applied_when_absent():
    dt = parse_exif_datetime("2023:10:15 14:30:22", assume_tz=NEPAL_TZ)
    assert dt.utcoffset() == timedelta(hours=5, minutes=45)


def test_parse_explicit_zone_beats_assumption():
    dt = parse_exif_datetime("2023:10:15 14:30:22Z", assume_tz=NEPAL_TZ)
    assert dt.tzinfo == timezone.utc


def test_parse_fractional_seconds_and_iso():
    dt = parse_exif_datetime("2023-10-15T14:30:22.500Z")
    assert dt.microsecond == 500000


def test_parse_rejects_garbage():
    assert parse_exif_datetime("not a date") is None
    assert parse_exif_datetime(None) is None
    assert parse_exif_datetime("0000:00:00 00:00:00") is None


# -- gps ---------------------------------------------------------------

def test_parse_gps_numeric():
    lat, lon, alt = parse_gps({"GPS:GPSLatitude": 27.9881, "GPS:GPSLongitude": 86.925,
                               "GPS:GPSAltitude": 5364.0})
    assert (lat, lon, alt) == (27.9881, 86.925, 5364.0)


def test_parse_gps_null_island_rejected():
    """0,0 is what a stripped or failed fix looks like, not a location."""
    lat, lon, _ = parse_gps({"GPS:GPSLatitude": 0, "GPS:GPSLongitude": 0})
    assert lat is None and lon is None


def test_parse_gps_absent():
    assert parse_gps({}) == (None, None, None)


def test_parse_gps_hemisphere_suffix_fallback():
    lat, lon, _ = parse_gps({"GPSLatitude": "27.9881 N", "GPSLongitude": "86.925 W"})
    assert lat == pytest.approx(27.9881)
    assert lon == pytest.approx(-86.925)


def test_parse_gps_out_of_range_rejected():
    lat, _, _ = parse_gps({"GPSLatitude": 991.0, "GPSLongitude": 10.0})
    assert lat is None


# -- durations ---------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (623.5, 623.5), ("623.5", 623.5), ("0:10:23", 623.0),
    ("10:23", 623.0), ("623.50 s", 623.5), (None, None), ("", None),
])
def test_parse_duration(value, expected):
    assert parse_duration(value) == expected


# -- walking -----------------------------------------------------------

def test_walk_media_skips_hidden_and_sorts(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "2.jpg").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.jpg").write_bytes(b"x")
    names = [p.relative_to(tmp_path).as_posix() for p in walk_media(tmp_path)]
    assert names == ["a.jpg", "b/2.jpg"]


# -- separate-tag timezone offsets -------------------------------------

from nepal.probe.manifest import parse_offset, asset_datetime


@pytest.mark.parametrize("raw,hours,minutes", [
    ("+05:45", 5, 45), ("+0545", 5, 45), ("-08:00", -8, 0), ("+00:00", 0, 0),
])
def test_parse_offset(raw, hours, minutes):
    tz = parse_offset(raw)
    expect = timedelta(hours=hours, minutes=minutes if hours >= 0 else -minutes)
    assert tz.utcoffset(None) == expect


def test_parse_offset_rejects_junk():
    assert parse_offset("") is None and parse_offset(None) is None
    assert parse_offset("unknown") is None


def test_asset_datetime_uses_the_separate_offset_tag():
    """The bug this guards: DateTimeOriginal carries no zone, and defaulting to
    UTC shifts every Nepal photo by 5 h 45 m onto the wrong day."""
    row = {"ExifIFD:DateTimeOriginal": "2023:10:15 09:00:37",
           "ExifIFD:OffsetTimeOriginal": "+05:45"}
    dt = asset_datetime(row)
    assert dt.utcoffset() == timedelta(hours=5, minutes=45)
    assert dt.astimezone(timezone.utc).hour == 3
    assert dt.astimezone(timezone.utc).minute == 15


def test_asset_datetime_inline_zone_wins_over_offset_tag():
    row = {"ExifIFD:DateTimeOriginal": "2023:10:15 09:00:37+01:00",
           "ExifIFD:OffsetTimeOriginal": "+05:45"}
    assert asset_datetime(row).utcoffset() == timedelta(hours=1)


def test_asset_datetime_falls_back_to_assume_tz():
    row = {"QuickTime:CreateDate": "2023:10:15 09:00:37"}
    assert asset_datetime(row, assume_tz=NEPAL_TZ).utcoffset() == timedelta(hours=5, minutes=45)


def test_asset_datetime_defaults_to_utc_with_no_hints():
    row = {"QuickTime:CreateDate": "2023:10:15 09:00:37"}
    assert asset_datetime(row).tzinfo == timezone.utc


def test_asset_datetime_missing():
    assert asset_datetime({"File:FileName": "x.jpg"}) is None
