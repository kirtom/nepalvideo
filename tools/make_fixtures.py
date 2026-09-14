#!/usr/bin/env python3
"""Generate a synthetic nepal_data/ tree with known ground truth.

Used by the test suite and handy for validating an install before pointing the
pipeline at several hundred GB of irreplaceable footage.

Ground truth it plants, so tests can assert recovery rather than plausibility:

  * ``TRUE_FOV``            -- camera clips are projected to dual-fisheye at
                               this lens FOV, so S01.4 must recover it.
  * ``TRUE_KELLER_OFFSET``  -- keller's phone clock is wrong by this many
                               seconds, so S01.5 must recover it.
  * chapter-split camera files that must collapse into one recording.
  * phone photos carrying real EXIF GPS along a plausible altitude profile.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRUE_FOV = 196
# Mirrors the real corpus: keller's phone agrees with GPS and becomes the
# reference clock, kulikov's drifts by seconds, and the camera's battery went
# flat so its clock is days out -- which only the coarse stage can find.
TRUE_KELLER_OFFSET = 0.0
TRUE_KULIKOV_OFFSET = 12.0
TRUE_CAMERA_OFFSET = 14 * 86400 + 63.0
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))
TREK_START = datetime(2023, 10, 15, 6, 0, 0, tzinfo=NEPAL_TZ)

# a plausible Everest-region ascent: lon/lat/elev per day
ROUTE = [
    (27.6869, 86.7314, 2860),   # Lukla
    (27.8069, 86.7133, 3440),   # Namche
    (27.8272, 86.7617, 3860),   # Tengboche
    (27.8917, 86.8250, 4410),   # Dingboche
    (27.9500, 86.8200, 4940),   # Lobuche
    (28.0026, 86.8528, 5364),   # Base Camp
    (27.9950, 86.8280, 5545),   # Kala Patthar -- high point
    (27.8069, 86.7133, 3440),   # back to Namche
]


def _muxer(dest: Path) -> list[str]:
    """.insv and .lrv are MP4 containers with extra Insta360 boxes; ffmpeg will
    not infer a muxer from those extensions, so name it explicitly."""
    return ["-f", "mp4"] if dest.suffix.lower() in (".insv", ".lrv") else []


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stderr[-1500:]}")


def dual_fisheye_clip(dest: Path, seconds: float, seed: int, with_audio: bool = True,
                      audio_seed: int | None = None, audio_offset: float = 0.0,
                      width: int = 1440) -> None:
    """Equirect test pattern -> dual fisheye at TRUE_FOV, so the seam is real."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"v360=input=e:output=dfisheye:h_fov={TRUE_FOV}:v_fov={TRUE_FOV},"
          f"scale={width}:{width // 2}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-f", "lavfi", "-i", f"testsrc2=size=1920x960:rate=30:duration={seconds}"]
    if with_audio:
        # a deterministic "world" sound both devices can hear
        a = audio_seed if audio_seed is not None else seed
        cmd += ["-f", "lavfi", "-i",
                f"anoisesrc=seed={a}:amplitude=0.5:duration={seconds + audio_offset + 5}"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-af", f"atrim=start={audio_offset},asetpts=PTS-STARTPTS",
                "-c:a", "aac", "-ac", "1", "-ar", "48000", "-shortest"]
    cmd += ["-t", str(seconds), *_muxer(dest), "-y", str(dest)]
    run(cmd)


def flat_clip(dest: Path, seconds: float, seed: int, audio_seed: int | None = None,
              audio_offset: float = 0.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    a = audio_seed if audio_seed is not None else seed
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
         "-f", "lavfi", "-i",
         f"anoisesrc=seed={a}:amplitude=0.5:duration={seconds + audio_offset + 5}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-af", f"atrim=start={audio_offset},asetpts=PTS-STARTPTS",
         "-c:a", "aac", "-ac", "1", "-ar", "48000", "-shortest",
         "-t", str(seconds), "-y", str(dest)])


def stamp_video(dest: Path, when_utc: datetime) -> None:
    """Give a generated clip a real creation time.

    ffmpeg writes 0000:00:00, which the manifest correctly treats as absent.
    Real camera and phone footage always carries one. The literal UTC reading
    is stored, so parsing it back as UTC round-trips.
    """
    stamp = when_utc.astimezone(timezone.utc).strftime("%Y:%m:%d %H:%M:%S")
    run(["exiftool", "-overwrite_original", "-q",
         f"-QuickTime:CreateDate={stamp}", f"-QuickTime:ModifyDate={stamp}",
         f"-Track1:MediaCreateDate={stamp}", str(dest)])


_PHOTO_TEMPLATE: Path | None = None


def _photo_template(tmp: Path) -> Path:
    """Encode one JPEG and reuse it. Re-encoding per photo dominates fixture
    build time for no benefit -- only the EXIF differs between them."""
    global _PHOTO_TEMPLATE
    if _PHOTO_TEMPLATE is None or not _PHOTO_TEMPLATE.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        t = tmp / "_template.jpg"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "lavfi", "-i", "testsrc2=size=1600x1200:duration=1:rate=1",
             "-frames:v", "1", "-y", str(t)])
        _PHOTO_TEMPLATE = t
    return _PHOTO_TEMPLATE


def photo(dest: Path, seed: int, when: datetime, lat: float, lon: float, alt: float,
          *, gps_time: datetime | None = None) -> None:
    """A phone photo with EXIF GPS.

    ``gps_time`` writes GPSDateTime, which is satellite time. Where it differs
    from DateTimeOriginal the device clock is provably wrong, and that is the
    evidence S01 uses to choose its reference clock.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_photo_template(Path(tempfile.gettempdir()) / "nepal_fixture_tmp"), dest)
    stamp = when.strftime("%Y:%m:%d %H:%M:%S")
    off = when.strftime("%z")
    off = f"{off[:3]}:{off[3:]}" if off else "+00:00"
    args = ["exiftool", "-overwrite_original", "-q",
            f"-DateTimeOriginal={stamp}", f"-CreateDate={stamp}",
            f"-OffsetTimeOriginal={off}",
            f"-GPSLatitude={abs(lat)}", f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
            f"-GPSLongitude={abs(lon)}", f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
            f"-GPSAltitude={alt}", "-GPSAltitudeRef=0",
            "-Make=Apple", "-Model=iPhone 14 Pro"]
    if gps_time is not None:
        g = gps_time.astimezone(timezone.utc)
        args.append(f"-GPSDateStamp={g.strftime('%Y:%m:%d')}")
        args.append(f"-GPSTimeStamp={g.strftime('%H:%M:%S')}")
    args.append(str(dest))
    run(args)


def music_track(dest: Path, seconds: float, freq: int, tempo: float, *,
                swell_period: float = 20.0, click_gain: float = 0.35,
                artist: str = "Fixture Artist", title: str | None = None) -> None:
    """Synthetic music with real rhythm and real dynamics.

    Pure tones are useless for testing S02.7: librosa finds no beats in a
    continuous sine, so tempo comes back 0, the beat grid is empty and dynamic
    range is ~0.0002. Every feature the act assignment depends on reads as
    degenerate, and the audio path looks like it works when nothing has been
    exercised.

    So each track gets percussive noise bursts at the requested tempo, which
    beat tracking can lock onto, and a slow amplitude swell, which gives
    energy_p95 - energy_p10 something to measure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    beat_period = 60.0 / max(tempo, 1.0)
    expr = (
        f"0.45*sin(2*PI*{freq}*t)"
        f"*(0.25+0.75*pow(sin(2*PI*t/{swell_period}),2))"
        f"+0.25*sin(2*PI*{freq * 2}*t)*pow(sin(2*PI*t/{swell_period}),2)"
        f"+{click_gain}*random(0)*exp(-9*mod(t,{beat_period:.4f}))"
    )
    # Commas inside the expression (from pow() and mod()) are filter separators
    # to ffmpeg's graph parser and must be escaped, or it reads the tail of the
    # expression as a second filter and fails on "No option name near".
    escaped = expr.replace("\\", "\\\\").replace(",", "\\,")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-f", "lavfi", "-i", f"aevalsrc={escaped}:d={seconds}:s=44100",
         "-c:a", "libmp3lame", "-b:a", "192k",
         "-metadata", f"artist={artist}",
         "-metadata", f"title={title or dest.stem}",
         "-y", str(dest)])


def build(root: Path, *, quick: bool = False) -> dict:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    dur = 8.0 if quick else 14.0

    cam = root / "media_from_camera"
    ph = root / "media_from_phones"

    # One take, split into two chapters, plus its .lrv proxy. These must
    # collapse into a single recording in S01.2.
    # The camera's own clock is wrong, so both its filenames and its embedded
    # timestamps are shifted back by TRUE_CAMERA_OFFSET.
    def cam_clock(true_time: datetime) -> datetime:
        return true_time - timedelta(seconds=TRUE_CAMERA_OFFSET)

    take = TREK_START + timedelta(days=2, hours=3)
    stamp = cam_clock(take).strftime("%Y%m%d_%H%M%S")
    # the .lrv is a genuine lower-resolution proxy of the .insv, as on the
    # device -- same content, different bytes, so a distinct asset_id
    for name, w in [(f"VID_{stamp}_00_001.insv", 1440), (f"LRV_{stamp}_01_001.lrv", 720)]:
        dual_fisheye_clip(cam / name, dur, seed=1, audio_seed=900, width=w)
        stamp_video(cam / name, cam_clock(take))
    dual_fisheye_clip(cam / f"VID_{stamp}_00_002.insv", dur, seed=2, audio_seed=901)
    stamp_video(cam / f"VID_{stamp}_00_002.insv", cam_clock(take + timedelta(seconds=dur)))

    # N camera/phone pairs that recorded the same moment, so S01.5 has enough
    # confident pairs to clear min_confident_pairs instead of falling back.
    pairs = 4 if quick else 6
    for k in range(pairs):
        shared_sound_starts = TREK_START + timedelta(days=4 + k, hours=2)
        cstamp = cam_clock(shared_sound_starts).strftime("%Y%m%d_%H%M%S")
        aseed = 700 + k
        cname = f"VID_{cstamp}_00_001.insv"
        dual_fisheye_clip(cam / cname, dur, seed=30 + k, audio_seed=aseed, audio_offset=0.0)
        stamp_video(cam / cname, cam_clock(shared_sound_starts))

        for who, skew in (("keller", TRUE_KELLER_OFFSET), ("kulikov", TRUE_KULIKOV_OFFSET)):
            lead = 2.0 + k * 0.5          # how far into the shared sound the phone starts
            true_start = shared_sound_starts + timedelta(seconds=lead)
            reported = true_start - timedelta(seconds=skew)
            dest = ph / who / f"VID_pair{k}.mp4"
            flat_clip(dest, dur, seed=40 + k, audio_seed=aseed, audio_offset=lead)
            stamp_video(dest, reported)

    # a phone clip with no shared audio, to prove unrelated pairs get rejected
    flat_clip(ph / "kulikov" / "VID_solo.mp4", dur, seed=99, audio_seed=555)
    stamp_video(ph / "kulikov" / "VID_solo.mp4",
                TREK_START + timedelta(days=4, hours=2, seconds=30))

    # Phone photos: the geolocation spine, and the signal the coarse clock
    # aligner locks onto. Counts vary per day on purpose -- an identical daily
    # routine leaves the histogram ambiguous against a whole-day shift.
    per_day = [5, 3, 8, 2, 6, 9, 4, 7] if not quick else [4, 2, 6, 2, 5, 7, 3, 5]
    for day, (lat, lon, alt) in enumerate(ROUTE):
        n = per_day[day % len(per_day)]
        for i in range(n):
            # offsets from TREK_START (06:00 Nepal): a real trekking day runs
            # from first light to late afternoon, and the camera films inside it
            hour = (0, 1, 2, 3, 5, 8, 9, 10, 11)[i % 9]
            when = TREK_START + timedelta(days=day, hours=hour, minutes=(i * 17) % 60)
            who = "keller" if i % 2 == 0 else "kulikov"
            skew = TRUE_KELLER_OFFSET if who == "keller" else TRUE_KULIKOV_OFFSET
            reported = when - timedelta(seconds=skew)
            # GPSDateTime is satellite time: it records the TRUE instant, so it
            # differs from the device reading by exactly the clock error.
            photo(ph / who / f"IMG_{day:02d}{i:02d}.jpg", day * 10 + i,
                  reported, lat + i * 0.001, lon + i * 0.001, alt + i * 5,
                  gps_time=when)

    chat = root / "chat_export"
    rv = chat / "round_video_messages" / "video_1@15-10-2023.mp4"
    flat_clip(rv, 3.0, seed=21)
    stamp_video(rv, (TREK_START + timedelta(days=2, hours=8)).astimezone(timezone.utc))
    photo(chat / "photos" / "photo_1@15-10-2023.jpg", 31,
          TREK_START - timedelta(days=20), 55.75, 37.61, 150)
    # Telegram strips EXIF -- emulate that, it is the whole reason S02.5 exists
    run(["exiftool", "-overwrite_original", "-q", "-all=",
         str(chat / "photos" / "photo_1@15-10-2023.jpg")])
    (chat / "files").mkdir(parents=True, exist_ok=True)
    _write_gpx(chat / "files" / "route.gpx")
    _write_result_json(chat / "result.json")

    # Distinct parameters per track: identical audio would be byte-identical and
    # the content-hash asset_id would legitimately collapse the two.
    # Tracks span the act targets: quiet/dark/slow through loud/bright/fast,
    # with Act 1 and Act 5 sharing an artist so the callback bonus can fire.
    dur_m = 40 if quick else 100
    music_track(root / "music" / "01_sparse_piano.mp3", dur_m, 180, 58,
                click_gain=0.10, swell_period=26, artist="Quiet Piano", title="Says")
    music_track(root / "music" / "02_strings.mp3", dur_m + 1, 320, 88,
                click_gain=0.22, swell_period=20, artist="Warm Strings", title="Near Light")
    music_track(root / "music" / "03_swell.mp3", dur_m + 2, 430, 96,
                click_gain=0.30, swell_period=11, artist="Post Rock", title="Longest Year")
    music_track(root / "music" / "04_peak.mp3", dur_m + 3, 880, 138,
                click_gain=0.55, swell_period=9, artist="Loud Crescendo", title="The Mountain")
    music_track(root / "music" / "05_return.mp3", dur_m + 4, 186, 60,
                click_gain=0.12, swell_period=24, artist="Quiet Piano", title="Ambre")
    # A sixth track so the library has more to choose from than acts to fill
    music_track(root / "music" / "06_mid.mp3", dur_m, 300, 110,
                click_gain=0.34, swell_period=16, artist="Cold Orchestral",
                title="Thin Air")

    truth = {
        "true_fov": TRUE_FOV,
        "reference_clock": "phone_keller",
        "true_keller_offset_s": TRUE_KELLER_OFFSET,
        "true_kulikov_offset_s": TRUE_KULIKOV_OFFSET,
        "true_camera_offset_s": TRUE_CAMERA_OFFSET,
        "expected_camera_recordings_min": 3,
        "route_points": len(ROUTE),
        "trek_start_utc": TREK_START.astimezone(timezone.utc).isoformat(),
    }
    (root / "_ground_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def _write_playlist_csv(dest: Path) -> None:
    """An Exportify-shaped playlist, including Russian-language entries so the
    exclusion path is exercised end to end."""
    rows = [
        ("Says", "Nils Frahm", 275000, 62, 0.09, -22.0, 0.20, 0.95, 0.94, 9, 0),
        ("Near Light", "Olafur Arnalds", 250000, 88, 0.28, -17.0, 0.34, 0.80, 0.90, 2, 1),
        ("Longest Year", "Hammock", 380000, 95, 0.52, -12.0, 0.30, 0.30, 0.92, 4, 1),
        ("The Mountain", "This Will Destroy You", 420000, 132, 0.88, -6.0, 0.42, 0.05, 0.88, 7, 1),
        ("Ambre", "Nils Frahm", 300000, 60, 0.12, -21.0, 0.22, 0.93, 0.95, 9, 0),
        ("Sudno", "Molchat Doma", 200000, 140, 0.80, -7.0, 0.70, 0.10, 0.10, 5, 0),
        ("Группа крови", "Кино", 280000, 120, 0.75, -8.0, 0.65, 0.20, 0.05, 0, 1),
    ]
    header = ("Track URI,Track Name,Artist Name(s),Album Name,Duration (ms),Tempo,"
              "Energy,Loudness,Danceability,Acousticness,Instrumentalness,Key,Mode")
    lines = [header]
    for i, (t, a, ms, tempo, en, loud, dance, acou, instr, key, mode) in enumerate(rows):
        lines.append(f"spotify:track:{i:022d},{t},{a},Album,{ms},{tempo},{en},"
                     f"{loud},{dance},{acou},{instr},{key},{mode}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gpx(dest: Path) -> None:
    pts = []
    for day, (lat, lon, alt) in enumerate(ROUTE):
        when = (TREK_START + timedelta(days=day, hours=4)).astimezone(timezone.utc)
        pts.append(f'      <trkpt lat="{lat}" lon="{lon}"><ele>{alt}</ele>'
                   f'<time>{when.strftime("%Y-%m-%dT%H:%M:%SZ")}</time></trkpt>')
    dest.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="fixture">\n  <trk><name>trek</name><trkseg>\n'
        + "\n".join(pts) + "\n  </trkseg></trk>\n</gpx>\n")


def _write_result_json(dest: Path) -> None:
    msgs = []
    mid = 1
    # planning phase
    for d in range(-30, -1, 6):
        for author, text in [("Keller", "Смотри какой маршрут через Гокио"),
                             ("Kulikov", "Пермиты сделаем в Катманду, не парься")]:
            when = TREK_START + timedelta(days=d)
            msgs.append({"id": mid, "type": "message", "date": when.strftime("%Y-%m-%dT%H:%M:%S"),
                         "date_unixtime": str(int(when.timestamp())),
                         "from": author, "from_id": f"user{1 if author=='Keller' else 2}",
                         "text": text})
            mid += 1
    # trek phase
    for d in range(0, len(ROUTE)):
        when = TREK_START + timedelta(days=d, hours=5)
        msgs.append({"id": mid, "type": "message", "date": when.strftime("%Y-%m-%dT%H:%M:%S"),
                     "date_unixtime": str(int(when.timestamp())),
                     "from": "Keller", "from_id": "user1",
                     "text": f"День {d+1}. Высота {ROUTE[d][2]}. Ноги отваливаются."})
        mid += 1
    msgs.append({"id": mid, "type": "message",
                 "date": (TREK_START + timedelta(days=3, hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
                 "date_unixtime": str(int((TREK_START + timedelta(days=3, hours=6)).timestamp())),
                 "from": "Kulikov", "from_id": "user2", "text": "",
                 "photo": "photos/photo_1@15-10-2023.jpg"})
    mid += 1
    msgs.append({"id": mid, "type": "message",
                 "date": (TREK_START + timedelta(days=2, hours=8)).strftime("%Y-%m-%dT%H:%M:%S"),
                 "date_unixtime": str(int((TREK_START + timedelta(days=2, hours=8)).timestamp())),
                 "from": "Kulikov", "from_id": "user2", "text": "",
                 "media_type": "video_message",
                 "file": "round_video_messages/video_1@15-10-2023.mp4"})
    dest.write_text(json.dumps(
        {"name": "Nepal", "type": "personal_chat", "id": 1, "messages": msgs},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default="tests/fixtures/nepal_data")
    ap.add_argument("--quick", action="store_true", help="shorter clips, fewer photos")
    a = ap.parse_args()
    for tool in ("ffmpeg", "exiftool"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is required to build fixtures")
    t = build(Path(a.dest), quick=a.quick)
    print(f"fixtures at {a.dest}\nground truth: {json.dumps(t, indent=2)}")
