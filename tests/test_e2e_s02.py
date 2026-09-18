"""End-to-end S02 against fixtures with planted ground truth.

Slow: builds real media, then runs S01 and S02. Run with:  pytest -m slow
"""
import json
import shutil
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.slow

needs_tools = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("exiftool")),
    reason="needs ffmpeg and exiftool",
)


@pytest.fixture(scope="module")
def spined(tmp_path_factory):
    from tools.make_fixtures import build
    from nepal.config import Config
    from nepal.stages import s01_probe, s02_spine
    import yaml

    base = tmp_path_factory.mktemp("e2e_s02")
    data = base / "nepal_data"
    truth = build(data, quick=True)

    cfg_data = yaml.safe_load((ROOT / "config" / "pipeline.yaml").read_text())
    cfg_data["project"]["data_root"] = str(data)
    cfg_data["project"]["work_root"] = str(base / "work")
    cfg_data["project"]["db_path"] = str(base / "work" / "db" / "nepal.sqlite")
    # real tiles if tools/fetch_reference.py has been run, absent otherwise
    cfg_data["spine"]["srtm_dir"] = str(ROOT / "data" / "srtm")
    cfg_data["spine"]["geonames_path"] = str(ROOT / "data" / "geonames" / "NP.txt")
    cfg_path = base / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))

    cfg = Config.load(cfg_path)
    s01_probe.run(cfg, skip_fov=True)
    report = s02_spine.run(cfg, skip_asr=True)
    return truth, report, cfg


# -- S02.1 the geolocation spine ---------------------------------------

@needs_tools
def test_gps_track_is_built_from_photos_and_gpx(spined):
    _, report, _ = spined
    t = report["gps_track"]
    assert t["n_points"] > 20
    assert t["n_from_photos"] > 0, "phone EXIF is the spine"
    assert t["n_from_gpx"] > 0, "the shared route file must be preferred where present"


@needs_tools
def test_track_is_time_ordered_and_unique(spined):
    from nepal import db
    _, _, cfg = spined
    conn = db.init(cfg.db_path)
    rows = [r["ts_utc"] for r in conn.execute("SELECT ts_utc FROM gps_points ORDER BY ts_utc")]
    conn.close()
    assert rows == sorted(rows)
    assert len(rows) == len(set(rows)), "ts_utc is the primary key -- no duplicates"


# -- S02.2 geotagging --------------------------------------------------

@needs_tools
def test_camera_recordings_are_placed_by_interpolation(spined):
    """The camera has no GPS. After the clock correction its recordings must
    fall inside the phone-derived track and pick up a position."""
    from nepal import db
    _, _, cfg = spined
    conn = db.init(cfg.db_path)
    placed = conn.execute("SELECT COUNT(*) n FROM assets WHERE source='camera' "
                          "AND lat IS NOT NULL").fetchone()["n"]
    total = conn.execute("SELECT COUNT(*) n FROM assets WHERE source='camera' "
                         "AND created_at_utc IS NOT NULL").fetchone()["n"]
    conn.close()
    assert total > 0
    assert placed >= total * 0.5, f"only {placed}/{total} camera assets placed"


@needs_tools
def test_planning_photos_are_refused_not_extrapolated(spined):
    """A Telegram photo from a month before the trek must stay unplaced rather
    than be snapped to the first GPS point."""
    _, report, _ = spined
    assert report["geotag"]["n_refused"] > 0


# -- S02.3 altitude ----------------------------------------------------

# The fixture trek is Lukla to Kala Patthar, which sits on these two tiles;
# the reference fetch covers the real trek's extent (Manaslu), so the
# Everest tiles are present only where somebody fetched them deliberately.
_FIXTURE_TILES = ("N27E086", "N28E086")


@needs_tools
@pytest.mark.skipif(
    not all((ROOT / "data" / "srtm" / f"{t}.hgt").exists() for t in _FIXTURE_TILES),
    reason=f"fixture tiles {_FIXTURE_TILES} not fetched (the real trek is elsewhere)")
def test_altitudes_come_from_the_dem_and_climb(spined):
    from nepal import db
    _, _, cfg = spined
    conn = db.init(cfg.db_path)
    rows = [r["alt_dem_m"] for r in conn.execute(
        "SELECT alt_dem_m FROM assets WHERE alt_dem_m IS NOT NULL")]
    conn.close()
    assert rows, "no altitudes resolved"
    assert min(rows) > 1000 and max(rows) < 9000, "implausible elevations for Nepal"
    assert max(rows) - min(rows) > 1000, "the ascent should span real elevation"


# -- S02.5 telegram ----------------------------------------------------

@needs_tools
def test_messages_are_split_across_phases(spined):
    _, report, _ = spined
    phases = report["telegram"]["phases"]
    assert phases.get("planning", 0) > 0, "Act 1 needs planning-phase messages"
    assert phases.get("trek", 0) > 0


@needs_tools
def test_caption_cards_are_identified(spined):
    """Act 1's on-screen text comes from here."""
    _, report, _ = spined
    assert report["telegram"]["n_cards"] > 0


@needs_tools
def test_telegram_media_is_linked_to_assets(spined):
    _, report, _ = spined
    assert report["telegram"]["n_media_linked"] > 0


@needs_tools
def test_vocabulary_is_extracted_for_the_caption_prompt(spined):
    _, report, cfg = spined
    assert report["telegram"]["n_vocab"] > 0
    f = cfg.work_root / "vocab" / "telegram_vocab.json"
    assert f.exists() and json.loads(f.read_text())


# -- S02.7 music from audio files --------------------------------------

@needs_tools
def test_music_comes_from_audio_files_with_real_features(spined):
    """Audio gives what a playlist cannot: real beat grids and swells, which S06
    needs to snap cuts, and real dynamic range, which Act 3's target leans on."""
    _, report, _ = spined
    m = report["music"]
    if m.get("skipped"):
        pytest.skip(m["skipped"])
    assert m["source"] == "audio"
    assert m["n_tracks"] >= 5


@needs_tools
def test_tracks_carry_measured_tempo_and_dynamics(spined):
    from nepal import db
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    conn = db.init(cfg.db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT track_id, tempo_bpm, energy_p95, energy_p10, centroid, key_est "
        "FROM music_tracks")]
    conn.close()
    assert rows
    assert any(r["tempo_bpm"] > 30 for r in rows), \
        "beat tracking found no tempo in any track"
    assert any((r["energy_p95"] - r["energy_p10"]) > 0.01 for r in rows), \
        "no track has measurable dynamic range"
    assert all(r["key_est"] for r in rows)


@needs_tools
def test_beat_grids_are_populated(spined):
    from nepal import db
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    conn = db.init(cfg.db_path)
    n = conn.execute("SELECT COUNT(*) n FROM beats").fetchone()["n"]
    conn.close()
    assert n > 0, "no beats stored -- S06 has nothing to snap cuts to"


@needs_tools
def test_act_assignment_spans_quiet_to_loud(spined):
    """Sparse and quiet opens; the loudest thing peaks at Act 4."""
    from nepal import db
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    a = report["music"]["assignment"]
    conn = db.init(cfg.db_path)
    energy = {r["track_id"]: r["energy_mean"] for r in
              conn.execute("SELECT track_id, energy_mean FROM music_tracks")}
    centroid = {r["track_id"]: r["centroid"] for r in
                conn.execute("SELECT track_id, centroid FROM music_tracks")}
    conn.close()
    assert centroid[a["4"]] > centroid[a["1"]], \
        f"Act 4 should be brighter than Act 1: {a['4']} vs {a['1']}"


@needs_tools
def test_act_five_echoes_act_one_by_artist(spined):
    """The callback is a design requirement, not an accident. Matched on the
    artist tag, which is why ID3 is read rather than filenames parsed."""
    from nepal import db
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    a = report["music"]["assignment"]
    conn = db.init(cfg.db_path)
    rows = {r["track_id"]: r["title"] for r in
            conn.execute("SELECT track_id, title FROM music_tracks")}
    conn.close()
    assert a["5"] != a["1"], "the callback should be a different track"
    assert report["music"]["callback_bonus"] > 0, \
        f"no callback bonus: act1={a['1']} act5={a['5']}"


# -- S02.8 the music map -----------------------------------------------

@needs_tools
def test_music_map_acts_are_contiguous_and_on_target(spined):
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    m = json.loads((cfg.work_root / "music" / "music_map.json").read_text())
    assert abs(m["total_duration_s"] - 1200) <= 30
    assert len(m["acts"]) == 5
    assert all(a["swells"] for a in m["acts"])
    for prev, nxt in zip(m["acts"], m["acts"][1:]):
        assert prev["t_end"] == pytest.approx(nxt["t_start"])


@needs_tools
def test_acts_are_filled_with_track_segments(spined):
    """Acts are longer than songs, so each carries a sequence of segments."""
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    m = json.loads((cfg.work_root / "music" / "music_map.json").read_text())
    for act in m["acts"]:
        assert act["segments"], f"act {act['act']} has no segments"
        assert act["segments"][0]["t_in"] == 0.0
        assert act["segments"][-1]["t_end"] == pytest.approx(
            act["t_end"] - act["t_start"], abs=0.1)


@needs_tools
def test_a_short_fixture_library_is_correctly_reported_as_too_thin(spined):
    """The fixture ships about four minutes of music for a twenty-minute film.
    Acceptance must say so rather than pass -- this is the check working, not
    failing."""
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    m = json.loads((cfg.work_root / "music" / "music_map.json").read_text())
    assert m["library_duration_s"] < 1200
    problems = report["music"]["music_map_problems"]
    assert any("must repeat" in p for p in problems), problems


@needs_tools
def test_silence_window_follows_the_act_four_peak(spined):
    """The single most powerful move in the film, per the brief."""
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")
    m = json.loads((cfg.work_root / "music" / "music_map.json").read_text())
    act4 = next(a for a in m["acts"] if a["act"] == 4)
    sw = m["silence_window"]
    assert sw["t_start"] == pytest.approx(act4["t_end"])
    # lengthened from 3 s: a hard cut to silence needs room to land
    assert sw["t_end"] - sw["t_start"] == pytest.approx(
        float(cfg.get("assemble.silence_window_s")), abs=0.01)


# -- act boundaries ----------------------------------------------------

@needs_tools
def test_five_contiguous_act_boundaries(spined):
    _, report, _ = spined
    b = report["acts"]["boundaries"]
    assert [x["act"] for x in b] == [1, 2, 3, 4, 5]
    for prev, nxt in zip(b, b[1:]):
        assert prev["end_utc"] == nxt["start_utc"]


@needs_tools
def test_act_boundaries_used_the_altitude_profile(spined):
    """Not the even-split fallback -- altitude is the dramatic axis."""
    _, report, _ = spined
    methods = " ".join(x["method"] for x in report["acts"]["boundaries"])
    assert "changepoint" in methods, f"fell back to: {methods}"


@needs_tools
def test_summit_act_is_narrow(spined):
    from datetime import datetime
    _, report, _ = spined
    b = {x["act"]: x for x in report["acts"]["boundaries"]}
    span = (datetime.fromisoformat(b[4]["end_utc"])
            - datetime.fromisoformat(b[4]["start_utc"])).total_seconds()
    assert span < 36 * 3600, "Act 4 is 1.5-2 min of film; it must be a narrow band"


# -- resumability and the checkpoint table -----------------------------

@needs_tools
def test_rerun_skips_completed_steps(spined):
    from nepal.stages import s02_spine
    _, _, cfg = spined
    again = s02_spine.run(cfg, skip_asr=True)
    assert again["music"] == {"skipped": "already done"}


@needs_tools
def test_chronology_table_renders(spined, capsys):
    from nepal.stages import s02_spine
    _, _, cfg = spined
    assert s02_spine.print_chronology(cfg) == 0
    out = capsys.readouterr().out
    assert "day" in out and "act" in out
    assert "planning" in out, "the pre-trek row must be shown -- Act 1 depends on it"
    assert "music map" in out


@needs_tools
def test_stale_music_rows_do_not_survive_a_rerun(spined):
    """The bug this guards: switching from a playlist to audio files left 25
    stale playlist rows in the table -- no tempo, no key, a default energy of
    0.50 -- and they competed in the act assignment, so acts drew from tracks
    that were no longer in the library."""
    from nepal import db
    _, report, cfg = spined
    if report["music"].get("skipped"):
        pytest.skip("librosa not installed")

    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO music_tracks(track_id, title, energy_mean, tempo_bpm) "
                 "VALUES ('ghost-from-an-earlier-run', 'Ghost', 0.5, 0.0)")
    conn.commit()
    conn.close()

    from nepal.stages import s02_spine
    again = s02_spine.analyse_music(cfg, db.init(cfg.db_path))
    assert again.get("n_stale_removed", 0) >= 1

    conn = db.init(cfg.db_path)
    rows = [r["track_id"] for r in conn.execute("SELECT track_id FROM music_tracks")]
    orphan_sections = conn.execute(
        "SELECT COUNT(*) n FROM music_sections WHERE track_id NOT IN "
        "(SELECT track_id FROM music_tracks)").fetchone()["n"]
    orphan_beats = conn.execute(
        "SELECT COUNT(*) n FROM beats WHERE track_id NOT IN "
        "(SELECT track_id FROM music_tracks)").fetchone()["n"]
    conn.close()
    assert "ghost-from-an-earlier-run" not in rows
    assert orphan_sections == 0, "sections left behind by a removed track"
    assert orphan_beats == 0, "beats left behind by a removed track"


# -- S02.6 file selection, against real containers --------------------
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_s02_6_skips_what_cannot_be_transcribed(tmp_path):
    """A Telegram export writes a thumbnail beside every round video, and a
    round video can be silent. faster-whisper hands both to PyAV, which raises
    IndexError from inside a generator -- so the stage died after large-v3 had
    already loaded, on an error naming neither the file nor the reason.

    Probed against real containers: the whole defect was the gap between what
    the extension claims and what the file holds."""
    from nepal.stages.s02_spine import usable_audio_files
    from nepal.util import proc

    def ff(*args):
        proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                 check=True)

    speech = tmp_path / "with_audio.mp4"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
       "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
       str(speech))
    silent = tmp_path / "silent.mp4"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent))
    thumb = tmp_path / "with_audio.jpg"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=1:duration=1",
       "-frames:v", "1", str(thumb))

    usable, skipped = usable_audio_files(sorted(tmp_path.iterdir()))
    assert usable == [speech]
    assert skipped["no audio stream"] == 1
    assert any("jpg" in reason for reason in skipped)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_s02_6_counts_an_unreadable_file_rather_than_raising(tmp_path):
    from nepal.stages.s02_spine import usable_audio_files

    broken = tmp_path / "truncated.mp4"
    broken.write_bytes(b"not actually an mp4")
    usable, skipped = usable_audio_files([broken])
    assert usable == []
    assert sum(skipped.values()) == 1
