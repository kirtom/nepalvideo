"""The timeline v2 end to end on a seeded database: no media, the real config.

Three hundred-odd shots from three sources over five acts, two phones filming
the same minute in portrait, a camera recording that mentions the bridge,
three story beats, two tracks with sections and a beat grid, act boundaries
and a short GPS track -- everything ``build_timeline`` reads, seeded the way
the real stages write it, so the wiring is exercised at the boundary it will
actually cross on the box.
"""
import json
import logging
import math
import subprocess
import sys, pathlib

import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

from nepal import db
from nepal.config import Config
from nepal.process import rhythm
from nepal.stages import s05_cut

ROOT = pathlib.Path(__file__).resolve().parents[1]
PHONES = ("phone_keller", "phone_kulikov")


def _cfg(tmp_path):
    real = Config.load(ROOT / "config" / "pipeline.yaml")
    data = dict(real._data)
    data["project"] = {"data_root": str(tmp_path / "data"), "work_root": str(tmp_path / "work"),
                       "db_path": str(tmp_path / "work" / "db" / "n.sqlite")}
    return Config(data, real.path)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


# One day per act in the trek, a long planning phase before and a return after.
ACT_UTC = {
    1: (datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 4, 25, tzinfo=timezone.utc)),
    2: (datetime(2024, 4, 26, tzinfo=timezone.utc), datetime(2024, 4, 30, tzinfo=timezone.utc)),
    3: (datetime(2024, 4, 30, tzinfo=timezone.utc), datetime(2024, 5, 4, tzinfo=timezone.utc)),
    4: (datetime(2024, 5, 4, tzinfo=timezone.utc), datetime(2024, 5, 5, tzinfo=timezone.utc)),
    5: (datetime(2024, 5, 5, tzinfo=timezone.utc), datetime(2024, 5, 20, tzinfo=timezone.utc)),
}


def _seed(cfg, *, lopsided_act=None):
    """``lopsided_act``: that act's bulk comes three parts keller to one
    kulikov, keller a fifth of a point stronger -- the corpus's act 3 in
    kind, not the round-robin symmetry of the other acts, which hides a
    share rule that only ever ran per gap.

    Three to one and not the corpus's five because the share is bounded by
    the material, not by the repair: an act that puts most of its clips on
    screen cannot give a quarter of its slots to a phone holding a sixth of
    its rows, however hard the repair swaps. Twice the recordings, for the
    same reason -- the act needs slack to be selective in, or the fill takes
    everything either phone has and there is nothing left to swap.
    """
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.init(cfg.db_path)
    recordings: list[tuple] = []
    assets: list[tuple] = []
    shots: list[dict] = []

    def recording(rid, source, start, duration_s, *, portrait=False, is_360=None):
        # The real corpus never stores portrait dimensions: 253 of 373 phone
        # videos are 1920x1080 in the asset row with a 90/270 rotation tag in
        # probe_json, and only 3 are portrait by width/height alone. Seed the
        # same shape here so the split-slot assertion below is proof that
        # pairs.display_dimensions reads the rotation, not an artifact of the
        # seed having done the swap itself.
        w, h = 1920, 1080
        probe_json = json.dumps({"streams": [{"codec_type": "video",
                                              "tags": {"rotate": "90"}}]}) if portrait else None
        recordings.append((rid, source, int(is_360 if is_360 is not None else source == "camera"),
                           _iso(start), duration_s, 1))
        assets.append((f"a_{rid}", f"raw/{rid}.mp4", source, "video360" if source == "camera" else "video_flat",
                       w, h, duration_s, _iso(start), rid, 0, probe_json))

    def shot(rid, idx, start_s, end_s, start, act, *, source_score=0.6, face=0, face_score=None,
             transcript=None, place=None, levity=0, status="shortlisted", lat=None, lon=None, alt=None):
        shots.append({"shot_id": f"{rid}#{idx:04d}", "recording_id": rid, "asset_id": None,
                      "media_kind": "video", "start_s": start_s, "end_s": end_s,
                      "start_utc": _iso(start + timedelta(seconds=start_s)), "act": act,
                      "place_name": place, "score_total": source_score, "has_face": face,
                      "face_score": face_score, "transcript": transcript, "tag_levity": levity,
                      "status": status, "motion_mag": 0.2, "lat": lat, "lon": lon, "alt_dem_m": alt,
                      "has_speech": int(bool(transcript))})

    def photo(pid, source, when, act, score=0.7):
        assets.append((pid, f"raw/{pid}.jpg", source, "photo", 3000, 4000, None, _iso(when), None, None, None))
        shots.append({"shot_id": f"photo_{pid}", "recording_id": None, "asset_id": pid,
                      "media_kind": "photo", "start_s": 0.0, "end_s": 4.0, "start_utc": _iso(when),
                      "act": act, "place_name": None, "score_total": score, "has_face": 0,
                      "face_score": None, "transcript": None, "tag_levity": 0, "status": "shortlisted",
                      "motion_mag": None, "lat": None, "lon": None, "alt_dem_m": None, "has_speech": 0})

    # -- act 1: planning, phones in the city ------------------------------
    t = datetime(2024, 3, 10, 9, tzinfo=timezone.utc)
    recording("k1", "phone_keller", t, 40)
    shot("k1", 0, 0, 15, t, 1, place="Kathmandu", source_score=0.5)
    shot("k1", 1, 15, 30, t, 1, place="Kathmandu", source_score=0.45)
    recording("q1", "phone_kulikov", t + timedelta(days=2), 40)
    shot("q1", 0, 0, 16, t + timedelta(days=2), 1, source_score=0.55)
    shot("q1", 1, 16, 32, t + timedelta(days=2), 1, source_score=0.4)
    photo("p1", "phone_keller", t + timedelta(days=5), 1)

    # -- act 2: approach; the bridge recording lives here -----------------
    t = datetime(2024, 4, 27, 4, tzinfo=timezone.utc)
    # the only material before the arrival is a still, and act 1 ends on one
    photo("p2z", "phone_kulikov", t - timedelta(hours=16), 2, score=0.8)
    recording("c2", "camera", t, 60)
    shot("c2", 0, 0, 20, t, 2, face=1, face_score=0.8, transcript="Мы приехали, тут жарко",
         source_score=0.8, place="Besisahar")
    shot("c2", 1, 20, 40, t, 2, source_score=0.6, place="Besisahar")
    shot("c2", 2, 40, 60, t, 2, source_score=0.5)
    recording("rb", "camera", t + timedelta(hours=2), 150)      # the bridge, 150 s of it
    for i in range(7):
        shot("rb", i, i * 20, min(150, (i + 1) * 20), t + timedelta(hours=2), 2, source_score=0.5,
             transcript="идём по мосту" if i == 0 else None, place="Bhulbhule")
    recording("c2b", "camera", t + timedelta(minutes=20), 40)      # B-roll for the arrival
    shot("c2b", 0, 0, 20, t + timedelta(minutes=20), 2, source_score=0.55, place="Besisahar")
    shot("c2b", 1, 20, 40, t + timedelta(minutes=20), 2, source_score=0.45)
    recording("k2", "phone_keller", t + timedelta(hours=4), 40)
    shot("k2", 0, 0, 18, t + timedelta(hours=4), 2, source_score=0.65, levity=1)
    shot("k2", 1, 18, 36, t + timedelta(hours=4), 2, source_score=0.5)
    recording("q2", "phone_kulikov", t + timedelta(hours=5), 40)
    shot("q2", 0, 0, 18, t + timedelta(hours=5), 2, source_score=0.6)
    shot("q2", 1, 18, 36, t + timedelta(hours=5), 2, source_score=0.55)
    # two stills a minute apart: never both on screen back to back
    photo("p2a", "phone_keller", t + timedelta(hours=6), 2, score=0.9)
    photo("p2b", "phone_keller", t + timedelta(hours=6, minutes=1), 2, score=0.85)

    # -- act 3: climb; the walking beat and the portrait pair --------------
    t = datetime(2024, 5, 1, 3, tzinfo=timezone.utc)
    recording("w3", "camera", t, 40)
    shot("w3", 0, 0, 20, t, 3, face=0, face_score=0.1, transcript="Я вот на этом курумнике прям сдох",
         source_score=0.9, place="Chame", lat=28.55, lon=84.24, alt=2700)
    shot("w3", 1, 20, 40, t, 3, source_score=0.6, place="Chame")
    recording("k3", "phone_keller", t + timedelta(minutes=30), 60, portrait=True)
    shot("k3", 0, 0, 60, t + timedelta(minutes=30), 3, source_score=0.7, face=1)
    recording("q3", "phone_kulikov", t + timedelta(minutes=30, seconds=20), 60, portrait=True)
    shot("q3", 0, 0, 60, t + timedelta(minutes=30, seconds=20), 3, source_score=0.75, face=1)
    recording("k3b", "phone_keller", t + timedelta(hours=1), 40)
    shot("k3b", 0, 0, 18, t + timedelta(hours=1), 3, source_score=0.6, levity=1)
    shot("k3b", 1, 18, 36, t + timedelta(hours=1), 3, source_score=0.5)
    recording("q3b", "phone_kulikov", t + timedelta(hours=2), 40)
    shot("q3b", 0, 0, 18, t + timedelta(hours=2), 3, source_score=0.62)
    shot("q3b", 1, 18, 36, t + timedelta(hours=2), 3, source_score=0.48)
    recording("c3", "camera", t + timedelta(hours=3), 40)
    shot("c3", 0, 0, 20, t + timedelta(hours=3), 3, source_score=0.7, place="Pisang")
    shot("c3", 1, 20, 40, t + timedelta(hours=3), 3, source_score=0.3, status="candidate")
    shot("c3", 2, 40, 40.5, t + timedelta(hours=3), 3, source_score=0.99, status="rejected")
    photo("p3", "phone_kulikov", t + timedelta(hours=4), 3)

    # -- act 4: the pass; the cold open comes from here --------------------
    t = datetime(2024, 5, 4, 1, tzinfo=timezone.utc)
    recording("c4", "camera", t, 40)
    shot("c4", 0, 0, 20, t, 4, face=1, face_score=0.7, transcript="Перевал. Пять тысяч сто.",
         source_score=0.95, alt=5100, lat=28.79, lon=83.93)
    shot("c4", 1, 20, 40, t, 4, source_score=0.7, alt=5100)
    recording("k4", "phone_keller", t + timedelta(minutes=10), 40)
    shot("k4", 0, 0, 18, t + timedelta(minutes=10), 4, source_score=0.6, levity=1)
    shot("k4", 1, 18, 36, t + timedelta(minutes=10), 4, source_score=0.5)
    recording("q4", "phone_kulikov", t + timedelta(minutes=20), 40)
    shot("q4", 0, 0, 18, t + timedelta(minutes=20), 4, source_score=0.65)
    shot("q4", 1, 18, 36, t + timedelta(minutes=20), 4, source_score=0.55)

    # -- act 5: the way down and home --------------------------------------
    t = datetime(2024, 5, 8, 6, tzinfo=timezone.utc)
    recording("c5", "camera", t, 40)
    shot("c5", 0, 0, 20, t, 5, source_score=0.6, place="Jomsom")
    shot("c5", 1, 20, 40, t, 5, source_score=0.5, place="Jomsom")
    recording("k5", "phone_keller", t + timedelta(days=1), 40)
    shot("k5", 0, 0, 18, t + timedelta(days=1), 5, source_score=0.6, place="Pokhara")
    shot("k5", 1, 18, 36, t + timedelta(days=1), 5, source_score=0.5, place="Pokhara")
    recording("q5", "phone_kulikov", t + timedelta(days=2), 40)
    shot("q5", 0, 0, 18, t + timedelta(days=2), 5, source_score=0.55, levity=1, place="Kathmandu")
    shot("q5", 1, 18, 36, t + timedelta(days=2), 5, source_score=0.45, place="Kathmandu")
    photo("p5", "phone_kulikov", t + timedelta(days=3), 5)

    # -- the bulk of each act, clumped on a couple of hours of one day ------
    # Each act's boundaries span days; its footage sits in a few hours of one
    # of them (act 5's a week before its end), as the corpus does, so most of
    # a gap's UTC windows hold nothing and the fill has to reach for the
    # nearest by time. Enough rows to run every act to its planned length --
    # act 1's cue opens on its swell, a ten-cut burst then two-second cuts,
    # so a minute of it takes twice the rows a minute elsewhere does -- and
    # short enough that their sum stays under what would grow the film past
    # its floor: the material must not lengthen the film it is meant to fill.
    for act, day, n_recs, shot_s in ((1, datetime(2024, 3, 12, 10, tzinfo=timezone.utc), 12, 4.0),
                                     (2, datetime(2024, 4, 27, 5, tzinfo=timezone.utc), 14, 6.5),
                                     (3, datetime(2024, 5, 1, 5, tzinfo=timezone.utc), 34, 6.5),
                                     (4, datetime(2024, 5, 4, 1, 30, tzinfo=timezone.utc), 10, 6.5),
                                     (5, datetime(2024, 5, 8, 8, tzinfo=timezone.utc), 26, 6.5)):
        lopsided = act == lopsided_act
        for i in range(60 if lopsided else n_recs):
            rid, start = f"f{act}_{i:02d}", day + timedelta(minutes=4 * i)
            src = (("phone_kulikov" if i % 4 == 0 else "phone_keller") if lopsided
                   else ("phone_keller", "phone_kulikov", "camera")[i % 3])
            recording(rid, src, start, 20)
            for j in range(3):
                shot(rid, j, shot_s * j, shot_s * (j + 1), start, act,
                     source_score=0.4 + 0.05 * ((i + j) % 5) + (0.2 if lopsided and src == "phone_keller" else 0.0))

    conn.executemany("INSERT INTO recordings(recording_id, source, is_360, start_utc, duration_s, "
                     "asset_count) VALUES (?,?,?,?,?,?)", recordings)
    conn.executemany("INSERT INTO assets(asset_id, s3_key, source, kind, width, height, duration_s, "
                     "created_at_utc, recording_id, chapter_index, probe_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     assets)
    db.upsert(conn, "shots", ["shot_id"], shots)

    conn.execute("INSERT INTO messages(msg_id, ts_utc, author, text, phase) VALUES "
                 "('m1', '2024-02-11T19:02:00+00:00', 'Kulikov', '20 км в день не проблема', 'planning')")
    beats = [
        ("b_arrive", "speech", 2, "c2#0000", None, 3.0, 9.0, "Мы приехали, тут жарко", 1, "none", 3),
        ("b_walk", "speech", 3, "w3#0000", None, 2.0, 12.0, "Я вот на этом курумнике прям сдох", 0, "none", 2),
        ("b_pass", "speech", 4, "c4#0000", None, 1.0, 8.0, "Перевал. Пять тысяч сто.", 0, "freeze", 1),
        ("q_plan", "quote", 1, None, "m1", None, None, "20 км в день не проблема", 1, "none", 4),
    ]
    conn.executemany("INSERT INTO story_beats(beat_id, kind, act, shot_id, msg_id, src_in, src_out, "
                     "text, levity, effect, rank) VALUES (?,?,?,?,?,?,?,?,?,?,?)", beats)

    for tid, length, energies in (("t1", 180.0, (0.3, 0.35, 0.5, 0.55, 0.8, 0.7, 0.4, 0.3)),
                                  ("t2", 200.0, (0.2, 0.3, 0.6, 0.65, 0.9, 0.75, 0.5, 0.35))):
        conn.execute("INSERT INTO music_tracks(track_id, s3_key, title, duration_s, tempo_bpm, "
                     "energy_mean, energy_p95, energy_p10, centroid, onset_rate, assigned_act) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (tid, f"music/{tid}.mp3", tid.upper(), length, 120.0, sum(energies) / len(energies),
                      max(energies), min(energies), 2000.0, 2.0, None))
        n = len(energies)
        for i, e in enumerate(energies):
            conn.execute("INSERT INTO music_sections(section_id, track_id, start_s, end_s, energy, "
                         "is_swell) VALUES (?,?,?,?,?,?)",
                         (f"{tid}_s{i}", tid, i * length / n, (i + 1) * length / n, e, int(e == max(energies))))
        conn.executemany("INSERT INTO beats(track_id, t_s, is_downbeat) VALUES (?,?,?)",
                         [(tid, b / 2.0, int(b % 4 == 0)) for b in range(int(length * 2))])

    db.set_decision(conn, "act_boundaries", json.dumps(
        [{"act": a, "start_utc": _iso(lo), "end_utc": _iso(hi), "method": "test"}
         for a, (lo, hi) in ACT_UTC.items()]))

    # A climb on the morning of the walking beat and a hard hour at the pass:
    # fixes five minutes apart, moving at a walk, gaining height, pulse up.
    pts = []
    for day, hr, alt0 in ((datetime(2024, 5, 1, 2, 30, tzinfo=timezone.utc), 120, 2600),
                          (datetime(2024, 5, 4, 0, 30, tzinfo=timezone.utc), 155, 4900)):
        for i in range(24):
            ts = day + timedelta(minutes=5 * i)
            pts.append((_iso(ts), 28.5 + 0.001 * i, 84.2, alt0 + 25 * i, "strava", hr + i % 3, None, "act"))
    conn.executemany("INSERT INTO gps_points(ts_utc, lat, lon, alt_dem_m, source, hr_bpm, alt_baro_m, "
                     "activity_id) VALUES (?,?,?,?,?,?,?,?)", pts)
    conn.commit()
    return conn


def _shots(conn):
    return {r["shot_id"]: dict(r) for r in conn.execute("SELECT * FROM shots")}


def test_build_timeline_v2_assembles_the_film_from_the_seeded_database(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    rep = s05_cut.build_timeline(cfg, conn)

    slots = [dict(r) for r in conn.execute("SELECT * FROM timeline ORDER BY slot_index")]
    shots = _shots(conn)
    recs = {r["recording_id"]: dict(r) for r in conn.execute("SELECT * FROM recordings")}
    assert rep["n_slots"] == len(slots) > 10
    # Gate 3's numbers as planned; the cues step reports what is played
    assert 0.0 <= rep["music_share_planned"] < 1.0, "music no longer runs from end to end"
    assert rep["n_music_windows_planned"] >= 1
    assert set(rep["music_windows_per_act_planned"]) == {str(a["act"]) for a in
                                                         json.loads((cfg.work_root / "music" /
                                                                     "music_map.json").read_text())["acts"]}
    assert rep["music_windows_per_act_planned"]["0"] == [], "the film does not open on music"
    # every locked slot the run lost -- pushed past its act's end or squeezed
    # to nothing before the write -- is counted, and a healthy seed loses none
    assert rep["n_dropped_locked"] == 0

    # the cold open, then the card, before act 1 begins
    assert slots[0]["kind"] == "video" and slots[0]["act"] == 0 and slots[0]["t_in"] == 0.0
    assert slots[0]["beat_id"] == "b_pass" and rep["cold_open_beat"] == "b_pass"
    assert 15.0 <= slots[0]["t_out"] - slots[0]["t_in"] <= 25.0
    assert slots[1]["kind"] == "card" and slots[1]["act"] == 0
    assert slots[1]["t_in"] == slots[0]["t_out"] and slots[1]["t_out"] - slots[1]["t_in"] == 3.0
    assert all(s["act"] >= 1 for s in slots[2:])
    # act 1's music is one section: nothing there is "the swell", so no burst
    act1 = [s for s in slots if s["act"] == 1]
    assert act1 and not all(math.isclose(s["t_out"] - s["t_in"], 0.5, abs_tol=1e-6) for s in act1)

    # every speech beat has its slots; the walking beat stays on its own recording
    for beat in ("b_arrive", "b_walk", "b_pass"):
        assert any(s["beat_id"] == beat and s["act"] >= 1 for s in slots), beat
    walk = [s for s in slots if s["beat_id"] == "b_walk"]
    assert all(shots[s["shot_id"]]["recording_id"] == "w3" for s in walk)
    assert any(shots[s["shot_id"]]["recording_id"] != "c2" for s in slots if s["beat_id"] == "b_arrive"), \
        "the speaking beat cuts to B-roll after the face hold"

    # the two phones at the same minute become one split slot
    splits = [s for s in slots if s["secondary_shot_id"]]
    assert splits and rep["n_pairs"] >= 1
    for s in splits:
        assert json.loads(s["motion"])["type"] == "split"
        assert shots[s["shot_id"]]["recording_id"] != shots[s["secondary_shot_id"]]["recording_id"]
        a, b = shots[s["shot_id"]], shots[s["secondary_shot_id"]]
        assert {r["source"] for r in (recs[a["recording_id"]], recs[b["recording_id"]])} == set(PHONES)

    # the bridge is one unbroken ninety-second take. `locked` is not a column
    # of the table, so "locked" is asserted through its consequence: the take
    # is still exactly ninety seconds after the rhythm pass re-timed the act.
    assert rep["long_take"] == "rb"
    take = [s for s in slots if s["shot_id"] and shots[s["shot_id"]]["recording_id"] == "rb"]
    assert len(take) == 1 and math.isclose(take[0]["t_out"] - take[0]["t_in"], 90.0, abs_tol=1e-6)
    assert take[0]["src_in"] == 0.0 and take[0]["src_out"] == 90.0

    # each phone keeps a quarter of an act's phone slots wherever it had material
    for act, sources in rep["per_act_sources"].items():
        if int(act) < 1:
            continue
        phone_slots = sum(sources.get(p, {}).get("slots", 0) for p in PHONES)
        if any(sources.get(p, {}).get("available", 0) for p in PHONES):
            assert phone_slots > 0, (act, sources)
        for p in PHONES:
            if sources.get(p, {}).get("available", 0):
                assert sources[p]["slots"] / phone_slots >= 0.25, (act, sources)

    # scenes and music: the map is on disk, and not every segment starts a track
    mmap = json.loads((cfg.work_root / "music" / "music_map.json").read_text())
    segs = [seg for a in mmap["acts"] for seg in a["segments"]]
    assert segs and any(seg["src_in"] != 0 for seg in segs)
    assert rep["n_scenes"] >= 5 and rep["music_assignment"] == "scene"
    assert all(s["scene_id"] is not None for s in slots)
    assert {a["act"] for a in mmap["acts"]} == {0, 1, 2, 3, 4, 5}

    # never two stills back to back
    for prev, cur in zip(slots, slots[1:]):
        assert not (prev["kind"] == "photo" and cur["kind"] == "photo"), (prev, cur)
    assert any(s["kind"] == "photo" for s in slots)

    # the timeline is monotone and never claims footage a shot does not have
    for prev, cur in zip(slots, slots[1:]):
        assert cur["t_in"] >= prev["t_in"] and cur["t_in"] >= prev["t_out"] - 1e-6
    for s in slots:
        assert s["t_out"] > s["t_in"], s
        if s["kind"] != "video":
            continue
        shot = shots[s["shot_id"]]
        length = s["t_out"] - s["t_in"]
        if shot["recording_id"] == "rb":      # the take runs past its first shot by design
            assert length <= recs["rb"]["duration_s"] - s["src_in"] + 1e-6
        else:
            assert length <= shot["end_s"] - s["src_in"] + 1e-6, s
        assert math.isclose(s["src_out"] - s["src_in"], length, abs_tol=1e-3), s
        if s["secondary_shot_id"]:
            other = shots[s["secondary_shot_id"]]
            assert length <= other["end_s"] - s["secondary_src_in"] + 1e-6, s
    assert "c3#0002" not in {s["shot_id"] for s in slots}          # rejected stays out
    assert rep["duration_s"] == slots[-1]["t_out"]

    # the natural-sound windows land on slots, and the files are written
    assert isinstance(rep["natural_windows"], list)
    for w in rep["natural_windows"]:
        assert 0 <= w["slot_index"] < len(slots) and w["t_out"] > w["t_in"]
    assert (cfg.work_root / "timeline.otio").exists() and (cfg.work_root / "timeline.fcpxml").exists()
    assert set(rep["per_act"]) == {"0", "1", "2", "3", "4", "5"}
    conn.close()


def test_acts_reach_their_planned_length(tmp_path, caplog):
    """Film time is laid over an act's UTC span proportionally and the seed's
    footage, like the corpus, clumps on a few hours of a days-long span; a
    gap whose window held nothing added nothing, and every act ended where
    its material stopped. A gap now takes the nearest footage by time, so
    the acts run to the length the film allotted them -- and nothing of zero
    length is dropped on the way to the table, which the write-path guard
    would otherwise hide."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    with caplog.at_level(logging.WARNING, logger="nepal.stages.s05_cut"):
        rep = s05_cut.build_timeline(cfg, conn)
    assert not [r.getMessage() for r in caplog.records if "zero-length" in r.getMessage()]
    tol = float(cfg.get("film.duration_tolerance_s"))
    for act, (start, end) in rep["act_spans"].items():
        planned = rep["act_planned_s"][act]
        assert abs((end - start) - planned) <= tol, (act, end - start, planned)
    conn.close()


def test_material_bounds_an_act_band_but_never_below_its_floor(tmp_path):
    """An act's band is capped by what its rows can put on screen: every clip
    once at what a slot actually runs (its own length if shorter), stills
    only as the share the photo budget admits on top -- and an act with less
    than its floor keeps the floor and comes out short."""
    cfg = _cfg(tmp_path)
    # Act 3, not act 2: acts 1 and 2 carry a photo_share_by_act override
    # (2026-09-24, they run out of clips) and the arithmetic below is the
    # film-wide share.
    assert (cfg.get("assemble.expected_slot_s"), cfg.get("film.photo_share")) == (2.5, 0.1) \
        and 3 not in (cfg.get("film.photo_share_by_act") or {}), \
        "the arithmetic below assumes these"
    rows = ([{"act": 3, "media_kind": "video", "start_s": 0.0, "end_s": 15.0}] * 36      # 36 x 2.5 = 90 s
            + [{"act": 3, "media_kind": "photo", "start_s": 0.0, "end_s": 4.0}] * 20     # only as the 10 % share
            + [{"act": 4, "media_kind": "video", "start_s": 0.0, "end_s": 1.0}] * 10)    # 10 x 1.0 = 10 s
    specs = [{"act": 3, "min_s": 50, "max_s": 700}, {"act": 4, "min_s": 90, "max_s": 150}]
    bounded, material = s05_cut._material_bound(cfg, specs, rows)
    assert material[3] == pytest.approx(100.0) and material[4] == pytest.approx(10.0 / 0.9)
    assert [(b["act"], b["min_s"]) for b in bounded] == [(3, 50), (4, 90)]
    assert bounded[0]["max_s"] == pytest.approx(100.0) and bounded[1]["max_s"] == 90.0


def test_a_refill_is_laid_at_what_a_slot_runs_and_stays_inside_its_gap(tmp_path):
    """Budgeted at 2.5 s a slot and laid at the act's 5-8 s plan lengths, a
    mid-act refill overran its gap two to three times; what started inside
    the locked anchor was dropped and what started after it played after
    the line it was chosen to precede. With a budget given, the refill is
    laid at expected_slot_s and ends within one slot of its gap."""
    cfg = _cfg(tmp_path)
    per = float(cfg.get("assemble.expected_slot_s"))
    day = datetime(2024, 5, 8, 8, tzinfo=timezone.utc)
    free = [{"shot_id": f"s{i}", "recording_id": f"r{i}", "act": 5, "media_kind": "video", "source": "camera",
             "start_s": 0.0, "end_s": 12.0, "score_total": 0.9, "place_name": None,
             "start_utc": _iso(day + timedelta(minutes=i))} for i in range(12)]
    t0, t1 = 100.0, 125.0
    laid = s05_cut._fill_gap(cfg, free, act=5, t0=t0, t1=t1, lo=None, hi=None, similarity=lambda a, b: 0.0,
                             existing=[], prev_photo=False, next_photo=False,
                             budget=s05_cut._refill_budget(t1 - t0, per))
    assert len(laid) == 10 and laid[0]["t_in"] == t0
    assert all(s["t_out"] - s["t_in"] == pytest.approx(per) for s in laid)
    assert laid[-1]["t_out"] <= t1 + per
    # the plan's own gaps still lay at plan length
    planned = s05_cut._fill_gap(cfg, free, act=5, t0=t0, t1=t1, lo=None, hi=None, similarity=lambda a, b: 0.0,
                                existing=[], prev_photo=False, next_photo=False)
    assert all(s["t_out"] - s["t_in"] >= cfg.get("assemble.shot_duration_s")[5][0] for s in planned)


def test_a_grid_rounded_to_the_timeline_lets_the_walk_pass_its_last_beat(tmp_path):
    """The map's beats are three-decimal and build_timeline rounds them
    again after adding the act's shift: added in float, the grid's last beat
    sat at the walk's own t_in plus 1e-11, retime took it as after t_in, and
    every slot from there rounded onto it at zero length. Rounded, a beat at
    t_in is equal to it and the unsnapped band length stands."""
    cfg = _cfg(tmp_path)
    t_in = 743.886
    last_beat = t_in + 1e-11                       # the grid ends here; the act does not
    assert last_beat > t_in and round(last_beat, 3) == t_in
    shot = {"shot_id": "x", "media_kind": "video", "start_s": 0.0, "end_s": 6.5}
    slot = s05_cut._set_length(s05_cut._new_slot(kind="video", act=5, shot_id="x", src_in=0.0), t_in, t_in + 5.0)
    grid = [round(last_beat, 3)]
    out = rhythm.retime([slot], sections=[{"t_in": 700.0, "t_out": 800.0, "energy": 0.5}],
                        beats=grid, downbeats=grid, table=cfg.get("assemble.rhythm"), burst_slots=(0, 0),
                        is_act4=False, held_shot_s=cfg.get("assemble.act4_held_shot_s"), silence_t=None,
                        shots={"x": shot})
    mid = sum(cfg.get("assemble.rhythm")["mid"]) / 2          # one section ranks mid; no beat to snap to
    assert len(out) == 1 and out[0]["t_out"] - out[0]["t_in"] == pytest.approx(mid)


def test_refill_budget_is_the_gap_at_what_a_slot_actually_runs():
    """A 25 s gap left by the rhythm pass takes ten shots at 2.5 s each, not
    the four or five the act's plan lengths would say; never fewer than one."""
    assert s05_cut._refill_budget(25.0, 2.5) == 10
    assert s05_cut._refill_budget(26.0, 2.5) == 11
    assert s05_cut._refill_budget(0.5, 2.5) == 1


def test_a_starved_phone_gets_its_share_of_the_act_not_of_each_gap(tmp_path):
    """source_share_repair runs per gap over that gap's picks and never over
    the act, so with one phone holding three times the rows and a score edge
    the other ended far under its share of the act. After the refill rounds
    the act is re-balanced by swapping shots, slot for slot, until the
    starved phone holds its share of the act's phone video slots.

    With the repair switched off this act gives kulikov 20% of its phone
    slots, so the assertion below is the repair's work and not the fill's."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg, lopsided_act=5)
    s05_cut.build_timeline(cfg, conn)
    by_source = {r["source"]: r["n"] for r in conn.execute(
        "SELECT rc.source AS source, COUNT(*) AS n FROM timeline t JOIN shots s ON s.shot_id = t.shot_id "
        "JOIN recordings rc ON rc.recording_id = s.recording_id WHERE t.act = 5 AND t.kind = 'video' GROUP BY 1")}
    phone = sum(by_source.get(p, 0) for p in PHONES)
    assert phone and by_source.get("phone_kulikov", 0) / phone >= float(cfg.get("assemble.source_share_min")), by_source
    conn.close()


def test_the_music_maps_problems_are_read_before_the_acts_are_cut_against_it(tmp_path, caplog):
    """Every act is timed on the map's grid, so a problem logged at the end
    of the run is a problem read after the map has already shaped the film.
    The seeded library is six minutes of music for a fifteen-minute film, so
    there is always something for it to say."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    with caplog.at_level(logging.INFO, logger="nepal.stages.s05_cut"):
        s05_cut.build_timeline(cfg, conn)
    lines = [r.getMessage() for r in caplog.records]
    problem = next(i for i, m in enumerate(lines) if m.startswith("S06 music map:"))
    first_act = next(i for i, m in enumerate(lines) if m.startswith("S06 act 1 rhythm"))
    assert problem < first_act, lines[:8]
    conn.close()


def test_a_music_track_with_no_measured_duration_is_named_by_the_cue_step(tmp_path, caplog):
    """The stage selected `WHERE duration_s IS NOT NULL`, so the one track
    whose cues could not be bounded was also the one track nothing said a
    word about -- it got exactly the unbounded stretch the bound exists to
    stop. The row comes through with its null now, and the step names it."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    s05_cut.build_timeline(cfg, conn)
    conn.execute("UPDATE music_tracks SET duration_s = NULL WHERE track_id = 't1'")
    with caplog.at_level(logging.WARNING, logger="nepal.stages.s05_cut"):
        s05_cut.build_cues(cfg, conn)
    assert "music track t1 has no measured duration" in caplog.text
    assert "music track t2" not in caplog.text, "only the track that is actually missing one"
    conn.close()


def _phone_slot(i, shot_id, **fields):
    s = s05_cut._new_slot(kind="video", act=5, shot_id=shot_id, src_in=0.0, **fields)
    return s05_cut._set_length(s, 10.0 * i, 10.0 * i + 4.0)


def _phone_row(shot_id, source, recording_id):
    return {"shot_id": shot_id, "source": source, "recording_id": recording_id,
            "media_kind": "video", "start_s": 0.0, "end_s": 60.0, "start_utc": None,
            "score_total": 1.0}


def test_the_share_repair_will_not_swap_in_a_shot_that_breaks_the_run_rule():
    """The swap replaces a slot in place, between two slots already on
    screen, so a candidate can join a run on both sides at once --
    max_consecutive_recording has to be checked there or the repair undoes
    what the fill had just satisfied. keller is starved (2 of 6) and its only
    unused clip belongs to the recording both of the giver's neighbours came
    from."""
    slots = [_phone_slot(0, "k1"), _phone_slot(1, "u1"), _phone_slot(2, "k2"),
             _phone_slot(3, "u2", beat_id="b1"), _phone_slot(4, "u3", beat_id="b2"),
             _phone_slot(5, "u4", beat_id="b3")]
    rows = [_phone_row("k1", "phone_keller", "rk"), _phone_row("k2", "phone_keller", "rk"),
            _phone_row("u1", "phone_kulikov", "rku"), _phone_row("u2", "phone_kulikov", "rku"),
            _phone_row("u3", "phone_kulikov", "rku"), _phone_row("u4", "phone_kulikov", "rku"),
            _phone_row("k3", "phone_keller", "rk")]
    shots = {r["shot_id"]: r for r in rows}

    swaps = s05_cut._rebalance_phones(list(slots), rows, shots, act=5, act0=[], excluded=set(),
                                      min_share=0.4, run_cap=2)
    assert swaps == [], "k3 would make three slots of rk in a row"

    # the same seed with the candidate on its own recording: the swap happens,
    # so it is the run rule refusing above and not the share or the length
    rows[-1] = _phone_row("k3", "phone_keller", "rk_other")
    shots["k3"] = rows[-1]
    assert s05_cut._rebalance_phones(list(slots), rows, shots, act=5, act0=[], excluded=set(),
                                     min_share=0.4, run_cap=2) == [("u1", "k3")]


def test_a_refused_candidate_is_counted_once_per_giver_not_once_per_round(caplog):
    """The repair re-reads the same pools every round, so a plain counter
    reported the same refusal again on every pass. Here keller needs two
    swaps and never gets the second: the first giver refuses both candidates
    in round one and the surviving one again in round two -- three counts by
    loop pass, two by (giver, candidate)."""
    slots = ([_phone_slot(0, "k1"), _phone_slot(1, "u1"), _phone_slot(2, "k2"),
              _phone_slot(3, "u2")]
             + [_phone_slot(i, f"u{i - 1}", beat_id=f"b{i}") for i in range(4, 12)])
    rows = ([_phone_row("k1", "phone_keller", "rk"), _phone_row("k2", "phone_keller", "rk"),
             _phone_row("kA", "phone_keller", "rk"), _phone_row("kB", "phone_keller", "rk")]
            + [_phone_row(f"u{i}", "phone_kulikov", "rku") for i in range(1, 11)])
    shots = {r["shot_id"]: r for r in rows}

    with caplog.at_level(logging.INFO, logger="nepal.stages.s05_cut"):
        swaps = s05_cut._rebalance_phones(slots, rows, shots, act=5, act0=[], excluded=set(),
                                          min_share=0.4, run_cap=2)
    assert swaps == [("u2", "kA")], "only the giver whose neighbours leave room takes one"
    assert "2 long-enough candidate(s) refused by the run rule" in caplog.text


def test_build_timeline_is_rebuilt_not_accumulated(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    first = s05_cut.build_timeline(cfg, conn)
    again = s05_cut.build_timeline(cfg, conn)
    n = conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0]
    assert n == first["n_slots"] == again["n_slots"]
    conn.close()


def test_render_draft_left_join_keeps_the_card_and_matches_row_count(tmp_path, monkeypatch):
    """render_draft used to INNER JOIN shots, which silently drops the card
    slot (shot_id NULL by construction) before the missing-media check ever
    runs -- one row short of the timeline, and nothing here would have caught
    a regression back to that join without checking the query's own output.

    Every other row gets a real stand-in file (render_draft never reads it;
    subprocess.run is captured here, not run) so the number of inputs
    build_command receives can be compared directly against the timeline's
    own row count -- the count an inner join would have been one short of.
    """
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    s05_cut.build_timeline(cfg, conn)
    n_rows = conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0]

    for r in conn.execute(
            "SELECT t.kind, t.shot_id, s.recording_id, s.media_kind FROM timeline t "
            "LEFT JOIN shots s ON s.shot_id = t.shot_id"):
        if r["kind"] == "card":
            continue
        if r["media_kind"] == "photo":
            cfg.work("stills", f"{r['shot_id']}.jpg").touch()
        else:
            cfg.work("proxies", f"{r['recording_id']}_eq.mp4").touch()

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    rep = s05_cut.render_draft(cfg, conn)
    assert rep["n_slots"] == n_rows, "an inner join would have made the draft one slot short"
    cmd = captured["cmd"]
    assert cmd.count("-i") == n_rows
    i = cmd.index("-f")
    assert cmd[i:i + 2] == ["-f", "lavfi"], "the card must reach build_command as a synthesised input"
    conn.close()


def test_gap_candidates_widen_an_empty_window_by_utc_distance():
    """A window that holds fewer rows than its budget takes the nearest rows
    by distance to it, the inside rows first, up to twice the budget."""
    day = datetime(2024, 5, 1, tzinfo=timezone.utc)
    rows = [{"shot_id": f"s{i}", "start_utc": _iso(day + timedelta(seconds=10 * i))} for i in range(10)]
    at = lambda s: day.timestamp() + s
    ids = lambda cands: [c["shot_id"] for c in cands]

    # four rows inside 60..120 s against a budget of three: nothing is widened
    cands, n_inside = s05_cut._gap_candidates(rows, at(60), at(120), budget=3)
    assert (ids(cands), n_inside) == (["s6", "s7", "s8", "s9"], 4)
    # a budget of six: the inside four first, then the nearest, to twice the budget
    cands, n_inside = s05_cut._gap_candidates(rows, at(60), at(120), budget=6)
    assert (ids(cands), n_inside) == (["s6", "s7", "s8", "s9", "s5", "s4", "s3", "s2", "s1", "s0"], 4)
    # a window over days that hold no footage at all: the nearest, twice the budget of them
    cands, n_inside = s05_cut._gap_candidates(rows, at(3 * 86400), at(5 * 86400), budget=2)
    assert (ids(cands), n_inside) == (["s9", "s8", "s7", "s6"], 0)
    # footage after the window is as near as its end, not its start
    cands, n_inside = s05_cut._gap_candidates(rows, at(-200), at(-100), budget=1)
    assert (ids(cands), n_inside) == (["s0", "s1"], 0)
    # an unstamped row is inside any window, and an unbounded side holds everything on it
    untimed = {"shot_id": "u", "start_utc": None}
    cands, n_inside = s05_cut._gap_candidates([untimed] + rows, None, at(45), budget=3)
    assert (ids(cands), n_inside) == (["u", "s0", "s1", "s2", "s3", "s4"], 6)


def test_fill_lifts_the_place_cap_only_for_what_the_cap_left_short():
    """A geocoded place is a whole day, so three per place is what a gap
    prefers, not a ceiling on the film: the cap holds for the first pass,
    and the rest of the budget comes from what that pass left without it."""
    def rows(places):
        return [{"shot_id": f"s{i}", "recording_id": f"r{i}", "place_name": p, "media_kind": "video",
                 "source": "camera", "score_total": 0.9 - 0.01 * i} for i, p in enumerate(places)]
    select = lambda cands: s05_cut._select(cands, budget=6, similarity=lambda a, b: 0.0, lam=0.3,
                                           existing=[], prev_photo=False, place_cap=3, run_cap=2, after=3)
    chosen, n_relaxed = select(rows(["Manang"] * 10))
    assert len(chosen) == 6 and n_relaxed == 3
    # the first pass took the three the cap allows, best first; the rest came without it
    assert [c["shot_id"] for c in chosen[:3]] == ["s0", "s1", "s2"]
    assert [c["shot_id"] for c in chosen[3:]] == ["s3", "s4", "s5"]
    # with places to spare the cap never bites and the second pass never runs
    chosen, n_relaxed = select(rows([f"place{i}" for i in range(10)]))
    assert len(chosen) == 6 and n_relaxed == 0


def test_two_locked_slots_never_cut_each_other():
    """A crowded act's anchors are clamped onto each other by place_anchors;
    the later beat then follows the earlier one, and neither loses a frame."""
    a = s05_cut._new_slot(kind="video", act=2, shot_id="x", src_in=0.0, locked=1)
    b = s05_cut._new_slot(kind="video", act=2, shot_id="y", src_in=5.0, locked=1)
    s05_cut._set_length(a, 10.0, 18.0)
    s05_cut._set_length(b, 14.0, 20.0)
    out = s05_cut._resolve_overlaps([a, b])
    assert [(s["shot_id"], s["t_in"], s["t_out"]) for s in out] == [("x", 10.0, 18.0), ("y", 18.0, 24.0)]
    assert out[1]["src_out"] - out[1]["src_in"] == 6.0
    # an unlocked slot that runs into a locked one is cut there
    c = s05_cut._new_slot(kind="video", act=2, shot_id="z", src_in=0.0)
    s05_cut._set_length(c, 4.0, 12.0)
    out = s05_cut._resolve_overlaps([c, a])
    assert (out[0]["t_out"], out[1]["t_in"]) == (10.0, 10.0) and out[1]["t_out"] == 18.0


def test_a_locked_slot_pushed_past_the_act_end_is_named_and_counted(caplog):
    """The same clamp as above, one act shorter: the later beat follows the
    earlier one past the act's own end, where the next retime round skips
    it. A story beat may not leave the film without a word about it."""
    a = s05_cut._new_slot(kind="video", act=2, shot_id="x", beat_id="b_first", src_in=0.0, locked=1)
    b = s05_cut._new_slot(kind="video", act=2, shot_id="y", beat_id="b_second", src_in=0.0, locked=1)
    s05_cut._set_length(a, 10.0, 26.0)
    s05_cut._set_length(b, 20.0, 27.0)
    resolved = s05_cut._resolve_overlaps([a, b])
    assert float(resolved[-1]["t_in"]) == 26.0, "the later beat follows the earlier one, whole"

    kept = rhythm.retime(resolved, sections=[{"t_in": 0.0, "t_out": 26.0, "energy": 0.5}],
                         beats=[], downbeats=[], table={"low": [5.0, 8.0], "mid": [3.0, 5.0],
                                                        "high": [1.5, 2.5]},
                         burst_slots=[0, 0], is_act4=False, held_shot_s=[6.0, 10.0],
                         silence_t=None, shots={"x": {"media_kind": "video", "start_s": 0.0, "end_s": 100.0},
                                                "y": {"media_kind": "video", "start_s": 0.0, "end_s": 100.0}})
    assert [s["shot_id"] for s in kept] == ["x"], "the act ends at 26s and y now starts there"

    with caplog.at_level(logging.WARNING):
        assert s05_cut._count_dropped_locked(resolved, kept, act=2) == 1
    assert "b_second" in caplog.text and "act 2" in caplog.text
    assert s05_cut._count_dropped_locked(resolved, resolved, act=2) == 0


def test_a_locked_slot_squeezed_to_nothing_is_counted_with_the_ones_pushed_off_the_end(caplog):
    """A locked slot is not only lost by being pushed past its act's end --
    the rounds before the write can also close it to zero length, and the
    pre-write filter then drops it just as silently. Same loss, same count:
    ``n_dropped_locked`` is the only line that says a beat the film was cut
    around is not in the film."""
    def slot(shot_id, t_in, t_out, locked=0):
        return s05_cut._set_length(
            s05_cut._new_slot(kind="video", act=1, shot_id=shot_id, beat_id="b1" if locked else None,
                              src_in=0.0, locked=locked), t_in, t_out)

    slots = [slot("a", 0.0, 4.0), slot("b", 4.0, 4.0, locked=1),
             slot("c", 4.0, 8.0), slot("d", 8.0, 8.0)]
    with caplog.at_level(logging.WARNING):
        kept, n_locked = s05_cut._drop_empty(slots)
    assert [s["shot_id"] for s in kept] == ["a", "c"]
    assert n_locked == 1, "the unlocked zero-length slot goes too, but costs the film nothing"
    assert "zero-length" in caplog.text

    assert s05_cut._drop_empty([slot("a", 0.0, 4.0)]) == ([slots[0]], 0)


def test_cues_sub_step_lays_the_tracks_over_the_seeded_timeline(tmp_path):
    """Part B reads only what the timeline step persisted, so it is run on
    the seeded database after ``build_timeline`` exactly as ``--redo cues``
    would be, and every row it writes is checked against the slot it serves."""
    cfg = _cfg(tmp_path)
    conn = _seed(cfg)
    rep_tl = s05_cut.build_timeline(cfg, conn)
    rep = s05_cut.build_cues(cfg, conn)
    slots = [dict(r) for r in conn.execute("SELECT * FROM timeline ORDER BY slot_index")]
    shots = _shots(conn)
    beats_rows = [dict(r) for r in conn.execute("SELECT * FROM story_beats ORDER BY rank, beat_id")]
    by_track: dict[str, list[dict]] = {}
    for c in conn.execute("SELECT * FROM audio_cues"):
        by_track.setdefault(c["track"], []).append(dict(c))
    assert rep["n_cues"] == {t: len(by_track.get(t, [])) for t in ("speech", "location", "music")}

    # every video slot hears its own recording, the stills hold the one before
    loc = {c["cue_id"]: c for c in by_track["location"]}
    for s in slots:
        if s["kind"] != "video":
            continue
        c = loc[f"lo_{s['slot_index']}"]
        assert c["source"] == shots[s["shot_id"]]["recording_id"]
        assert (c["t_in"], c["t_out"], c["src_in"], c["src_out"]) == (s["t_in"], s["t_out"], s["src_in"], s["src_out"])
    assert all(c["gain_lufs"] in {cfg.get("render.duck_lufs"), cfg.get("render.location_full_lufs"),
                                  cfg.get("render.location_under_speech_lufs")} for c in loc.values())

    # every speech beat with slots has its voice, starting where its first
    # slot reaches the words: the locked act slot opens on them, the cold
    # open runs into them after its extension
    speech = {c["cue_id"]: c for c in by_track["speech"]}
    for beat in ("b_arrive", "b_walk", "b_pass"):
        first = next(s for s in slots if s["beat_id"] == beat)
        c = speech[f"sp_{beat}"]
        assert c["source"] == shots[first["shot_id"]]["recording_id"]
        assert math.isclose(c["t_in"] - first["t_in"], c["src_in"] - first["src_in"], abs_tol=1e-3), beat
        if first["act"] >= 1:
            assert math.isclose(c["t_in"], first["t_in"], abs_tol=1e-3), beat
        assert c["gain_lufs"] == cfg.get("render.speech_lufs")
    # the cold open's line is heard again where act 4 delivers it
    assert rep_tl["cold_open_beat"] == "b_pass" and set(speech) == {"sp_b_arrive", "sp_b_walk", "sp_b_pass", "sp_b_pass_2"}
    again = next(s for s in slots if s["beat_id"] == "b_pass" and s["act"] == 4)
    c2 = speech["sp_b_pass_2"]
    assert math.isclose(c2["t_in"], again["t_in"], abs_tol=1e-3) and c2["t_in"] > speech["sp_b_pass"]["t_out"]
    assert (c2["src_in"], c2["src_out"]) == (speech["sp_b_pass"]["src_in"], speech["sp_b_pass"]["src_out"])

    # music covers each of its WINDOWS end to end (Gate 3, 2026-09-24: "Music
    # should be played only on those parts of the video where nothing else is
    # spoken"), on the table's own spans rather than the map's planned ones
    # (the map was not rebuilt, and the acts drifted from it by up to 8 s
    # here); the silence follows act 4's real end
    mmap = json.loads((cfg.work_root / "music" / "music_map.json").read_text())
    spans = {a: (min(s["t_in"] for s in slots if s["act"] == a), max(s["t_out"] for s in slots if s["act"] == a))
             for a in {s["act"] for s in slots}}
    assert rep_tl["music_map_recomputed"] == [] and any(
        abs(spans[a["act"]][0] - a["t_start"]) > 1.0 for a in mmap["acts"])
    q0 = spans[4][1]
    q1 = q0 + float(cfg.get("assemble.silence_window_s"))
    # the same three passes build_cues makes: onto the acts the table has,
    # onto the cuts the picture has, and clear of everything the rhythm pass
    # moved under the windows after they were placed
    natural_now, _ = s05_cut._natural_windows(
        cfg, s05_cut.effort.profile(s05_cut.place_mod.load_track(conn)), slots, shots)
    blocked = s05_cut._info_spans(
        cfg, slots, beats_rows, natural_now,
        cold_open_end_s=max((s["t_out"] for s in slots if s["act"] == 0), default=0.0))
    on_film = s05_cut.cues_mod.snap_windows_to_cuts(
        s05_cut.cues_mod.map_on_film_time(mmap, spans), slots, blocked=blocked,
        min_window_s=float(cfg.get("music.placement.min_window_s")))
    # the bed never plays over a line, a card or a natural-sound window
    assert not any(c["t_in"] < b1 and c["t_out"] > b0
                   for c in by_track["music"] for b0, b1 in blocked), \
        "a window the picture moved a line into must be trimmed clear of it"
    # and the gate reads what was played, not what was planned
    assert rep["music_share"] <= rep_tl["music_share_planned"] + 1e-6
    assert rep["music_share"] == pytest.approx(
        sum(c["t_out"] - c["t_in"] for c in by_track["music"]) /
        max(s["t_out"] for s in slots), abs=1e-3)
    music_spans = s05_cut.cues_mod.music_spans(on_film)
    assert music_spans, "the seeded film has somewhere music may play"
    for a in on_film["acts"]:
        mine = [c for c in by_track["music"] if c["cue_id"].startswith(f"mu_{a['act']}_")]
        assert bool(mine) == bool(a["segments"]), a["act"]
        for w in a["music_windows"]:
            lo, hi = float(w["t_start"]), float(w["t_end"])
            inside = [c for c in mine if lo - 1e-3 <= c["t_in"] < hi]
            assert inside, (a["act"], w)
            lo = q1 if lo < q1 <= hi else lo
            hi = q0 if lo <= q0 < hi else hi
            assert math.isclose(min(c["t_in"] for c in inside), lo, abs_tol=1e-3), (a["act"], w)
            assert math.isclose(max(c["t_out"] for c in inside), hi, abs_tol=1e-3), (a["act"], w)
    # and nowhere else: not in the silence, and not outside a window
    assert not any(c["t_in"] < q1 and c["t_out"] > q0 for c in by_track["music"])
    assert all(any(lo - 1e-3 <= c["t_in"] < hi + 1e-3 for lo, hi in music_spans)
               for c in by_track["music"])
    # the location track is the mix wherever no window covers it
    assert all(c["gain_lufs"] != cfg.get("render.duck_lufs")
               for c in loc.values()
               if not any(lo <= (c["t_in"] + c["t_out"]) / 2 <= hi for lo, hi in music_spans))
    # and the location cue that ends on the silence fades out over the window
    # fade, which places the silence the location track heard at act 4's end
    ending = [c for c in loc.values() if math.isclose(c["t_out"], q0, abs_tol=1e-3)]
    assert ending and all(c["fade_out_s"] == cfg.get("render.window_fade_s") for c in ending)
    assert all(c["fade_in_s"] == cfg.get("render.window_fade_s")
               for c in loc.values() if q0 - 1e-3 <= c["t_in"] <= q1 + 1e-3)
    assert all(c["gain_lufs"] == cfg.get("render.location_full_lufs")
               for c in loc.values() if q0 <= (c["t_in"] + c["t_out"]) / 2 <= q1)

    # the chat card carries the author's tag, never the name, and sits in act 1
    overlays = [dict(r) for r in conn.execute("SELECT * FROM overlays")]
    assert rep["n_overlays"] == len(overlays) == 1 and overlays[0]["overlay_id"] == "cc_q_plan"
    assert "Kulikov" not in json.dumps(overlays, ensure_ascii=False)
    assert json.loads(overlays[0]["payload"])["author_tag"] == "A"
    act1 = [s for s in slots if s["act"] == 1]
    assert act1[0]["t_in"] <= overlays[0]["t_in"] < overlays[0]["t_out"] <= act1[-1]["t_out"]

    # no report on disk: the windows are recomputed and match the timeline step's;
    # with one, the report's list is what is used
    assert rep["natural_windows"] == rep_tl["natural_windows"]
    made_up = [{"t_in": 1.0, "t_out": 2.0, "slot_index": 0}]
    cfg.work("reports", "s05_cut.json").write_text(json.dumps({"timeline": {"natural_windows": made_up}}))
    assert s05_cut.build_cues(cfg, conn)["natural_windows"] == made_up
    assert conn.execute("SELECT COUNT(*) FROM audio_cues").fetchone()[0] == sum(rep["n_cues"].values())
    conn.close()


def test_the_cues_step_reads_the_windows_the_timeline_step_just_wrote(tmp_path):
    """``nepal cut`` runs timeline then cues in one process, and cues reads
    the windows from the report on disk: written once after the loop, that
    was the previous run's list, mapped to the previous cut's slots, and
    the location track went to full level on the wrong shots."""
    cfg = _cfg(tmp_path)
    _seed(cfg).close()
    stale = [{"t_in": 1.0, "t_out": 2.0, "slot_index": 0}]
    cfg.work("reports", "s05_cut.json").write_text(json.dumps({"timeline": {"natural_windows": stale}}))
    rep = s05_cut.run(cfg, redo={"timeline", "cues"})
    assert rep["timeline"]["natural_windows"] != stale
    assert rep["cues"]["natural_windows"] == rep["timeline"]["natural_windows"]
    on_disk = json.loads(cfg.work("reports", "s05_cut.json").read_text())
    assert on_disk["cues"]["natural_windows"] == rep["timeline"]["natural_windows"] and on_disk["finished_utc"]
    assert not list(cfg.work_root.joinpath("reports").glob("*.tmp"))
