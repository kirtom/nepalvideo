"""S02 -- build the story spine.

No ML and no GPU. This is where the film's structure is decided, and it is
cheap enough to re-run freely while tuning.

Output is the chronological table the operator reads at the Milestone 1
checkpoint -- the spec's stated decision point for whether the project is worth
continuing -- plus ``work/music/music_map.json`` and the act boundaries that
S05 assigns shots against.
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from nepal import db, freshness
from nepal.config import Config
from nepal.probe import manifest
from nepal.util import proc as proc_util
from nepal.util.progress import Progress
from nepal.spine import acts as acts_mod
from nepal.spine import dem as dem_mod
from nepal.spine import geocode as geo_mod
from nepal.spine import gps as gps_mod
from nepal.spine import music as music_mod
from nepal.spine import playlist as playlist_mod
from nepal.spine import strava as strava_mod
from nepal.spine import telegram as tg_mod

log = logging.getLogger(__name__)
STAGE = "S02"
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return manifest.parse_exif_datetime(value)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------ S02.1 track

def build_gps_track(cfg: Config, conn) -> dict[str, Any]:
    """Merge phone photo EXIF, and any shared GPX, into one track."""
    photo_points: list[gps_mod.GpsPoint] = []
    for r in conn.execute(
        "SELECT created_at_utc, lat, lon, alt_dem_m, source FROM assets "
        "WHERE has_gps=1 AND lat IS NOT NULL AND lon IS NOT NULL "
        "AND created_at_utc IS NOT NULL AND source LIKE 'phone_%'"
    ):
        ts = _dt(r["created_at_utc"])
        if ts:
            photo_points.append(gps_mod.GpsPoint(ts, r["lat"], r["lon"], None, r["source"]))
    photo_points.sort(key=lambda p: p.ts)

    # A shared route file outranks photo EXIF: sampled every few seconds
    # rather than every few minutes.
    gpx_points: list[gps_mod.GpsPoint] = []
    gpx_files: list[str] = []
    chat_root = cfg.data_root / "chat_export"
    for pattern in ("**/*.gpx", "**/*.GPX"):
        for f in sorted(chat_root.glob(pattern)):
            pts = gps_mod.parse_gpx(f.read_text(errors="replace"))
            if pts:
                gpx_points.extend(pts)
                gpx_files.append(str(f.relative_to(cfg.data_root)))

    # The watch outranks everything: a fix a second, barometric altitude,
    # heart rate. Photo EXIF fills the days it did not run (jeep days, the
    # cities).
    strava_dir = cfg.data_root / str(cfg.get("spine.strava_dir", "strava"))
    activities, strava_points, strava_rep = strava_mod.load_strava(
        strava_dir, sample_s=float(cfg.get("spine.strava_sample_s", 5.0))) \
        if strava_dir.exists() else ([], [], {"skipped": f"no {strava_dir}"})

    merged = gps_mod.merge_points([strava_points, gpx_points, photo_points])
    merged = gps_mod.drop_outliers(merged)

    # S02.3 for the track itself. A barometric (Strava) or GPX elevation
    # outranks the DEM; photo points have none, so they take the DEM value.
    srtm = dem_mod.Srtm(cfg.srtm_dir)
    rows = []
    n_alt = 0
    for p in merged:
        alt, _src = dem_mod.resolve_altitude(srtm.elevation(p.lat, p.lon), p.ele, None)
        if alt is not None:
            n_alt += 1
        rows.append({"ts_utc": p.key(), "lat": p.lat, "lon": p.lon,
                     "alt_dem_m": alt, "source": p.source,
                     "hr_bpm": p.hr,
                     "alt_baro_m": p.ele if p.source == "strava" else None,
                     "activity_id": p.activity_id})
    # A re-run with a different sample spacing must not leave the old
    # spacing's points behind: the track is rebuilt, not accumulated.
    conn.execute("DELETE FROM gps_points")
    db.upsert(conn, "gps_points", ["ts_utc"], rows)
    per = strava_rep.get("points_per_activity", {})
    db.upsert(conn, "activities", ["activity_id"], [{
        "activity_id": a.activity_id, "name": a.name, "kind": a.kind,
        "start_utc": a.start_utc.isoformat(), "end_utc": a.end_utc.isoformat(),
        "elapsed_s": a.elapsed_s, "moving_s": a.moving_s, "distance_m": a.distance_m,
        "gain_m": a.gain_m, "hr_max": a.hr_max, "hr_avg": a.hr_avg,
        "filename": a.filename, "n_points": per.get(a.activity_id, 0),
    } for a in activities])
    if merged and not n_alt:
        log.warning("S02.3 no altitude for any track point -- SRTM tiles missing from "
                    "%s. Act segmentation will fall back to an even split.", srtm.dir)

    if merged:
        out = cfg.work("gpx", "trip.gpx")
        out.write_text(gps_mod.write_gpx(merged))
        env = gps_mod.envelope(merged)
        log.info("S02.1 %d GPS points (%d from Strava, %d from photos, %d from GPX), "
                 "%s .. %s", len(merged), len(strava_points), len(photo_points),
                 len(gpx_points), env[0].isoformat(), env[1].isoformat())
    else:
        log.error("S02.1 no GPS points at all -- the geolocation spine is empty. "
                  "Nothing downstream can be placed or act-assigned.")

    return {"n_points": len(merged), "n_from_photos": len(photo_points),
            "n_from_gpx": len(gpx_points), "gpx_files": gpx_files,
            "n_from_strava": len(strava_points), "strava": strava_rep,
            "n_activities": len(activities),
            "envelope": [p.isoformat() for p in gps_mod.envelope(merged)] if merged else None}


def _track(conn) -> list[gps_mod.GpsPoint]:
    out = []
    for r in conn.execute("SELECT ts_utc, lat, lon, alt_dem_m, source FROM gps_points "
                          "ORDER BY ts_utc"):
        ts = _dt(r["ts_utc"])
        if ts:
            out.append(gps_mod.GpsPoint(ts, r["lat"], r["lon"], r["alt_dem_m"], r["source"]))
    return out


# ------------------------------------------------- S02.2/3/4 place assets

def geotag_assets(cfg: Config, conn) -> dict[str, Any]:
    """Interpolate position for everything without its own fix, then attach
    DEM altitude and a place name."""
    track = _track(conn)
    if not track:
        return {"skipped": "no GPS track"}

    max_gap = float(cfg.get("spine.max_interp_gap_s"))
    srtm = dem_mod.Srtm(cfg.srtm_dir)
    gaz = geo_mod.Gazetteer(geo_mod.load_geonames(
        cfg.geonames_path))

    rows = [dict(r) for r in conn.execute(
        "SELECT asset_id, created_at_utc, lat, lon, has_gps FROM assets "
        "WHERE created_at_utc IS NOT NULL")]

    interpolated = refused = 0
    coords: list[tuple[float, float]] = []
    index: list[str] = []
    place_prog = Progress("S02.2 placing assets", len(rows))
    for r in rows:
        place_prog.step()
        lat, lon = r["lat"], r["lon"]
        if not r["has_gps"] or lat is None or lon is None:
            ts = _dt(r["created_at_utc"])
            got = gps_mod.interpolate_at(track, ts, max_gap_s=max_gap) if ts else None
            if got is None:
                refused += 1
                continue
            lat, lon = got
            interpolated += 1
        coords.append((lat, lon))
        index.append(r["asset_id"])

    place_prog.close(f"{interpolated} interpolated, {refused} refused")

    # cluster once, geocode once per place -- section S02.4
    clusters = geo_mod.cluster_coords(coords, float(cfg.get("spine.geocode_cluster_m")))
    place_of: dict[int, str | None] = {}
    for centroid, members in clusters:
        place = gaz.nearest(*centroid)
        for m in members:
            place_of[m] = place.name if place else None

    altitudes = 0
    for i, (aid, (lat, lon)) in enumerate(zip(index, coords)):
        alt_dem = srtm.elevation(lat, lon)
        alt, _src = dem_mod.resolve_altitude(alt_dem, None, None)
        if alt is not None:
            altitudes += 1
        conn.execute("UPDATE assets SET lat=?, lon=?, alt_dem_m=?, place_name=? "
                     "WHERE asset_id=?", (lat, lon, alt, place_of.get(i), aid))
    conn.commit()

    if srtm.missing:
        # The fetcher covers the trek bounding box only and rejects fixes far
        # from the route on purpose, so a tile missing here is almost always
        # one under home or a layover -- points that carry no altitude in the
        # film anyway. Saying "fetch it" sent one session chasing tiles the
        # fetcher will never download.
        log.warning("S02.3 %d SRTM tile(s) not present in %s: %s -- altitude is "
                    "null for those points. Expected for off-route material "
                    "(home, layovers); `nepal fetch-reference` only covers the "
                    "trek extent",
                    len(srtm.missing), srtm.dir, ", ".join(sorted(srtm.missing)[:8]))
    if not gaz.places:
        log.warning("S02.4 no GeoNames gazetteer at %s -- place names stay null",
                    cfg.geonames_path)

    log.info("S02.2 placed %d assets (%d interpolated, %d refused across gaps > %.0fh), "
             "%d with altitude, %d distinct places",
             len(index), interpolated, refused, max_gap / 3600, altitudes, len(clusters))
    return {"n_placed": len(index), "n_interpolated": interpolated,
            "n_refused": refused, "n_with_altitude": altitudes,
            "n_place_clusters": len(clusters),
            "missing_srtm_tiles": sorted(srtm.missing),
            "gazetteer_loaded": bool(gaz.places)}


# ------------------------------------------------------- S02.5 telegram

def parse_telegram(cfg: Config, conn) -> dict[str, Any]:
    result = cfg.data_root / "chat_export" / "result.json"
    if not result.exists():
        log.warning("S02.5 no chat_export/result.json -- Act 1 loses its core material")
        return {"skipped": "result.json absent"}

    messages = tg_mod.parse_export(result)
    track = _track(conn)
    # The envelope that decides message phase must describe the trek, not every
    # place a phone happened to have GPS on.
    env, n_off_route = gps_mod.trek_envelope(
        track, max_radius_km=float(cfg.get("spine.trek_radius_km", 200.0)))
    start, end = (env if env else (None, None))
    if start and end:
        log.info("S02.5 trek envelope %s .. %s (%d off-route points ignored)",
                 start.date(), end.date(), n_off_route)

    climb_events = _climb_events(track)
    tg_mod.mark_notable(messages, climb_events=climb_events,
                        window_s=float(cfg.get("spine.notable_window_s")))
    n_cards = tg_mod.mark_cards(messages, max_chars=int(cfg.get("spine.card_max_chars")))

    # link export media to the assets already manifested from disk
    chat_root = cfg.data_root / "chat_export"
    by_key = {r["s3_key"]: r["asset_id"] for r in
              conn.execute("SELECT asset_id, s3_key FROM assets WHERE source='telegram'")}
    linked = 0
    rows = []
    for m in messages:
        asset_id = None
        if m.media_path:
            resolved = tg_mod.resolve_media_path(m.media_path, chat_root)
            if resolved:
                key = f"raw/{resolved.relative_to(cfg.data_root).as_posix()}"
                asset_id = by_key.get(key)
                if asset_id:
                    linked += 1
                    # Telegram strips EXIF, so the message time IS the capture time
                    conn.execute(
                        "UPDATE assets SET created_at_utc=COALESCE(created_at_utc,?) "
                        "WHERE asset_id=?",
                        (m.ts_utc.astimezone(timezone.utc).isoformat(), asset_id))
        rows.append({
            "msg_id": m.msg_id,
            "ts_utc": m.ts_utc.astimezone(timezone.utc).isoformat(),
            "author": m.author, "text": m.text, "media_asset": asset_id,
            "phase": tg_mod.classify_phase(m.ts_utc, start, end),
            "is_notable": int(m.is_notable), "usable_as_card": int(m.usable_as_card),
        })
    db.upsert(conn, "messages", ["msg_id"], rows)
    conn.commit()

    vocab = tg_mod.extract_vocabulary(messages)
    cfg.work("vocab", "telegram_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=1))

    phases: dict[str, int] = {}
    for r in rows:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    log.info("S02.5 %d messages %s, %d notable, %d usable as cards, %d media linked, "
             "%d vocabulary terms", len(rows), phases,
             sum(r["is_notable"] for r in rows), n_cards, linked, len(vocab))
    return {"n_messages": len(rows), "phases": phases, "n_cards": n_cards,
            "n_media_linked": linked, "n_vocab": len(vocab), "vocab_sample": vocab[:20]}


def _climb_events(track: list[gps_mod.GpsPoint]) -> list[datetime]:
    """Times of local maxima in altitude gain -- what S02.5 marks notable near."""
    dated = [p for p in track if p.ele is not None]
    if len(dated) < 3:
        return []
    gains = [(dated[i].ts, dated[i].ele - dated[i - 1].ele) for i in range(1, len(dated))]
    if not gains:
        return []
    threshold = statistics.median([g for _, g in gains if g > 0] or [0]) * 2
    return [ts for ts, g in gains if g > max(threshold, 1.0)]


# ---------------------------------------------------------- S02.6 speech

def usable_audio_files(files: Sequence[Path]) -> tuple[list[Path], dict[str, int]]:
    """Split a folder listing into what can be transcribed and what cannot.

    Separated from the stage because it is the part that was wrong and the part
    worth testing, and a test for it should not have to load a 3 GB model.

    A Telegram export writes a thumbnail beside every round video. Handed a
    .jpg, faster-whisper passes it to PyAV, which asks the container for its
    first audio stream and raises IndexError from inside a generator -- an error
    naming neither the file nor the reason, which killed the whole stage after
    large-v3 had spent half a minute loading. Extensions are not trusted beyond
    dropping the obvious: a silent .mp4 is just as fatal, so every survivor is
    probed for an audio stream.
    """
    usable: list[Path] = []
    skipped: dict[str, int] = {}

    def note(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for f in files:
        if not f.is_file() or f.suffix.lower() in manifest.PHOTO_EXT:
            if f.is_file():
                note(f"not audio or video ({f.suffix.lower() or 'no extension'})")
            continue
        try:
            if not proc_util.probe_summary(f).get("has_audio"):
                note("no audio stream")
                continue
        except Exception as exc:                  # a bad file, not a bug
            log.debug("S02.6 could not probe %s: %s", f.name, exc)
            note(f"unreadable ({type(exc).__name__})")
            continue
        usable.append(f)
    return usable, skipped


def transcribe_round_videos(cfg: Config, conn) -> dict[str, Any]:
    """The best narration material in the project: face plus voice, timestamped
    and unperformed. Small folder, so it is transcribed here rather than waiting
    for S03's speech gating."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("S02.6 faster-whisper not installed -- skipping transcription "
                    "(pip install '.[asr]')")
        return {"skipped": "faster-whisper not installed"}

    folder = cfg.data_root / "chat_export" / "round_video_messages"
    listing = sorted(folder.glob("*")) if folder.exists() else []
    files, skipped = usable_audio_files(listing)
    if not files:
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            log.info("S02.6 %d file(s) carry nothing to transcribe: %s", n, reason)
        return {"skipped": "no round_video_messages with audio", "reasons": skipped}

    model_name = cfg.get("spine.whisper_model")
    language = cfg.get("spine.whisper_language")
    log.info("S02.6 transcribing %d round video message(s) with %s (%s)",
             len(files), model_name, language or "auto-detect")
    model = WhisperModel(model_name, device="auto", compute_type="int8")

    out_dir = cfg.workdir("transcripts")
    done = db.done_units(conn, f"{STAGE}.asr")
    results = []

    def note(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    bar = Progress("S02.6 transcribing", len([f for f in files if f.name not in done]))
    for f in files:
        if f.name in done:
            continue
        bar.step(note=f.name[-24:])
        try:
            segments, info = model.transcribe(str(f), language=language or None)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:                  # one file must not end the stage
            log.warning("S02.6 %s failed to transcribe: %s", f.name, exc)
            note(f"transcription failed ({type(exc).__name__})")
            continue
        (out_dir / f"{f.stem}.json").write_text(json.dumps(
            {"file": f.name, "language": info.language, "text": text},
            ensure_ascii=False, indent=1))
        results.append({"file": f.name, "chars": len(text), "language": info.language})
        db.mark_unit(conn, f"{STAGE}.asr", f.name)
    bar.close(f"{len(results)} transcribed")
    log.info("S02.6 transcribed %d file(s)", len(results))
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        log.info("S02.6 %d file(s) skipped: %s", n, reason)
    return {"n_transcribed": len(results), "files": results, "skipped": skipped}


# ----------------------------------------------------------- S02.7 music

def remove_stale_music(conn, keep: set[str]) -> int:
    """Drop music rows left behind by a previous run, and their children.

    Upsert alone leaves rows from a previous run in place, and they compete in
    the act assignment. Switching from a playlist to audio files left 25 stale
    playlist rows behind -- with no tempo, no key and a default energy of 0.50 --
    which is why the acts drew from tracks that were no longer there.

    Order matters. ``beats`` and ``music_sections`` both reference
    ``music_tracks(track_id)`` and foreign keys are enforced, so the children
    have to go before the parent or the delete aborts the whole stage.
    """
    stale = [r["track_id"] for r in conn.execute("SELECT track_id FROM music_tracks")
             if r["track_id"] not in keep]
    if not stale:
        return 0
    log.info("S02.7 removing %d music row(s) left by a previous run: %s",
             len(stale), ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else ""))
    rows = [(t,) for t in stale]
    conn.executemany("DELETE FROM beats WHERE track_id=?", rows)
    conn.executemany("DELETE FROM music_sections WHERE track_id=?", rows)
    conn.executemany("DELETE FROM music_tracks WHERE track_id=?", rows)
    conn.commit()
    return len(stale)


def analyse_music(cfg: Config, conn) -> dict[str, Any]:
    """Track features from audio files, or from a playlist export."""
    music_dir = cfg.data_root / "music"
    source = cfg.get("music.source", "auto")
    audio = sorted([f for f in music_dir.glob("*")
                    if f.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}]) \
        if music_dir.exists() else []

    csv_cfg = cfg.get("music.playlist_csv", None)
    csv_path = Path(csv_cfg).expanduser() if csv_cfg else None
    if csv_path is None and music_dir.exists():
        csvs = sorted(music_dir.glob("*.csv"))
        csv_path = csvs[0] if csvs else None

    if source == "auto":
        source = "audio" if audio else ("playlist" if csv_path else "none")

    tracks: list[music_mod.Track] = []
    report: dict[str, Any] = {"source": source}

    if source == "audio":
        if not audio:
            log.error("S02.7 music.source=audio but music/ holds no playable audio")
            return {"source": "audio", "error": "no audio files"}
        licences = _licence_manifest(music_dir)
        excluded: list[dict[str, Any]] = []
        for i, f in enumerate(audio, 1):
            try:
                t = music_mod.analyse_track(
                    f, n_sections=int(cfg.get("spine.music_segments")),
                    licence=music_mod.detect_licence(f, licences))
            except ImportError:
                log.warning("S02.7 librosa not installed (pip install '.[music]')")
                return {"source": "audio", "skipped": "librosa not installed"}
            except Exception as exc:                      # noqa: BLE001
                log.warning("S02.7 could not analyse %s: %s", f.name, exc)
                continue
            tracks.append(t)
            log.info("S02.7 [%d/%d] %s - %s: %.0f bpm, %d beats, dyn %.3f, key %s",
                     i, len(audio), t.artist or "?", t.title or f.stem,
                     t.tempo_bpm, len(t.beats), t.dyn_range, t.key_est)
        report["n_audio_files"] = len(audio)
        report["n_excluded"] = len(excluded)
        report["excluded"] = excluded

    elif source == "playlist":
        if csv_path is None or not csv_path.exists():
            log.error("S02.7 music.source=playlist but no CSV found under music/")
            return {"source": "playlist", "error": "no playlist csv"}
        tracks, prep = playlist_mod.parse_playlist(
            csv_path,
            min_instrumentalness=cfg.get("music.min_instrumentalness", None))
        report.update({
            "playlist_csv": str(csv_path.relative_to(cfg.data_root)),
            "n_rows": prep.n_rows, "n_tracks": prep.n_tracks,
            "n_excluded": prep.n_excluded,
            "has_audio_features": prep.has_audio_features,
            "notes": prep.notes,
            "excluded": [{"title": e.title, "artist": e.artist, "reason": e.reason}
                         for e in prep.exclusions if e.excluded],
        })
        for note in prep.notes:
            log.info("S02.7 %s", note)
        # A playlist has no waveform, so sections and beats are unavailable.
        # Give each track one nominal section so the music map still validates,
        # and mark the absence rather than fabricating a beat grid.
        for t in tracks:
            if not t.sections:
                t.sections = [{"section_id": f"{t.track_id}_s0", "track_id": t.track_id,
                               "start_s": 0.0, "end_s": t.duration_s or 180.0,
                               "energy": t.energy_mean, "is_swell": 1}]
        report["beat_grid_available"] = False
    else:
        log.error("S02.7 no music available: music/ has no audio and no playlist CSV. "
                  "Act assignment cannot run, and S06 has no beat grid to cut against.")
        return {"source": "none", "error": "no music available"}

    if not tracks:
        return {**report, "error": "no usable tracks"}

    report["n_stale_removed"] = remove_stale_music(conn, {t.track_id for t in tracks})

    db.upsert(conn, "music_tracks", ["track_id"], [{
        "track_id": t.track_id, "s3_key": t.s3_key, "title": t.title,
        # Written, not just logged: S05 rebuilds its Tracks from this table,
        # and the Act 5 callback is scored on the artist.
        "artist": t.artist, "duration_s": t.duration_s,
        "tempo_bpm": t.tempo_bpm, "key_est": t.key_est,
        "energy_mean": t.energy_mean, "energy_p95": t.energy_p95,
        "energy_p10": t.energy_p10, "centroid": t.centroid,
        "onset_rate": t.onset_rate, "assigned_act": None,
    } for t in tracks])
    db.upsert(conn, "music_sections", ["section_id"],
              [{"section_id": s["section_id"], "track_id": s["track_id"],
                "start_s": s["start_s"], "end_s": s["end_s"],
                "energy": s.get("energy", 0.0), "is_swell": int(s.get("is_swell", 0))}
               for t in tracks for s in t.sections])
    conn.execute("DELETE FROM beats")
    conn.executemany("INSERT INTO beats(track_id, t_s, is_downbeat) VALUES (?,?,?)",
                     [(t.track_id, b, int(b in set(t.downbeats)))
                      for t in tracks for b in t.beats])
    conn.commit()

    # One track may be reserved for the end credits. It is withdrawn before the
    # assignment rather than filtered afterwards: left in the pool it wins an
    # act on its features, and then the act it won has been scored against a cue
    # that is never going to play there.
    credits_name = cfg.get("music.credits_track", None)
    credits_track = music_mod.pick_credits_track(tracks, credits_name)
    if credits_name and credits_track is None:
        log.warning("S02.7 music.credits_track is set to %r but no track in "
                    "music/ matches it -- the credits have no cue and every "
                    "track is still competing for an act", credits_name)
    if credits_track is not None:
        db.set_decision(conn, "credits_track", credits_track.track_id,
                        confidence=1.0, method="config:music.credits_track")
        log.info("S02.7 %s - %s is held back for the end credits",
                 credits_track.artist or "?", credits_track.title or credits_track.s3_key)
    scored = [t for t in tracks if t is not credits_track]

    assignment = music_mod.assign_acts(
        scored, license_mode=str(cfg.get("music.license_mode")),
        feature_mask=playlist_mod.feature_mask_for(source))
    report["credits_track"] = credits_track.track_id if credits_track else None
    for t in tracks:
        conn.execute("UPDATE music_tracks SET assigned_act=? WHERE track_id=?",
                     (t.assigned_act, t.track_id))
    conn.commit()

    # The film may run to film.max_duration_s where the material earns it, but
    # what the material actually holds is not known until S05 has scored the
    # shots. Here the runtime is the target, and S05 re-runs this allocation --
    # and with it the music map -- once it can answer the question honestly.
    total_s = music_mod.choose_total_duration(
        float(cfg.get("film.target_duration_s")),
        float(cfg.get("film.max_duration_s")),
        min_s=float(cfg.get("film.min_duration_s")),
        material_s=None,
        growth_bias=float(cfg.get("film.growth_bias")),
        selectivity=float(cfg.get("film.material_selectivity")))
    mmap = music_mod.build_music_map(
        tracks, assignment, cfg.act_targets(),
        total_s=total_s,
        silence_s=float(cfg.get("assemble.silence_window_s")),
        feature_mask=playlist_mod.feature_mask_for(source),
        max_segment_s=float(cfg.get("music.max_segment_s")))
    problems = music_mod.check_music_map(
        mmap, target_s=total_s,
        tolerance_s=float(cfg.get("film.duration_tolerance_s")))
    cfg.work("music", "music_map.json").write_text(json.dumps(mmap, indent=2))

    for p in problems:
        log.warning("S02.8 %s", p)
    log.info("S02.7 %d tracks, assignment %s%s", len(tracks),
             {a: mmap_act["track_id"] for a, mmap_act in
              zip([x["act"] for x in mmap["acts"]], mmap["acts"])},
             " (FALLBACK: " + assignment.note + ")" if assignment.fallback_used else "")
    return {**report, "n_tracks": len(tracks),
            # string keys so the in-memory report and its JSON form agree, and
            # so this matches the Gate 1 return shape in the spec
            "assignment": {str(a): tid for a, tid in sorted(assignment.by_act.items())},
            "assignment_note": assignment.note,
            "callback_bonus": assignment.callback_bonus,
            "music_map_problems": problems}


def _licence_manifest(music_dir: Path) -> dict[str, str]:
    f = music_dir / "licences.json"
    if not f.exists():
        f = music_dir / "licenses.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            log.warning("could not parse %s", f)
    return {}


# ------------------------------------------------------- act boundaries

def segment_acts(cfg: Config, conn) -> dict[str, Any]:
    days = day_stats(conn)
    if not days:
        return {"skipped": "no dated assets"}

    planning = conn.execute(
        "SELECT MIN(ts_utc) t FROM messages WHERE phase='planning'").fetchone()
    after = conn.execute(
        "SELECT MAX(ts_utc) t FROM messages WHERE phase='after'").fetchone()

    bounds = acts_mod.segment_acts(
        days,
        planning_start=_dt(planning["t"]) if planning and planning["t"] else None,
        after_end=_dt(after["t"]) if after and after["t"] else None,
        max_gap_days=int(cfg.get("spine.act_max_gap_days", 3)),
        planning_window_days=int(cfg.get("spine.act_planning_window_days", 180)),
        after_window_days=int(cfg.get("spine.act_after_window_days", 30)))
    payload = [b.as_dict() for b in bounds]
    db.set_decision(conn, "act_boundaries", json.dumps(payload),
                    1.0 if any("changepoint" in b.method for b in bounds) else 0.4,
                    bounds[0].method if bounds else "")
    for b in bounds:
        log.info("S02 act %d: %s .. %s", b.act,
                 b.start_utc.astimezone(NEPAL_TZ).strftime("%Y-%m-%d %H:%M"),
                 b.end_utc.astimezone(NEPAL_TZ).strftime("%Y-%m-%d %H:%M"))
    return {"boundaries": payload}


def day_stats(conn) -> list[acts_mod.DayStat]:
    """One row per trek day, in Nepal local time."""
    rows = [dict(r) for r in conn.execute(
        "SELECT created_at_utc, alt_dem_m, place_name FROM assets "
        "WHERE created_at_utc IS NOT NULL")]
    track = [dict(r) for r in conn.execute(
        "SELECT ts_utc, alt_dem_m FROM gps_points WHERE alt_dem_m IS NOT NULL")]
    if not rows:
        return []

    times = sorted(t for t in (_dt(r["created_at_utc"]) for r in rows) if t)
    if not times:
        return []
    first = times[0]

    by_day: dict[int, list[float]] = {}
    bounds: dict[int, list[datetime]] = {}
    on_route: dict[int, int] = {}
    for r in rows:
        ts = _dt(r["created_at_utc"])
        if not ts:
            continue
        d = gps_mod.day_index(ts, first)
        bounds.setdefault(d, []).append(ts)
        if r["alt_dem_m"] is not None:
            by_day.setdefault(d, []).append(float(r["alt_dem_m"]))
        # An asset with a DEM altitude or a gazetteer place name is inside the
        # reference data's coverage, which is the trek region.
        if r["alt_dem_m"] is not None or r["place_name"]:
            on_route[d] = on_route.get(d, 0) + 1
    for r in track:
        ts = _dt(r["ts_utc"])
        if ts:
            by_day.setdefault(gps_mod.day_index(ts, first), []).append(float(r["alt_dem_m"]))

    out = []
    for d in sorted(bounds):
        alts = by_day.get(d) or []
        ts_list = sorted(bounds[d])
        out.append(acts_mod.DayStat(
            day_index=d,
            date=ts_list[0].astimezone(NEPAL_TZ).date().isoformat(),
            alt_max=max(alts) if alts else None,
            alt_min=min(alts) if alts else None,
            start_utc=ts_list[0], end_utc=ts_list[-1],
            # asset volume is how the trek window is found: a trek is a dense
            # burst of capture, planning photos are sparse and scattered
            asset_count=len(bounds[d]),
            on_route_count=on_route.get(d, 0)))
    return out


# ------------------------------------------------------------ chronology

def print_unreachable(conn, bounds: Sequence[acts_mod.ActBoundary]) -> dict[str, Any]:
    """Report media that falls in no act, with what it costs the film."""
    rows = conn.execute(
        "SELECT created_at_utc, kind, source, COALESCE(duration_s,0) dur, lat "
        "FROM assets WHERE created_at_utc IS NOT NULL "
        "AND kind IN ('video360','video_flat','photo')").fetchall()
    lost: list[Any] = [r for r in rows
                       if acts_mod.act_for(_dt(r["created_at_utc"]), bounds) is None]
    if not lost:
        return {"n": 0}

    all_video_s = sum(r["dur"] for r in rows if r["kind"] != "photo")
    lost_video_s = sum(r["dur"] for r in lost if r["kind"] != "photo")
    n_clips = sum(1 for r in lost if r["kind"] != "photo")
    n_photos = sum(1 for r in lost if r["kind"] == "photo")
    share = (lost_video_s / all_video_s * 100.0) if all_video_s else 0.0

    print(f"{len(lost)} media asset(s) fall outside every act window and cannot "
          f"enter the film: {n_clips} clip(s), {n_photos} photo(s)")
    if lost_video_s:
        print(f"  that is {lost_video_s/60:.0f} min of video, {share:.0f}% of all "
              f"{all_video_s/3600:.1f} h shot")
    by_source: dict[str, int] = {}
    for r in lost:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(by_source.items(), key=lambda kv: -kv[1])))
    positioned = sum(1 for r in lost if r["lat"] is not None)
    if positioned:
        print(f"  {positioned} of them DO carry a GPS position, so their timestamp is "
              f"wrong rather than their location -- run `nepal diagnose --timestamps`")
    return {"n": len(lost), "n_clips": n_clips, "n_photos": n_photos,
            "lost_video_s": round(lost_video_s, 1), "share_pct": round(share, 1)}


def print_chronology(cfg: Config) -> int:
    """The Milestone 1 checkpoint table.

    Per the spec this is the decision point for whether the project is worth
    continuing, so it is a table to read rather than JSON to parse.
    """
    conn = db.init(cfg.db_path)
    days = day_stats(conn)
    if not days:
        print("nothing to report -- run `nepal s01` then `nepal s02` first")
        conn.close()
        return 1

    bounds_raw = db.get_decision(conn, "act_boundaries")
    bounds = []
    if bounds_raw:
        for b in json.loads(bounds_raw):
            bounds.append(acts_mod.ActBoundary(b["act"], _dt(b["start_utc"]),
                                               _dt(b["end_utc"]), b.get("method", "")))

    first = days[0].start_utc
    window, outside = acts_mod.trek_window(
        days, max_gap_days=int(cfg.get("spine.act_max_gap_days", 3)))
    in_trek = {d.day_index for d in window}

    print(f"\n{'day':>4} {'':1} {'date':<10} {'act':>3}  {'place':<24} {'alt':>6}  "
          f"{'clips':>5} {'photos':>6} {'msgs':>5}")
    print("-" * 81)

    # Act 1 is built from everything BEFORE the trek: planning messages,
    # screenshots, gear photos, round video messages. Omitting it from the
    # checkpoint table hides whether the act has anything to work with.
    trek_lo = first.astimezone(timezone.utc).isoformat()
    pre = conn.execute(
        "SELECT SUM(CASE WHEN kind IN ('video360','video_flat') THEN 1 ELSE 0 END) clips,"
        " SUM(CASE WHEN kind='photo' THEN 1 ELSE 0 END) photos "
        "FROM assets WHERE created_at_utc IS NOT NULL AND created_at_utc < ?",
        (trek_lo,)).fetchone()
    pre_msgs = conn.execute("SELECT COUNT(*) n FROM messages WHERE ts_utc < ?",
                            (trek_lo,)).fetchone()["n"]
    pre_cards = conn.execute("SELECT COUNT(*) n FROM messages WHERE ts_utc < ? "
                             "AND usable_as_card=1", (trek_lo,)).fetchone()["n"]
    if (pre["clips"] or 0) or (pre["photos"] or 0) or pre_msgs:
        print(f"{'pre':>4} {'':1} {'planning':<10} {1:>3}  {'(before the trek)':<24} {'-':>6}  "
              f"{pre['clips'] or 0:>5} {pre['photos'] or 0:>6} {pre_msgs:>5}"
              f"   <- {pre_cards} usable as caption cards")

    for d in days:
        # Full Nepal-local calendar day, not the span between that day's first
        # and last asset: a message sent before the first photo or after the
        # last one belongs to the day a person would name, and counting it into
        # the asset span drops it from the table entirely.
        local_midnight = datetime.fromisoformat(d.date).replace(tzinfo=NEPAL_TZ)
        lo = local_midnight.astimezone(timezone.utc).isoformat()
        hi = (local_midnight + timedelta(days=1)).astimezone(timezone.utc).isoformat()
        counts = conn.execute(
            "SELECT "
            " SUM(CASE WHEN kind IN ('video360','video_flat') THEN 1 ELSE 0 END) clips,"
            " SUM(CASE WHEN kind='photo' THEN 1 ELSE 0 END) photos "
            "FROM assets WHERE created_at_utc >= ? AND created_at_utc < ?", (lo, hi)
        ).fetchone()
        msgs = conn.execute("SELECT COUNT(*) n FROM messages "
                            "WHERE ts_utc >= ? AND ts_utc < ?", (lo, hi)).fetchone()["n"]
        place = conn.execute(
            "SELECT place_name, COUNT(*) n FROM assets "
            "WHERE created_at_utc >= ? AND created_at_utc < ? AND place_name IS NOT NULL "
            "GROUP BY place_name ORDER BY n DESC LIMIT 1", (lo, hi)).fetchone()
        act = acts_mod.act_for(d.start_utc, bounds) if bounds else None
        alt = f"{d.alt_max:.0f}" if d.alt_max is not None else "-"
        mark = "*" if d.day_index in in_trek else " "
        print(f"{d.day_index:>4} {mark:1} {d.date:<10} {act if act else '-':>3}  "
              f"{(place['place_name'] if place else '-')[:24]:<24} {alt:>6}  "
              f"{counts['clips'] or 0:>5} {counts['photos'] or 0:>6} {msgs:>5}")

    tot = conn.execute(
        "SELECT COUNT(*) n, SUM(CASE WHEN kind IN ('video360','video_flat') THEN "
        "COALESCE(duration_s,0) ELSE 0 END) secs FROM assets").fetchone()
    print("-" * 81)
    # This table reads the database, not the media. A fix to a stage changes
    # nothing here until that stage is re-run, and an unchanged report is the
    # same whether the fix is wrong or was never applied. The timestamps
    # distinguish the two.
    built = {f"{r['stage']}.{r['unit_id']}": str(r["updated_at"])[:16]
             for r in conn.execute("SELECT stage, unit_id, updated_at "
                                   "FROM stage_units WHERE status='done'")}
    if built:
        shown = [(k, built[k]) for k in ("S01.manifest", "S01.clock", "S02.geotag",
                                         "S02.music", "S02.acts") if k in built]
        print("computed (UTC): "
              + ", ".join(f"{k}={v}" for k, v in shown or built.items()))
        stale_pairs = freshness.stale_units(conn)
        if stale_pairs:
            names = ", ".join(f"{st}.{u}" for st, u in stale_pairs)
            print(f"  STALE: {names} predate the code now installed -- they were "
                  f"computed by the previous version.")
            print(f"         {freshness.rerun_command(stale_pairs)}")
            if any(u in freshness.EXPENSIVE.get(st, ()) for st, u in stale_pairs):
                print(f"         (the FOV and clock solves are the slow part, and "
                      f"--skip-fov/--skip-clock would skip exactly these)")
    secs = tot["secs"] or 0
    dur = f"{secs/3600:.1f} h" if secs >= 3600 else f"{secs/60:.1f} min"
    print(f"{tot['n']} assets, {dur} of video, {len(days)} days carrying material")
    if window:
        trek_assets = sum(d.asset_count for d in window)
        print(f"* the trek: days {window[0].day_index}-{window[-1].day_index}, "
              f"{window[0].date} .. {window[-1].date}, {len(window)} days, "
              f"{trek_assets} assets. Act boundaries come from these days only.")
    if outside:
        far = [d for d in outside if d.day_index > (window[-1].day_index + 60)] if window else []
        if far:
            print(f"  {len(far)} day(s) of material sit far outside the trek "
                  f"({', '.join(d.date for d in far[:4])}"
                  f"{' ...' if len(far) > 4 else ''}) -- almost always a device clock "
                  f"error rather than real material from that date")

    # Material outside every act window cannot enter the timeline. Naming the
    # day it sits on is not enough: what matters is how much film is being lost,
    # and on real material one mis-clocked camera accounted for nearly half the
    # footage. An unreachable clip is a silent deletion unless it is counted.
    if bounds:
        print_unreachable(conn, bounds)

    # Anything the spine could not place is worth naming: an unplaced shot
    # cannot be act-assigned, and silently dropping it shrinks the film.
    unplaced = conn.execute(
        "SELECT COUNT(*) n FROM assets WHERE created_at_utc IS NOT NULL "
        "AND lat IS NULL AND kind IN ('video360','video_flat','photo')").fetchone()["n"]
    if unplaced:
        print(f"{unplaced} media asset(s) have no position -- outside the GPS envelope "
              f"or across a gap longer than "
              f"{float(cfg.get('spine.max_interp_gap_s'))/3600:.0f} h")
    # Distinguish missing reference data (actionable) from assets that are simply
    # not in the trek region -- planning photos taken at home have no Nepal
    # altitude or place name and never should, so reporting them as a missing
    # tile sends the operator to fetch data that would change nothing.
    trek_box = conn.execute(
        "SELECT MIN(lat) la0, MIN(lon) lo0, MAX(lat) la1, MAX(lon) lo1 "
        "FROM assets WHERE alt_dem_m IS NOT NULL").fetchone()

    def _split(column: str) -> tuple[int, int]:
        if not trek_box or trek_box["la0"] is None:
            n = conn.execute(f"SELECT COUNT(*) n FROM assets WHERE lat IS NOT NULL "
                             f"AND {column} IS NULL").fetchone()["n"]
            return n, 0
        pad = 1.0
        inside = conn.execute(
            f"SELECT COUNT(*) n FROM assets WHERE lat IS NOT NULL AND {column} IS NULL "
            f"AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (trek_box["la0"] - pad, trek_box["la1"] + pad,
             trek_box["lo0"] - pad, trek_box["lo1"] + pad)).fetchone()["n"]
        total = conn.execute(
            f"SELECT COUNT(*) n FROM assets WHERE lat IS NOT NULL "
            f"AND {column} IS NULL").fetchone()["n"]
        return inside, total - inside

    alt_inside, alt_outside = _split("alt_dem_m")
    if alt_inside:
        print(f"{alt_inside} asset(s) in the trek region have no altitude -- "
              f"SRTM tile missing, run `nepal fetch-reference`")
    if alt_outside:
        print(f"{alt_outside} asset(s) have no altitude because they are outside the "
              f"trek region (planning photos taken elsewhere) -- expected")

    place_inside, place_outside = _split("place_name")
    if place_inside:
        gaz = cfg.geonames_path
        print(f"{place_inside} asset(s) in the trek region have no place name -- "
              + (f"gazetteer missing at {gaz}, run `nepal fetch-reference`"
                 if not gaz.exists() else
                 "no named place within 5 km of those coordinates"))
    if place_outside:
        print(f"{place_outside} asset(s) have no place name because they are outside "
              f"the gazetteer's country -- expected")

    if bounds:
        print("\nacts:")
        for b in bounds:
            print(f"  {b.act}  {b.start_utc.astimezone(NEPAL_TZ):%Y-%m-%d %H:%M} .. "
                  f"{b.end_utc.astimezone(NEPAL_TZ):%Y-%m-%d %H:%M}  {b.method[:44]}")

    mm = cfg.work_root / "music" / "music_map.json"
    if mm.exists():
        m = json.loads(mm.read_text())
        # Every segment, not just the act's opening track. Printing only the
        # primary made a film built from nine pieces look like a film built
        # from five, which is not a detail when the question is whether the
        # soundtrack has enough variety.
        used = {seg["track_id"] for a in m["acts"] for seg in a.get("segments", [])}
        print(f"\nmusic map: {m['total_duration_s']:.0f}s total, "
              f"{len(used)} of {m.get('n_tracks_available', '?')} track(s) used")
        for a in m["acts"]:
            print(f"  act {a['act']} {a['name'] or '':<14} "
                  f"{a['t_start']:>7.1f}..{a['t_end']:>7.1f}s  "
                  f"swells={len(a['swells'])}")
            for seg in a.get("segments", []) or []:
                span = seg["t_end"] - seg["t_in"]
                label = f"{seg.get('artist') or '?'} - {seg.get('title') or seg['track_id']}"
                print(f"        {seg['t_in']:>7.1f}..{seg['t_end']:>7.1f}s "
                      f"({span:>5.1f}s)  {label[:58]}")
        sw = m.get("silence_window") or {}
        if sw:
            print(f"  silence after Act 4: {sw['t_start']:.1f}..{sw['t_end']:.1f}s")

    print()
    conn.close()
    return 0


# ---------------------------------------------------------------- driver

S02_UNITS = ("gps_track", "telegram", "geotag", "asr", "music", "acts")


def run(cfg: Config, *, force: bool = False, skip_asr: bool = False,
        redo: set[str] | None = None) -> dict[str, Any]:
    redo = set(redo or ())
    unknown = redo - set(S02_UNITS)
    if unknown:
        raise SystemExit(f"unknown --redo unit(s): {', '.join(sorted(unknown))}; "
                         f"valid: {', '.join(S02_UNITS)}")
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    done = db.done_units(conn, STAGE)
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal s02 --force")

    # Order matters. Telegram strips EXIF, so Telegram media takes its capture
    # time from the message that carried it -- which means S02.5 has to run
    # before geotagging, or Act 1's core material is never placed at all.
    steps = [
        ("gps_track", lambda: build_gps_track(cfg, conn)),
        ("telegram", lambda: parse_telegram(cfg, conn)),
        ("geotag", lambda: geotag_assets(cfg, conn)),
        ("music", lambda: analyse_music(cfg, conn)),
        ("acts", lambda: segment_acts(cfg, conn)),
    ]
    if not skip_asr:
        steps.insert(3, ("asr", lambda: transcribe_round_videos(cfg, conn)))

    for name, fn in steps:
        if not force and name in done and name not in redo:
            report[name] = {"skipped": "already done"}
            continue
        report[name] = fn()
        db.mark_unit(conn, STAGE, name, detail=json.dumps(report[name], default=str)[:2000])

    report["finished_utc"] = db.utcnow()
    cfg.work("reports", "s02_spine.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log.info("S02 report written to %s", cfg.work_root / "reports" / "s02_spine.json")
    conn.close()
    return report
