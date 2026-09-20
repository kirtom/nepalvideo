"""S05-S07 driven end to end: score, assemble, render.

One driver rather than three, because the three are one question -- what is
the film -- and the operator iterates on the whole answer. The weights move,
the cut is rebuilt, the draft is watched; splitting that across three commands
would make the loop three times as long for no gain.

The Bedrock halves are deliberately absent: S04.3's captions feed one term of
``score_sem``, and S06's ordering refinement is a polish pass over a list that
is already ordered. Neither is required to produce a cut, and S05's
missing-term rule means the ranking is correct without them rather than
merely usable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nepal import db, freshness
from nepal import select as select_mod
from nepal.config import Config
from nepal.process import assemble as asm, render as render_mod, score as score_mod
from nepal.process import longtake as longtake_mod, pairs as pairs_mod, rhythm as rhythm_mod
from nepal.process import timeline_io
from nepal.spine import effort, music as music_mod, place as place_mod, scenes as scenes_mod
from nepal.story import anchors as anchors_mod
from nepal.util.progress import Progress

log = logging.getLogger(__name__)
STAGE = "S05"


def _load_embeddings(cfg: Config) -> dict[str, np.ndarray]:
    path = cfg.work_root / "semantic" / "clip.npy"
    idx = cfg.work_root / "semantic" / "clip_index.json"
    if not path.exists() or not idx.exists():
        log.info("S06 no CLIP embeddings yet; MMR will fall back to score order "
                 "(diversity unenforced, which shows as repetition in the cut)")
        return {}
    E = np.load(path)
    ids = json.loads(idx.read_text())["shot_ids"]
    return {sid: E[i] for i, sid in enumerate(ids) if i < len(E)}


def _message_proximity(conn, when: str | None, *, window_s: float = 1200.0) -> float:
    if not when:
        return 0.0
    row = conn.execute(
        "SELECT MIN(ABS(strftime('%s', ts_utc) - strftime('%s', ?))) "
        "FROM messages WHERE ts_utc IS NOT NULL", (when,)).fetchone()
    if not row or row[0] is None:
        return 0.0
    return float(max(0.0, 1.0 - float(row[0]) / window_s))


def score_shots(cfg: Config, conn) -> dict[str, Any]:
    """S05 -- score every surviving shot, then shortlist."""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM shots WHERE status IN ('candidate','shortlisted') "
        "ORDER BY COALESCE(start_utc, ''), shot_id")]
    if not rows:
        return {"n_shots": 0, "note": "nothing survived the gate"}

    w = cfg.get("score")
    peak = cfg.get("score.duration_fit_peak_s", [4.0, 8.0])
    ranks = score_mod.percentile_rank([r.get("sharpness") for r in rows])

    # "first at a place" and "a new altitude record" are facts about the
    # sequence, not the shot, so they are computed in chronological order once.
    seen_places: set[str] = set()
    max_alt = float("-inf")
    for i, r in enumerate(rows):
        place = r.get("place_name")
        first_here = bool(place) and place not in seen_places
        if place:
            seen_places.add(place)
        alt = r.get("alt_dem_m")
        new_alt = alt is not None and float(alt) > max_alt
        if alt is not None:
            max_alt = max(max_alt, float(alt))
        r["_tech"] = score_mod.score_tech(r, sharpness_rank=ranks.get(i),
                                          weights=w["tech"], peak=peak)
        r["_sem"] = score_mod.score_sem(r, clip_similarity=None, weights=w["sem"])
        r["_ctx"] = score_mod.score_ctx(
            r, new_max_alt=new_alt, first_at_place=first_here,
            msg_proximity=_message_proximity(conn, r.get("start_utc")),
            weights=w["ctx"])
        r["score_total"] = score_mod.combine(r["_tech"], r["_sem"], r["_ctx"], w["total"])

    conn.executemany(
        "UPDATE shots SET score_tech=?, score_sem=?, score_ctx=?, score_total=? "
        "WHERE shot_id=?",
        [(r["_tech"], r["_sem"], r["_ctx"], r["score_total"], r["shot_id"]) for r in rows])

    picked = set(score_mod.shortlist(rows, size=int(cfg.get("score.shortlist_size", 400)),
                                     min_per_act=int(cfg.get("score.min_per_act", 40))))
    conn.execute("UPDATE shots SET status='candidate' WHERE status='shortlisted'")
    conn.executemany("UPDATE shots SET status='shortlisted' WHERE shot_id=?",
                     [(s,) for s in picked])
    conn.commit()

    by_act: dict[Any, int] = {}
    for r in rows:
        if r["shot_id"] in picked:
            by_act[r.get("act")] = by_act.get(r.get("act"), 0) + 1
    totals = [r["score_total"] for r in rows if r["score_total"] is not None]
    log.info("S05 scored %d shot(s); shortlisted %d; per act %s", len(rows), len(picked),
             {k: by_act[k] for k in sorted(by_act, key=lambda x: (x is None, x))})
    if totals:
        log.info("S05 score_total %s", {k: round(float(v), 3) for k, v in
                 zip(("p5", "p50", "p95"), np.percentile(totals, [5, 50, 95]))})
    n_sem = sum(1 for r in rows if r["_sem"] is not None)
    log.info("S05 %d of %d shot(s) have a semantic score (captions are a "
             "Bedrock stage; the rest are ranked on what is known)", n_sem, len(rows))
    return {"n_shots": len(rows), "n_shortlisted": len(picked),
            "per_act": {str(k): v for k, v in by_act.items()}, "n_with_sem": n_sem}


# -- the timeline v2 (Film v2 step 4) ----------------------------------
#
# The order of operations is the order in which each decision needs the one
# before it: the voice spine (anchors) claims its moments first, the fixed
# pictures (the bridge take, the phone pairs) take theirs, the fill goes
# between them, and only then is there a picture track to group into scenes
# and hand music; only with music is there a beat grid to re-time against.

# Act 1 is the planning: screenshots, the itinerary, a message or two. Levity
# cannot be shot there, so the per-act minimum is asked of the trek onward.
LEVITY_FROM_ACT = 2


def _shot_rows(conn) -> list[dict[str, Any]]:
    """Every surviving shot with what assembly reads beyond the row: its
    source (the recording's for video, the asset's for a photo), whether the
    recording is 360, and the frame size (the recording's first chapter's,
    since a recording's shots carry none) so ``pairs.is_portrait`` can tell a
    phone held up from the camera."""
    return [dict(r) for r in conn.execute(
        "SELECT s.*, COALESCE(r.source, a.source) AS source, r.is_360 AS is_360, "
        "COALESCE(a.width, fa.width) AS width, COALESCE(a.height, fa.height) AS height "
        "FROM shots s "
        "LEFT JOIN recordings r ON r.recording_id = s.recording_id "
        "LEFT JOIN assets a ON a.asset_id = s.asset_id "
        "LEFT JOIN assets fa ON fa.asset_id = ("
        "  SELECT asset_id FROM assets WHERE recording_id = s.recording_id "
        "  ORDER BY chapter_index, asset_id LIMIT 1) "
        "WHERE s.status <> 'rejected' AND s.act IS NOT NULL "
        "ORDER BY COALESCE(s.start_utc, ''), s.shot_id")]


def _act_spans(bounds: Sequence[Mapping[str, Any]]) -> dict[int, tuple[float | None, float | None]]:
    """The act boundaries as epoch seconds -- what ``place_anchors`` compares."""
    out = {}
    for b in bounds:
        lo, hi = (anchors_mod._epoch(b.get(k)) for k in ("start_utc", "end_utc"))
        out[int(b["act"])] = (lo, hi)
    return out


def _tracks(conn) -> list[music_mod.Track]:
    """The music library as S02.7 stored it, with sections and the beat grid."""
    by_id: dict[str, music_mod.Track] = {}
    for r in conn.execute("SELECT * FROM music_tracks"):
        by_id[r["track_id"]] = music_mod.Track(
            track_id=r["track_id"], s3_key=r["s3_key"] or "", title=r["title"] or "",
            duration_s=float(r["duration_s"] or 0.0), tempo_bpm=float(r["tempo_bpm"] or 0.0),
            key_est=r["key_est"], energy_mean=float(r["energy_mean"] or 0.0),
            energy_p95=float(r["energy_p95"] or 0.0), energy_p10=float(r["energy_p10"] or 0.0),
            centroid=float(r["centroid"] or 0.0), onset_rate=float(r["onset_rate"] or 0.0),
            assigned_act=r["assigned_act"])
    for r in conn.execute("SELECT * FROM music_sections ORDER BY track_id, start_s"):
        if r["track_id"] in by_id:
            by_id[r["track_id"]].sections.append(
                {"section_id": r["section_id"], "track_id": r["track_id"],
                 "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                 "energy": r["energy"], "is_swell": int(r["is_swell"] or 0)})
    for r in conn.execute("SELECT track_id, t_s, is_downbeat FROM beats ORDER BY t_s"):
        t = by_id.get(r["track_id"])
        if t is None:
            continue
        t.beats.append(float(r["t_s"]))
        if r["is_downbeat"]:
            t.downbeats.append(float(r["t_s"]))
    return list(by_id.values())


def _slot_utc(slot: Mapping[str, Any], shot: Mapping[str, Any]) -> datetime | None:
    """When the slot's first frame was shot: the shot's stamp moved forward by
    how far into the shot the slot starts."""
    ts = anchors_mod._epoch(shot.get("start_utc"))
    if ts is None:
        return None
    offset = 0.0
    if slot.get("src_in") is not None:
        offset = float(slot["src_in"]) - float(shot.get("start_s") or 0.0)
    return datetime.fromtimestamp(ts + offset, tz=timezone.utc)


def _attrs_for(prof: Sequence[effort.Effort], rows: Sequence[Mapping[str, Any]], *,
               beats: Sequence[Mapping[str, Any]]):
    """What ``group_scenes`` asks about a slot, answered from the effort
    profile at the slot's own moment and from the shot row; None wherever
    the track has no fix near enough, so a slot with no sample is not
    thereby "resting"."""
    by_id = {r["shot_id"]: r for r in rows}
    speech = {b["beat_id"] for b in beats if b.get("kind") == "speech"}
    levity = {b["beat_id"] for b in beats if b.get("levity")}

    def attrs(slot: Mapping[str, Any]) -> dict[str, Any]:
        shot = by_id.get(slot.get("shot_id")) or {}
        ts = _slot_utc(slot, shot) if shot else None
        e = effort.at(prof, ts) if ts is not None else None
        return {"speed_ms": e.speed_ms if e else None, "hr_bpm": e.hr_bpm if e else None,
                "gain_m_per_h": e.gain_m_per_h if e else None, "alt_m": e.alt_m if e else None,
                "hour": effort.nepal_hour(ts) if ts is not None else None,
                "place_name": shot.get("place_name"), "act": slot.get("act"),
                "levity": bool(shot.get("tag_levity")) or slot.get("beat_id") in levity,
                "motion_mag": shot.get("motion_mag"),
                "under_speech": slot.get("beat_id") in speech}
    return attrs


def _new_slot(**fields: Any) -> dict[str, Any]:
    slot: dict[str, Any] = {k: None for k in asm.SLOT_KEYS if k != "slot_index"}
    slot.update(speed=1.0, transition="cut", locked=0)
    slot.update(fields)
    return slot


def _is_photo(row: Mapping[str, Any]) -> bool:
    return str(row.get("media_kind") or row.get("kind") or "video") == "photo"


def _duration_range(cfg: Config, act: int) -> list[float]:
    d = cfg.get("assemble.shot_duration_s")
    return [float(x) for x in (d.get(act) or d.get(str(act)))]


def _set_length(slot: dict[str, Any], t_in: float, t_out: float) -> dict[str, Any]:
    """Move a slot's film span and keep ``src_out`` honest with it -- render.py
    cuts ``t_out - t_in`` from ``src_in``, so a stale ``src_out`` is a lie."""
    slot["t_in"], slot["t_out"] = round(t_in, 3), round(t_out, 3)
    if slot.get("src_in") is not None:
        slot["src_out"] = round(float(slot["src_in"]) + (slot["t_out"] - slot["t_in"]), 3)
    return slot


def _no_adjacent_photos(ordered: Sequence[Mapping[str, Any]], *, prev_photo: bool,
                        next_photo: bool) -> list[Mapping[str, Any]]:
    """Stills are punctuation; two in a row is a slideshow. The later of any
    two neighbouring stills is dropped, the slot's neighbours outside this
    run counting as its first and last neighbour."""
    out: list[Mapping[str, Any]] = []
    last_photo = prev_photo
    for r in ordered:
        if _is_photo(r) and last_photo:
            continue
        out.append(r)
        last_photo = _is_photo(r)
    if out and next_photo and _is_photo(out[-1]):
        out.pop()
    return out


def _fill_gap(cfg: Config, cands: Sequence[Mapping[str, Any]], *, act: int, t0: float, t1: float,
              similarity, existing: Sequence[Mapping[str, Any]], prev_photo: bool,
              next_photo: bool) -> list[dict[str, Any]]:
    """The next-best shots for one gap of the act, laid from its start with
    no grid yet -- the rhythm pass snaps them once there is music to snap to.

    ``existing`` are the shot rows already on screen in this act, so the
    per-place cap counts the whole act, not just this gap.
    """
    rng = _duration_range(cfg, act)
    budget = asm.slot_budget(t1 - t0, rng)
    place_cap = int(cfg.get("assemble.max_shots_per_place_per_act"))
    run_cap = int(cfg.get("assemble.max_consecutive_recording"))
    after = int(cfg.get("assemble.source_alternation_after"))

    def admissible(c, chosen):
        if _is_photo(c) and (_is_photo(chosen[-1]) if chosen else prev_photo):
            return False
        return (asm.place_count_ok(c, list(existing) + chosen, limit=place_cap)
                and asm.recording_run_ok(c, chosen, limit=run_cap))

    def prefer(c, chosen):
        return asm.source_alternation_bonus(c, chosen, after=after)

    chosen = asm.mmr_select(cands, budget=budget, similarity=similarity,
                            lam=float(cfg.get("assemble.mmr_lambda")),
                            admissible=admissible, prefer=prefer)
    taken = {c["shot_id"] for c in chosen}
    if log.isEnabledFor(logging.DEBUG):
        left = [c for c in cands if c["shot_id"] not in taken]
        refused = {"photo_rule": sum(1 for c in left if _is_photo(c) and (_is_photo(chosen[-1]) if chosen else prev_photo)),
                   "place_cap": sum(1 for c in left if not asm.place_count_ok(c, list(existing) + list(chosen), limit=place_cap)),
                   "recording_run": sum(1 for c in left if not asm.recording_run_ok(c, list(chosen), limit=run_cap))}
        log.debug("S06 act %d fill %.1f-%.1fs: budget %d, %d candidate(s), chose %d, %d left of which "
                  "refused %s", act, t0, t1, budget, len(cands), len(chosen), len(left), refused)
    chosen = asm.source_share_repair(chosen, [c for c in cands if c["shot_id"] not in taken],
                                     min_share=float(cfg.get("assemble.source_share_min")))
    ordered = _no_adjacent_photos(asm.chronological(chosen), prev_photo=prev_photo,
                                  next_photo=next_photo)
    return asm.lay_out(ordered, start_s=t0, duration_range=rng, beats=[])


def _resolve_overlaps(slots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """A slot ends where the next one begins. A locked slot carries a beat
    or the bridge and never gives way to an unlocked one: an unlocked slot
    that runs into it is cut there, one that starts inside it is dropped
    (the fill loop may reclaim the time after it)."""
    ordered = sorted(slots, key=lambda s: (float(s["t_in"]), -int(s.get("locked") or 0)))
    out: list[dict[str, Any]] = []
    for s in ordered:
        if out and float(s["t_in"]) < float(out[-1]["t_out"]) - 1e-6:
            prev = out[-1]
            if prev.get("locked") and s.get("locked"):
                # A beat never cuts a beat. Two locked slots overlap only
                # when place_anchors clamped a crowded act's anchors onto
                # each other, and chronology is already gone there -- so the
                # later one simply follows, whole.
                length = float(s["t_out"]) - float(s["t_in"])
                _set_length(s, float(prev["t_out"]), float(prev["t_out"]) + length)
            elif prev.get("locked"):
                continue
            else:
                _set_length(prev, float(prev["t_in"]), float(s["t_in"]))
                if prev["t_out"] - prev["t_in"] <= 1e-6:
                    out.pop()
        out.append(s)
    return out


def _close_holes(slots: Sequence[dict[str, Any]], t0: float) -> list[dict[str, Any]]:
    """Whatever no material could cover is closed, not left black: every
    slot after a hole moves up by the hole's length, locked or not -- an
    anchor was placed by chronology, and with nothing to show before it,
    chronology says it comes next."""
    out: list[dict[str, Any]] = []
    cursor = t0
    for s in slots:
        length = float(s["t_out"]) - float(s["t_in"])
        if float(s["t_in"]) > cursor + 1e-6:
            _set_length(s, cursor, cursor + length)
        out.append(s)
        cursor = float(s["t_out"])
    return out


def _last_slot(by_act: Mapping[int, Sequence[dict[str, Any]]], acts: Sequence[int], act: int,
               act0: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The slot on screen when this act begins: the previous act's last,
    else the title card (act 0 always ends on it)."""
    earlier = [a for a in acts if a < act and by_act.get(a)]
    return by_act[earlier[-1]][-1] if earlier else act0[-1]


def _gaps(slots: Sequence[Mapping[str, Any]], t0: float, t1: float, *,
          min_len: float) -> list[tuple[float, float, int]]:
    """Uncovered spans of [t0, t1) worth a shot, each with the index of the
    slot that follows it (``len(slots)`` for the tail)."""
    out = []
    cursor = t0
    for i, s in enumerate(slots):
        if float(s["t_in"]) - cursor > min_len:
            out.append((cursor, float(s["t_in"]), i))
        cursor = max(cursor, float(s["t_out"]))
    if t1 - cursor > min_len:
        out.append((cursor, t1, len(slots)))
    return out


def _utc_window(prev: Mapping[str, Any] | None, nxt: Mapping[str, Any] | None,
                shots_by_id: Mapping[str, Mapping[str, Any]],
                act_utc: tuple[float | None, float | None]) -> tuple[float | None, float | None]:
    """What moments a gap may show: after the slot before it, before the slot
    after it, inside the act. None on a side that has no stamp to bound it."""
    def edge(slot, default, end):
        if slot is None or not slot.get("shot_id"):
            return default
        ts = _slot_utc(slot, shots_by_id.get(slot["shot_id"]) or {})
        if ts is None:
            return default
        return ts.timestamp() + ((float(slot["t_out"]) - float(slot["t_in"])) if end else 0.0)
    return edge(prev, act_utc[0], True), edge(nxt, act_utc[1], False)


def _in_window(row: Mapping[str, Any], lo: float | None, hi: float | None) -> bool:
    ts = anchors_mod._epoch(row.get("start_utc"))
    if ts is None:
        return True                     # an unstamped shot may go anywhere; chronological() puts it last
    return (lo is None or ts >= lo) and (hi is None or ts <= hi)


def _fixed_anchor(kind: str, slot: Mapping[str, Any], shot: Mapping[str, Any]) -> anchors_mod.Anchor:
    """A pair or the bridge take as an anchor, so ``place_anchors`` lays it
    among the speech anchors by the same chronology and the same gaps."""
    ts = _slot_utc(slot, shot)
    return anchors_mod.Anchor(
        beat_id=f"{kind}:{slot['shot_id']}", act=int(slot["act"]), kind=kind,
        utc=ts.isoformat() if ts else None, recording_id=shot.get("recording_id"),
        shot_id=slot["shot_id"], src_in=float(slot["src_in"]), src_out=float(slot["src_out"]),
        duration_s=float(slot["src_out"]) - float(slot["src_in"]), own_picture=True,
        face_hold_s=0.0, text="", effect="none")


def plan_act(cfg: Config, act: int, act_rows: Sequence[Mapping[str, Any]],
             beats: Sequence[Mapping[str, Any]], *, t0: float, act_len_s: float,
             act_span_utc: tuple[float | None, float | None], similarity,
             photo_budget: Sequence[str], long_take: Mapping[str, Any] | None,
             pairs: Sequence[Mapping[str, Any]], used: set[str], prev_photo: bool = False
             ) -> tuple[list[dict[str, Any]], list[anchors_mod.Anchor]]:
    """One act's picture track before any music exists.

    The speech anchors, the bridge take and the pairs are placed first by
    chronology; every gap between them is filled by MMR from the shots whose
    moment falls inside that gap; then each speech anchor gets its picture --
    its own recording for the whole utterance when the speaker is not in
    frame (a walking shot), else the face for ``face_hold_s`` and B-roll to
    the end of the line.
    """
    shots_by_id = {r["shot_id"]: r for r in act_rows}
    act_beats = [b for b in beats if b.get("act") == act and b.get("shot_id") in shots_by_id]
    anchors = anchors_mod.speech_anchors(
        act_beats, shots_by_id, face_hold_s=float(cfg.get("beats.face_hold_s")),
        pre_roll_s=float(cfg.get("beats.pre_roll_s")),
        own_picture_below=float(cfg.get("beats.own_picture_face_score_below")))
    fixed: dict[str, dict[str, Any]] = {}
    for kind, slot in [("long_take", long_take)] + [("pair", p) for p in pairs]:
        if slot is not None:
            a = _fixed_anchor(kind, slot, shots_by_id[slot["shot_id"]])
            fixed[a.beat_id] = dict(slot)
            anchors.append(a)
    anchors = anchors_mod.place_anchors(anchors, act_t0=t0, act_len_s=act_len_s,
                                        act_utc0=act_span_utc[0], act_utc1=act_span_utc[1])

    # What the fill may not touch: the voices' recordings, the bridge's, and
    # every shot already on screen (both halves of a pair included).
    excluded_recordings = {a.recording_id for a in anchors if a.recording_id and a.kind != "pair"}
    for slot in fixed.values():
        used.update(x for x in (slot.get("shot_id"), slot.get("secondary_shot_id")) if x)
    allowed_photos = set(photo_budget)
    pool = [r for r in act_rows if r.get("recording_id") not in excluded_recordings
            and (not _is_photo(r) or r["shot_id"] in allowed_photos)]
    rng = _duration_range(cfg, act)

    slots: list[dict[str, Any]] = []
    prev_end, prev_utc = t0, act_span_utc[0]
    for a in anchors:
        a_end = a.t_in + a.duration_s
        if a.t_in - prev_end > rng[0]:
            cands = [r for r in pool if r["shot_id"] not in used
                     and _in_window(r, prev_utc, anchors_mod._epoch(a.utc))]
            filled = _fill_gap(cfg, cands, act=act, t0=prev_end, t1=a.t_in, similarity=similarity,
                               existing=[shots_by_id[s["shot_id"]] for s in slots if s.get("shot_id")],
                               prev_photo=prev_photo, next_photo=False)
            slots.extend(filled)
            used.update(s["shot_id"] for s in filled)
        if a.beat_id in fixed:
            slots.append(_set_length(fixed[a.beat_id], a.t_in, a_end))
        else:
            slots.extend(_anchor_slots(cfg, a, pool, used, t_in=a.t_in, rng=rng))
        prev_end, prev_photo = a_end, False
        a_ts = anchors_mod._epoch(a.utc)
        prev_utc = a_ts + a.duration_s if a_ts is not None else prev_utc
    if t0 + act_len_s - prev_end > rng[0]:
        cands = [r for r in pool if r["shot_id"] not in used and _in_window(r, prev_utc, act_span_utc[1])]
        filled = _fill_gap(cfg, cands, act=act, t0=prev_end, t1=t0 + act_len_s, similarity=similarity,
                           existing=[shots_by_id[s["shot_id"]] for s in slots if s.get("shot_id")],
                           prev_photo=prev_photo, next_photo=False)
        slots.extend(filled)
        used.update(s["shot_id"] for s in filled)
    out = _close_holes(_resolve_overlaps(slots), t0)
    log.info("S06 act %d plan: %d anchor(s) (%d fixed), pool %d of %d rows, %d slot(s) covering "
             "%.1fs of %.1fs planned", act, len(anchors), len(fixed), len(pool), len(act_rows),
             len(out), (float(out[-1]["t_out"]) - t0) if out else 0.0, act_len_s)
    return out, [a for a in anchors if a.beat_id not in fixed]


def _anchor_slots(cfg: Config, a: anchors_mod.Anchor, pool: Sequence[Mapping[str, Any]],
                  used: set[str], *, t_in: float, rng: Sequence[float]) -> list[dict[str, Any]]:
    """The picture under one speech beat: locked on the speaker's own shot
    first, B-roll from nearby recordings to the end of the utterance.
    ``pool`` is what the fill may use -- the other voices' recordings and
    the bridge are not B-roll for anyone."""
    a_end = t_in + a.duration_s
    hold = a.duration_s if a.own_picture else a.face_hold_s
    first = _new_slot(kind="video", act=a.act, shot_id=a.shot_id, src_in=round(a.src_in, 3),
                      beat_id=a.beat_id, locked=1)
    _set_length(first, t_in, t_in + hold)
    used.add(a.shot_id)
    if a.own_picture or a_end - (t_in + hold) <= 1e-6:
        return [first]
    broll = [r for r in anchors_mod.broll_candidates(a, pool, window_s=float(cfg.get("beats.broll_window_s")))
             if r["shot_id"] not in used]
    if not broll:
        # Nobody else was filming near this moment: the face stays up for
        # the whole line rather than cutting to nothing.
        return [_set_length(first, t_in, a_end)]
    n = asm.slot_budget(a_end - (t_in + hold), rng)
    laid = asm.lay_out(broll[:n], start_s=t_in + hold, duration_range=rng, beats=[])
    for s in laid:
        s["beat_id"] = a.beat_id
        used.add(s["shot_id"])
    last = laid[-1]
    if last["t_out"] < a_end:
        # Stretch the last B-roll to the end of the line, as far as its shot allows.
        avail = asm.shot_available_s(next(r for r in broll if r["shot_id"] == last["shot_id"]))
        _set_length(last, last["t_in"], min(a_end, last["t_in"] + avail))
    return [first] + laid


def _cold_open(cfg: Config, beats: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
               ) -> list[dict[str, Any]]:
    """Act 0: the cold open and the title card, both locked, from t = 0."""
    shots_by_id = {r["shot_id"]: r for r in rows}
    length_range = [float(x) for x in cfg.get("film.cold_open_s")]
    usable = [b for b in beats if b.get("kind") != "speech" or b.get("shot_id") in shots_by_id]
    cold = anchors_mod.cold_open_pick(usable, shots_by_id, length_range=length_range)
    if cold is None:
        # No late speech beat to open on: the best of the summit's footage.
        best = max((r for r in rows if r.get("act") == 4 and not _is_photo(r)
                    and r.get("score_total") is not None),
                   key=lambda r: float(r["score_total"]), default=None)
        if best is not None:
            src_in = float(best["start_s"])
            cold = _new_slot(kind="video", shot_id=best["shot_id"], act=4, src_in=src_in,
                             src_out=src_in + min(length_range[1], asm.shot_available_s(best)),
                             transition="dip_black", locked=1)
    out: list[dict[str, Any]] = []
    t = 0.0
    if cold is not None:
        cold = dict(cold)
        cold["act"] = 0
        _set_length(cold, 0.0, float(cold["src_out"]) - float(cold["src_in"]))
        out.append(cold)
        t = cold["t_out"]
    card = _new_slot(kind="card", act=0, locked=1,
                     motion=json.dumps({"type": "card", "text": cfg.get("film.cold_open_card")},
                                       ensure_ascii=False))
    out.append(_set_length(card, t, t + float(cfg.get("assemble.card_s"))))
    return out


def _sections_for_act(entry: Mapping[str, Any], tracks_by_id: Mapping[str, music_mod.Track], *,
                      shift: float, t0: float, t1: float) -> list[dict[str, Any]]:
    """The act's music on the film timeline, one piece per track section so
    the rhythm pass can rank energies -- the map's segments are act-local
    and a merged run may span several sections.

    The last piece is stretched to the act's planned end (and one silent
    piece covers an act with no music at all): ``retime`` drops whatever
    lies past its last section, and the fill that follows must be able to
    reach the whole act, not just the part the planned slots covered.
    """
    out: list[dict[str, Any]] = []
    t_start = float(entry.get("t_start", 0.0)) + shift
    for seg in entry.get("segments", []):
        f_in, f_out = t_start + float(seg["t_in"]), t_start + float(seg["t_end"])
        base = t_start + float(seg["t_in"]) - float(seg["src_in"])
        track = tracks_by_id.get(seg["track_id"])
        pieces = [(max(f_in, base + float(s["start_s"])), min(f_out, base + float(s["end_s"])), s.get("energy"))
                  for s in (track.sections if track else [])
                  if base + float(s["end_s"]) > f_in and base + float(s["start_s"]) < f_out]
        covered = max((p[1] for p in pieces), default=f_in)
        if covered < f_out - 1e-6:
            pieces.append((covered, f_out, None))   # past the piece's end: nothing to read
        out += [{"t_in": round(a, 3), "t_out": round(b, 3), "energy": e} for a, b, e in pieces if b > a]
    if out:
        out[-1]["t_out"] = max(out[-1]["t_out"], round(t1, 3))
    else:
        out = [{"t_in": round(t0, 3), "t_out": round(t1, 3), "energy": None}]
    return out


def _check_material(slot: Mapping[str, Any], shots_by_id: Mapping[str, Mapping[str, Any]],
                    recordings_by_id: Mapping[str, Mapping[str, Any]], long_take_id: str | None) -> None:
    """The trap, asserted before anything is written: a slot never claims
    footage its shot does not have -- a split on either of its two shots.
    The bridge take alone is bounded by its recording, because running past
    the first shot is what a long take is."""
    if slot.get("kind") != "video" or not slot.get("shot_id"):
        return
    shot = shots_by_id[slot["shot_id"]]
    length = float(slot["t_out"]) - float(slot["t_in"])
    src_in = float(slot["src_in"] if slot.get("src_in") is not None else shot["start_s"])
    if slot["shot_id"] == long_take_id and shot.get("recording_id") in recordings_by_id:
        avail = float(recordings_by_id[shot["recording_id"]].get("duration_s") or 0.0) - src_in
    else:
        avail = min(asm.shot_available_s(shot), float(shot["end_s"]) - src_in)
    if length > avail + 1e-3:
        raise RuntimeError(
            f"slot on {slot['shot_id']} at {slot['t_in']}s claims {length:.2f}s but the "
            f"source has {avail:.2f}s from src_in={src_in:.2f}: the timeline would over-report")
    if slot.get("secondary_shot_id"):
        # A split shows two clips for the same span, so the other half is
        # bounded the same way from its own offset.
        other = shots_by_id[slot["secondary_shot_id"]]
        other_in = float(slot["secondary_src_in"] if slot.get("secondary_src_in") is not None
                         else other["start_s"])
        other_avail = float(other["end_s"]) - other_in
        if length > other_avail + 1e-3:
            raise RuntimeError(
                f"split on {slot['shot_id']}+{slot['secondary_shot_id']} at {slot['t_in']}s "
                f"claims {length:.2f}s but the second source has {other_avail:.2f}s from "
                f"src_in={other_in:.2f}: the timeline would over-report")


def _scene_id_by_time(slot: Mapping[str, Any], scenes: Sequence[scenes_mod.Scene], shift: float
                      ) -> int | None:
    """The scene a slot belongs to, by where it now sits: the map is kept
    from before the rhythm pass, so a slot the pass moved or added takes the
    scene whose span (shifted with its act) contains it, else the nearest."""
    t = float(slot["t_in"]) - shift
    same = [sc for sc in scenes if sc.act == slot.get("act")]
    for sc in same:
        if sc.t_in <= t < sc.t_out:
            return sc.scene_id
    nearest = min(same, key=lambda sc: abs(sc.t_in - t), default=None)
    return nearest.scene_id if nearest else None


def _scenes_and_map(cfg: Config, slots: Sequence[Mapping[str, Any]], attrs, tracks, *,
                    act_spans: Mapping[int, tuple[float, float]], act0_span: tuple[float, float],
                    total_s: float) -> tuple[list[scenes_mod.Scene], dict[str, Any], str, list[str]]:
    """Scenes from the picture track, a section of a track per scene, and
    the map S06's audio graph reads. Act 0 slots are grouped too so every
    slot has a scene; the map takes Act 0's music from Act 4's swell."""
    beats_spans: dict[str, list[float]] = {}
    for s in slots:
        if s.get("beat_id"):
            span = beats_spans.setdefault(s["beat_id"], [float(s["t_in"]), float(s["t_out"])])
            span[0], span[1] = min(span[0], float(s["t_in"])), max(span[1], float(s["t_out"]))
    scenes = scenes_mod.group_scenes(
        slots, attrs, min_scene_s=float(cfg.get("music.min_scene_s")),
        max_scene_s=float(cfg.get("music.max_segment_s")),
        beats_spans=[tuple(v) for v in beats_spans.values()],
        cities=list(cfg.get("assemble.cities")))
    mode = str(cfg.get("music.assignment"))
    if mode == "act":
        # S02.7's per-act choice, laid through the same machinery: every
        # scene of an act on the act's track from its first section.
        by_act = {t.assigned_act: t for t in tracks if t.assigned_act is not None and t.sections}
        assignment = music_mod.SceneAssignment(
            {sc.scene_id: (by_act[sc.act].track_id, by_act[sc.act].sections[0]["section_id"])
             for sc in scenes if sc.act in by_act}, 0.0, "per-act tracks from S02.7")
    else:
        hr_rest, hr_max = float(cfg.get("effort.hr_rest_bpm")), float(cfg.get("effort.hr_max_bpm"))
        credits = cfg.get("music.credits_track", None)
        assignment = music_mod.assign_scenes(
            scenes, tracks,
            targets={sc.scene_id: scenes_mod.scene_target(sc, hr_rest=hr_rest, hr_max=hr_max)
                     for sc in scenes},
            weights=cfg.get("music.scene_targets"),
            switch_cost=float(cfg.get("music.switch_cost")),
            continuity_bonus=float(cfg.get("music.continuity_bonus")),
            repeat_penalty=float(cfg.get("music.repeat_penalty")),
            reuse_gap_s=float(cfg.get("music.reuse_gap_s")),
            preferred=list(cfg.get("music.preferred_tracks", []) or []),
            preferred_bonus=float(cfg.get("music.preferred_bonus")),
            exclude=[credits] if credits else [])
    mmap = music_mod.music_map_from_scenes(
        scenes, assignment, tracks, act_spans=act_spans,
        silence_s=float(cfg.get("assemble.silence_window_s")), act0_span=act0_span)
    problems = music_mod.check_music_map(
        mmap, target_s=total_s, tolerance_s=float(cfg.get("film.duration_tolerance_s")))
    return scenes, mmap, mode, problems


def build_timeline(cfg: Config, conn) -> dict[str, Any]:
    """S06 -- the picture track v2: anchors, pairs, the long take, the fill,
    scenes, music, rhythm; then the table, the map and the OTIO/FCPXML."""
    rows = _shot_rows(conn)
    if not rows:
        return {"n_slots": 0, "note": "nothing survived the gate"}
    shots_by_id = {r["shot_id"]: r for r in rows}
    recordings = [dict(r) for r in conn.execute("SELECT * FROM recordings")]
    recordings_by_id = {r["recording_id"]: r for r in recordings}
    beats = [dict(r) for r in conn.execute("SELECT * FROM story_beats ORDER BY rank, beat_id")]
    orphaned = [b["beat_id"] for b in beats if b.get("kind") == "speech" and b.get("shot_id") not in shots_by_id]
    if orphaned:
        log.warning("S06 %d speech beat(s) sit on shots that no longer survive and are "
                    "skipped: %s", len(orphaned), orphaned)
    act_span_utc = _act_spans(json.loads(db.get_decision(conn, "act_boundaries") or "[]"))
    prof = effort.profile(place_mod.load_track(conn))
    tracks = _tracks(conn)

    # -- how long, and how long each act ------------------------------------
    material_s = sum(min(asm.shot_available_s(r), 20.0) for r in rows)
    total_s = music_mod.choose_total_duration(
        float(cfg.get("film.target_duration_s")), float(cfg.get("film.max_duration_s")),
        min_s=float(cfg.get("film.min_duration_s")), material_s=material_s,
        growth_bias=float(cfg.get("film.growth_bias")),
        selectivity=float(cfg.get("film.material_selectivity")))
    act0 = _cold_open(cfg, beats, rows)
    t0 = float(act0[-1]["t_out"])
    # The cold open is carved off the top; the acts share the rest.
    act_len = music_mod.allocate_act_durations(cfg.act_targets(), total_s - t0)
    acts = sorted(act_len)
    similarity = asm.make_similarity(_load_embeddings(cfg),
                                     fallback_weights=cfg.get("assemble.similarity_fallback"))

    # -- the fixed pictures: pairs and the bridge ---------------------------
    video_rows = [r for r in rows if not _is_photo(r)]
    pairs = pairs_mod.find_pairs(video_rows, window_s=float(cfg.get("assemble.pair_window_s")),
                                 per_act=int(cfg.get("assemble.pairs_per_act")))
    long_take = None
    bridge = longtake_mod.bridge_candidates(video_rows, recordings,
                                            keywords=list(cfg.get("assemble.long_take_keywords")))
    if bridge:
        by_rec: dict[str, list] = {}
        for r in video_rows:
            by_rec.setdefault(r["recording_id"], []).append(r)
        long_take = longtake_mod.long_take_slot(bridge[0], recordings_by_id, by_rec,
                                               length_range=[float(x) for x in cfg.get("assemble.long_take_s")])
    long_take_id = long_take["shot_id"] if long_take else None

    photo_rows = {act: [{"shot_id": r["shot_id"], "score": float(r.get("score_total") or 0.0),
                         "duration_s": float(r["end_s"]) - float(r["start_s"]), "act": act}
                        for r in rows if r.get("act") == act and _is_photo(r)] for act in acts}
    photo_plan = select_mod.plan_photo_slots(
        photo_rows, {a: act_len[a] for a in acts}, share=float(cfg.get("film.photo_share")),
        share_by_act={int(k): float(v) for k, v in (cfg.get("film.photo_share_by_act") or {}).items()})

    # -- the plan, act by act ------------------------------------------------
    used: set[str] = {s["shot_id"] for s in act0 if s.get("shot_id")}
    planned: dict[int, list[dict[str, Any]]] = {}
    planned_spans: dict[int, tuple[float, float]] = {}
    n_anchors = 0
    quality: dict[str, dict[str, Any]] = {}
    t = t0
    for act in acts:
        act_rows = [r for r in rows if r.get("act") == act]
        slots, anchors = plan_act(
            cfg, act, act_rows, beats, t0=t, act_len_s=act_len[act],
            act_span_utc=act_span_utc.get(act, (None, None)), similarity=similarity,
            photo_budget=photo_plan[act].chosen,
            long_take=long_take if long_take and long_take.get("act") == act else None,
            pairs=[p for p in pairs if p.get("act") == act], used=used,
            prev_photo=_is_photo(_last_slot(planned, acts, act, act0)))
        planned[act] = slots
        planned_spans[act] = (t, t + act_len[act])
        n_anchors += len(anchors)
        chosen_rows = [shots_by_id[s["shot_id"]] for s in slots if s.get("shot_id") in shots_by_id]
        quality[str(act)] = {
            "needs_subject": asm.needs_subject(chosen_rows, runtime_s=act_len[act],
                                               every_s=float(cfg.get("assemble.subject_shot_every_s"))),
            "missing_levity": asm.missing_levity(chosen_rows, minimum=int(cfg.get("assemble.levity_min_per_act")))
            if act >= LEVITY_FROM_ACT else 0}
        t += act_len[act]

    # -- scenes, then music ---------------------------------------------------
    attrs = _attrs_for(prof, rows, beats=beats)
    all_slots = list(act0) + [s for a in acts for s in planned[a]]
    scenes, mmap, mode, problems = _scenes_and_map(
        cfg, all_slots, attrs, tracks, act_spans=planned_spans, act0_span=(0.0, t0), total_s=total_s)
    tracks_by_id = {tr.track_id: tr for tr in tracks}

    # -- rhythm, act by act, each act ending where its last cut lands ------
    rhythm = cfg.get("assemble.rhythm")
    max_loops = int(cfg.get("assemble.max_reassembly_loops"))
    final: dict[int, list[dict[str, Any]]] = {}
    final_spans: dict[int, tuple[float, float]] = {}
    shifts: dict[int, float] = {}
    t = t0
    for act in acts:
        entry = next((a for a in mmap["acts"] if int(a["act"]) == act), {})
        shift = t - planned_spans[act][0]
        shifts[act] = shift
        act_t0, act_t1 = t, t + act_len[act]
        sections = _sections_for_act(entry, tracks_by_id, shift=shift, t0=act_t0, t1=act_t1)
        grid = [b + shift for b in entry.get("beat_grid", [])]
        downs = [b + shift for b in entry.get("downbeats", [])]
        silence_t = act_t1 if act == 4 and mmap.get("silence_window") else None
        rng = _duration_range(cfg, act)
        slots = [_set_length(dict(s), float(s["t_in"]) + shift, float(s["t_out"]) + shift)
                 for s in planned[act]]
        act_rows = [r for r in rows if r.get("act") == act]
        excluded = {shots_by_id[s["shot_id"]].get("recording_id") for s in slots
                    if s.get("locked") and s.get("shot_id")}
        allowed_photos = set(photo_plan[act].chosen)

        # The burst runs on the act's loudest section, and with one section
        # that is the whole act: no swell to rank against, no burst.
        burst = rhythm["burst_slots"] if len(sections) >= 2 else (0, 0)

        def retimed(current):
            return _resolve_overlaps(rhythm_mod.retime(
                current, sections=sections, beats=grid, downbeats=downs, table=rhythm,
                burst_slots=burst, is_act4=(act == 4),
                held_shot_s=cfg.get("assemble.act4_held_shot_s"), silence_t=silence_t,
                shots=shots_by_id))

        log.info("S06 act %d rhythm: %d section(s) %.1f-%.1fs, grid %d beat(s) to %.1fs, planned "
                 "%.1f-%.1fs", act, len(sections), sections[0]["t_in"], sections[-1]["t_out"],
                 len(grid), max(grid, default=act_t0), act_t0, act_t1)
        for round_no in range(max_loops + 1):
            before = len(slots)
            slots = retimed(slots)
            gaps = _gaps(slots, act_t0, act_t1, min_len=rng[0])
            log.info("S06 act %d round %d: retime kept %d of %d slot(s), end %.1fs of %.1fs, "
                     "%d gap(s) totalling %.1fs", act, round_no, len(slots), before,
                     float(slots[-1]["t_out"]) if slots else act_t0, act_t1, len(gaps),
                     sum(g1 - g0 for g0, g1, _ in gaps))
            if not gaps or round_no == max_loops:
                break
            on_screen = {x for s in slots + act0
                         for x in (s.get("shot_id"), s.get("secondary_shot_id")) if x}
            existing = [shots_by_id[x] for x in on_screen if x in shots_by_id]
            added: list[dict[str, Any]] = []
            for g0, g1, idx in gaps:
                prev = slots[idx - 1] if idx > 0 else _last_slot(final, acts, act, act0)
                nxt = slots[idx] if idx < len(slots) else None
                lo, hi = _utc_window(prev, nxt, shots_by_id, act_span_utc.get(act, (None, None)))
                taken = on_screen | {s["shot_id"] for s in added}
                free = [r for r in act_rows if r["shot_id"] not in taken
                        and r.get("recording_id") not in excluded
                        and (not _is_photo(r) or r["shot_id"] in allowed_photos)]
                cands = [r for r in free if _in_window(r, lo, hi)]
                filled = _fill_gap(cfg, cands, act=act, t0=g0, t1=g1, similarity=similarity,
                                   existing=existing + [shots_by_id[s["shot_id"]] for s in added],
                                   prev_photo=bool(prev and prev.get("kind") == "photo"),
                                   next_photo=bool(nxt and nxt.get("kind") == "photo"))
                log.debug("S06 act %d round %d gap %.1f-%.1fs (%.1fs): utc window %s..%s, "
                          "%d of %d free row(s) inside, %d slot(s) laid to %.1fs", act, round_no,
                          g0, g1, g1 - g0, lo, hi, len(cands), len(free), len(filled),
                          float(filled[-1]["t_out"]) if filled else g0)
                added += filled
            log.info("S06 act %d round %d: %d slot(s) added for %d gap(s)", act, round_no,
                     len(added), len(gaps))
            if not added:
                break
            slots = _resolve_overlaps(slots + added)
        # What the material could not fill is closed, and the cuts re-snapped
        # to the grid from where they now sit.
        slots = _close_holes(retimed(_close_holes(slots, act_t0)), act_t0)
        # retime drops slots and the fill sees one gap at a time, so two
        # stills can still meet -- across an act boundary too. The later
        # one goes, and what follows it moves up.
        slots = _close_holes(_no_adjacent_photos(
            slots, prev_photo=_is_photo(_last_slot(final, acts, act, act0)), next_photo=False), act_t0)
        final[act] = slots
        end = float(slots[-1]["t_out"]) if slots else act_t0
        final_spans[act] = (act_t0, end)
        log.info("S06 act %d final: %d slot(s), %.1fs of %.1fs planned (%+.1fs)", act, len(slots),
                 end - act_t0, act_len[act], end - act_t1)
        t = end

    # The map was built on the planned spans; only a real change of length
    # earns a new one, because the cuts above were timed against its grid.
    moved = {a for a in acts if abs((final_spans[a][1] - final_spans[a][0]) - act_len[a])
             > float(cfg.get("music.min_scene_s"))}
    ordered = list(act0) + [s for a in acts for s in final[a]]
    if moved:
        scenes, mmap, mode, problems = _scenes_and_map(
            cfg, ordered, attrs, tracks, act_spans=final_spans, act0_span=(0.0, t0), total_s=total_s)
        for sc in scenes:
            for i in sc.slot_indices:
                ordered[i]["scene_id"] = sc.scene_id
    else:
        for s in ordered:
            s["scene_id"] = _scene_id_by_time(s, scenes, shifts.get(int(s["act"]), 0.0))
    cfg.work("music", "music_map.json").write_text(json.dumps(mmap, indent=2, ensure_ascii=False))
    for p in problems:
        log.warning("S06 music map: %s", p)

    # -- the natural-sound windows, on film time through the slot they hit --
    windows = effort.hardest_windows(
        prof, count=int(cfg.get("assemble.natural_sound_windows")),
        window_s=sum(float(x) for x in cfg.get("assemble.natural_sound_window_s")) / 2.0)
    natural: list[dict[str, Any]] = []
    for w0, w1 in windows:
        centre = w0 + (w1 - w0) / 2
        for i, s in enumerate(ordered):
            if not s.get("shot_id"):
                continue
            s_utc = _slot_utc(s, shots_by_id[s["shot_id"]])
            if s_utc is None:
                continue
            length = float(s["t_out"]) - float(s["t_in"])
            if s_utc <= centre <= s_utc + timedelta(seconds=length):
                mid = float(s["t_in"]) + (centre - s_utc).total_seconds()
                half = (w1 - w0).total_seconds() / 2
                natural.append({"t_in": round(mid - half, 3), "t_out": round(mid + half, 3), "slot_index": i})
                break

    # -- the write ---------------------------------------------------------------
    for s in ordered:
        _check_material(s, shots_by_id, recordings_by_id, long_take_id)
    table_rows = [{**{c: s.get(c) for c in db.TIMELINE_V2_COLUMNS}, "slot_index": i}
                  for i, s in enumerate(ordered)]
    conn.execute("DELETE FROM timeline")
    db.upsert(conn, "timeline", ["slot_index"], table_rows)
    paths = timeline_io.write(table_rows, cfg.workdir("."), media_dir=str(cfg.work_root / "proxies"),
                              fps=int(cfg.get("render.fps")))

    duration = float(ordered[-1]["t_out"]) if ordered else 0.0
    per_act: dict[str, int] = {}
    for s in ordered:
        per_act[str(s["act"])] = per_act.get(str(s["act"]), 0) + 1
    log.info("S06 %d slot(s), %.1f min (planned %.1f); per act %s; %d anchor(s), %d pair(s), "
             "long take %s, %d scene(s), %d natural-sound window(s)", len(ordered), duration / 60,
             total_s / 60, per_act, n_anchors, len(pairs), long_take_id, len(scenes), len(natural))
    log.info("S06 wrote %s and %s", paths["otio"].name, paths["fcpxml"].name)
    return {"n_slots": len(ordered), "duration_s": round(duration, 3), "planned_s": total_s,
            "per_act": per_act, "per_act_sources": db.per_act_sources(conn),
            "act_spans": {str(a): [round(x, 3) for x in final_spans[a]] for a in acts},
            "n_anchors": n_anchors, "n_pairs": len(pairs),
            "long_take": shots_by_id[long_take_id]["recording_id"] if long_take_id else None,
            "n_scenes": len(scenes), "music_assignment": mode,
            "music_map_recomputed": sorted(moved), "music_map_problems": problems,
            "cold_open_beat": act0[0].get("beat_id") if act0 and act0[0]["kind"] == "video" else None,
            "natural_windows": natural, "n_natural_windows_unplaced": len(windows) - len(natural),
            "quality": quality}


def photo_source(cfg: Config, row: Mapping[str, Any]) -> Path | None:
    """What the renderer reads for a photograph slot.

    The JPEG S03.0 wrote while decoding, when it exists; the original only
    as a fallback. An iPhone still is HEIC and whether ffmpeg opens HEIC is a
    property of its build, not of the file -- the box's Ubuntu ffmpeg 4.4
    refused every one of them, and the draft died on the first.
    """
    still = cfg.work_root / "stills" / f"{row['shot_id']}.jpg"
    if still.exists():
        return still
    key = row.get("s3_key")
    return cfg.data_root / str(key).replace("raw/", "") if key else None


def render_draft(cfg: Config, conn) -> dict[str, Any]:
    """S07 -- render the cut to a watchable file."""
    rows = [dict(r) for r in conn.execute(
        "SELECT t.*, s.recording_id, s.media_kind, a.s3_key FROM timeline t "
        "LEFT JOIN shots s ON s.shot_id = t.shot_id "
        "LEFT JOIN assets a ON a.asset_id = s.asset_id ORDER BY t.slot_index")]
    if not rows:
        return {"skipped": "no timeline"}
    proxies = cfg.work_root / "proxies"
    sources: dict[str, Any] = {}
    usable: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for r in rows:
        if r["kind"] == "card":
            # A card slot has no shot -- the inner join used to drop it
            # (shot_id is NULL), which shortened the draft by exactly its
            # length. It needs no source file, so it is never "missing".
            usable.append(r)
            continue
        if r["media_kind"] == "photo":
            p = photo_source(cfg, r)
        else:
            p = proxies / f"{r['recording_id']}_eq.mp4"
        if p is not None and p.exists():
            sources[r["shot_id"]] = p
            usable.append(r)
        else:
            missing[r["media_kind"] or "?"] = missing.get(r["media_kind"] or "?", 0) + 1
    if not usable:
        return {"skipped": "no source media for any slot"}
    if missing:
        # Loudly: a slot that silently vanishes shortens the film, and the
        # first draft lost 94 photographs -- half its running time -- exactly
        # this way.
        log.warning("S07 %d slot(s) have no source media and are omitted, "
                    "which shortens the cut: %s", sum(missing.values()), missing)

    out = cfg.workdir("gates", "gate3") / "draft.mp4"
    d = cfg.get("render.draft")
    cmd = render_mod.build_command(
        usable, sources=sources, out_path=out,
        width=int(d["width"]), height=int(d["height"]), crf=int(d["crf"]),
        fps=int(cfg.get("render.fps", render_mod.DRAFT_FPS)))
    log.info("S07 rendering %d of %d slot(s) to %s", len(usable), len(rows), out)
    log.debug("S07 %s", render_mod.describe(cmd))
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("S07 render failed: %s", (proc.stderr or "")[-600:])
        return {"error": (proc.stderr or "")[-600:], "n_slots": len(usable)}
    size = out.stat().st_size if out.exists() else 0
    log.info("S07 draft written: %s (%.1f MB)", out, size / 1e6)
    return {"path": str(out), "bytes": size, "n_slots": len(usable),
            "n_skipped": len(rows) - len(usable)}


def run(cfg: Config, *, force: bool = False,
        redo: set[str] | None = None) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal cut --force")
    # All three steps are cheap relative to S03 and they depend on weights the
    # operator is actively tuning, so they always re-run.
    for name, fn in (("score", lambda: score_shots(cfg, conn)),
                     ("timeline", lambda: build_timeline(cfg, conn)),
                     ("draft", lambda: render_draft(cfg, conn))):
        if redo and name not in redo:
            continue
        report[name] = fn()
        db.mark_unit(conn, STAGE, name, detail=json.dumps(report[name], default=str)[:2000])
    report["finished_utc"] = db.utcnow()
    cfg.work("reports", "s05_cut.json").write_text(json.dumps(report, indent=2, default=str))
    conn.close()
    return report
