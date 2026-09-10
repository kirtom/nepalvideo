"""S02.7 / S02.8 -- music analysis, act assignment and the music map.

Feature extraction needs librosa; everything that decides anything is pure
numpy/scipy and testable without it. The assignment is where the film's shape
is actually set, so it is the part that must be inspectable.

The one structural departure from a plain Hungarian solve: the spec requires a
bonus when the Act 5 track shares a key or artist with the Act 1 track, because
the callback is what makes twenty minutes read as a film rather than a
montage. That bonus makes the cost of an Act 5 assignment depend on which
track won Act 1, which a single linear assignment cannot express. With a
library of this size the honest fix is to solve the assignment once per
candidate Act 1 track and keep the best total -- a handful of tiny solves.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler profiles, normalised at use.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                          5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                          4.75, 3.98, 2.69, 3.34, 3.17])

# Qualitative targets from the spec's act table, as z-scores.
LEVEL = {"low": -1.0, "low-mid": -0.5, "mid": 0.0, "mid-high": 0.5,
         "high": 1.0, "highest": 1.5, "any": 0.0}

FEATURES = ("energy_mean", "dyn_range", "centroid", "onset_rate", "tempo_bpm")

ACT_TARGETS: dict[int, dict[str, str]] = {
    1: {"energy_mean": "low",      "dyn_range": "low",  "centroid": "low",  "onset_rate": "low",  "tempo_bpm": "low"},
    2: {"energy_mean": "mid",      "dyn_range": "mid",  "centroid": "mid",  "onset_rate": "mid",  "tempo_bpm": "mid"},
    3: {"energy_mean": "mid-high", "dyn_range": "high", "centroid": "mid",  "onset_rate": "mid",  "tempo_bpm": "mid"},
    4: {"energy_mean": "highest",  "dyn_range": "high", "centroid": "high", "onset_rate": "high", "tempo_bpm": "any"},
    5: {"energy_mean": "low-mid",  "dyn_range": "mid",  "centroid": "low",  "onset_rate": "low",  "tempo_bpm": "low"},
}
# "any" means the dimension carries no weight for that act, not that it is mid.
ACT_WEIGHTS: dict[int, np.ndarray] = {
    a: np.array([0.0 if spec[f] == "any" else 1.0 for f in FEATURES])
    for a, spec in ACT_TARGETS.items()
}


@dataclass
class Track:
    track_id: str
    s3_key: str
    title: str
    duration_s: float = 0.0
    tempo_bpm: float = 0.0
    key_est: str | None = None
    energy_mean: float = 0.0
    energy_p95: float = 0.0
    energy_p10: float = 0.0
    centroid: float = 0.0
    onset_rate: float = 0.0
    artist: str | None = None
    licence: str = "personal"
    sections: list[dict[str, Any]] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    assigned_act: int | None = None

    @property
    def dyn_range(self) -> float:
        return self.energy_p95 - self.energy_p10


def parse_artist_title(filename: str) -> tuple[str | None, str]:
    """'03 - Nils Frahm - Says.mp3' -> ('Nils Frahm', 'Says')."""
    stem = Path(filename).stem
    stem = re.sub(r"^\s*\d{1,3}[\s._-]+", "", stem)          # leading track number
    parts = [p.strip() for p in re.split(r"\s+-\s+|\s+—\s+", stem) if p.strip()]
    if len(parts) >= 2:
        return parts[0], " - ".join(parts[1:])
    return None, stem or filename


def estimate_key(chroma_mean: Sequence[float]) -> str:
    """Krumhansl-Schmuckler correlation against all 24 keys."""
    v = np.asarray(chroma_mean, dtype=float)
    if v.size != 12 or not np.any(v):
        return "unknown"
    v = (v - v.mean()) / (v.std() + 1e-9)
    best, best_r = "unknown", -np.inf
    for mode, profile in (("maj", MAJOR_PROFILE), ("min", MINOR_PROFILE)):
        p = (profile - profile.mean()) / profile.std()
        for shift in range(12):
            r = float(np.corrcoef(v, np.roll(p, shift))[0, 1])
            if r > best_r:
                best, best_r = f"{PITCH_CLASSES[shift]}{mode}", r
    return best


def mark_swells(sections: Sequence[dict[str, Any]], *, percentile: float = 75.0) -> list[dict[str, Any]]:
    """A section is a swell where its energy clears the track's p75 *and* rises
    from the section before it. The second half is what distinguishes a build
    from a passage that is merely loud."""
    out = [dict(s) for s in sections]
    if not out:
        return out
    energies = [float(s.get("energy", 0.0)) for s in out]
    threshold = float(np.percentile(energies, percentile))
    for i, s in enumerate(out):
        rising = i > 0 and energies[i] > energies[i - 1]
        s["is_swell"] = int(energies[i] >= threshold and rising)
    return out


# -- act assignment ----------------------------------------------------

def _matrix(tracks: Sequence[Track]) -> np.ndarray:
    return np.array([[getattr(t, f) if f != "dyn_range" else t.dyn_range
                      for f in FEATURES] for t in tracks], dtype=float)


def znorm(m: np.ndarray) -> np.ndarray:
    """Z-normalise each feature across the library. With one track, or a
    feature that never varies, the column collapses to zero rather than NaN."""
    if m.size == 0:
        return m
    mu = m.mean(axis=0)
    sd = m.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (m - mu) / sd


def target_vector(act: int) -> np.ndarray:
    spec = ACT_TARGETS[act]
    return np.array([LEVEL[spec[f]] for f in FEATURES], dtype=float)


def callback_affinity(act1: Track, candidate: Track) -> float:
    """How strongly ``candidate`` reads as a return to the Act 1 theme.

    Same artist is the strongest signal, same key next. Returned as a positive
    bonus in cost units, subtracted from the Act 5 distance.
    """
    if candidate.track_id == act1.track_id:
        return 0.0
    bonus = 0.0
    if act1.artist and candidate.artist and act1.artist.lower() == candidate.artist.lower():
        bonus += 1.5
    if act1.key_est and candidate.key_est and act1.key_est not in ("unknown", None):
        if act1.key_est == candidate.key_est:
            bonus += 1.0
        elif act1.key_est[:-3] == candidate.key_est[:-3]:   # same tonic, other mode
            bonus += 0.4
    return bonus


@dataclass
class Assignment:
    by_act: dict[int, str]
    cost: float
    callback_bonus: float
    fallback_used: bool = False
    note: str = ""


def assign_acts(tracks: Sequence[Track], *, acts: Sequence[int] = (1, 2, 3, 4, 5),
                license_mode: str = "personal") -> Assignment:
    """Hungarian assignment of tracks to acts, with the Act 5 callback bonus.

    Falls back to reusing tracks when the library is smaller than the number of
    acts, which is a legitimate outcome for a five-act film scored from a
    handful of pieces -- and is surfaced so Gate 1 can show it.
    """
    from scipy.optimize import linear_sum_assignment

    pool = [t for t in tracks
            if license_mode != "cleared" or t.licence == "cleared"]
    if not pool:
        return Assignment({}, math.inf, 0.0, True,
                          f"no tracks available under license_mode={license_mode}")

    if len(pool) < len(acts):
        return _assign_with_reuse(pool, acts, license_mode)

    z = znorm(_matrix(pool))
    other_acts = [a for a in acts if a != 1]

    best: Assignment | None = None
    for i1, cand1 in enumerate(pool):
        d1 = float(np.linalg.norm((z[i1] - target_vector(1)) * ACT_WEIGHTS[1]))
        rest_idx = [i for i in range(len(pool)) if i != i1]
        if len(rest_idx) < len(other_acts):
            continue

        cost = np.zeros((len(other_acts), len(rest_idx)))
        bonus_m = np.zeros_like(cost)
        for r, act in enumerate(other_acts):
            tgt, w = target_vector(act), ACT_WEIGHTS[act]
            for c, i in enumerate(rest_idx):
                d = float(np.linalg.norm((z[i] - tgt) * w))
                b = callback_affinity(cand1, pool[i]) if act == 5 else 0.0
                bonus_m[r, c] = b
                cost[r, c] = d - b

        rows, cols = linear_sum_assignment(cost)
        total = d1 + float(cost[rows, cols].sum())
        bonus = float(sum(bonus_m[r, c] for r, c in zip(rows, cols)))
        by_act = {1: cand1.track_id}
        for r, c in zip(rows, cols):
            by_act[other_acts[r]] = pool[rest_idx[c]].track_id
        if best is None or total < best.cost:
            best = Assignment(by_act, round(total, 4), round(bonus, 4))

    assert best is not None
    for t in tracks:
        t.assigned_act = next((a for a, tid in best.by_act.items() if tid == t.track_id), None)
    log.info("S02.7 act assignment cost=%.3f callback_bonus=%.3f: %s",
             best.cost, best.callback_bonus,
             {a: best.by_act[a] for a in sorted(best.by_act)})
    return best


def _assign_with_reuse(pool: Sequence[Track], acts: Sequence[int],
                       license_mode: str) -> Assignment:
    """Fewer tracks than acts: pick the nearest track per act, allowing reuse,
    and force Act 5 to echo Act 1 where more than one track exists."""
    z = znorm(_matrix(pool))
    by_act: dict[int, str] = {}
    for act in acts:
        tgt, w = target_vector(act), ACT_WEIGHTS[act]
        d = [float(np.linalg.norm((z[i] - tgt) * w)) for i in range(len(pool))]
        by_act[act] = pool[int(np.argmin(d))].track_id
    if 1 in by_act and 5 in acts and len(pool) > 1:
        by_act[5] = by_act[1]
    for t in pool:
        t.assigned_act = next((a for a, tid in by_act.items() if tid == t.track_id), None)
    return Assignment(by_act, math.inf, 0.0, True,
                      f"only {len(pool)} track(s) for {len(acts)} acts -- reused; "
                      f"confirm at Gate 1")


# -- act durations and the music map -----------------------------------

def allocate_act_durations(act_specs: Sequence[dict[str, Any]],
                           total_s: float) -> dict[int, float]:
    """Split the runtime across acts, respecting each act's min/max.

    Starts from the midpoint of each act's range and distributes the shortfall
    or surplus proportionally to the slack available in the right direction, so
    no act is pushed outside the band the creative brief set for it.
    """
    acts = [int(a["act"]) for a in act_specs]
    lo = {int(a["act"]): float(a["min_s"]) for a in act_specs}
    hi = {int(a["act"]): float(a["max_s"]) for a in act_specs}
    cur = {a: (lo[a] + hi[a]) / 2 for a in acts}

    for _ in range(64):
        delta = total_s - sum(cur.values())
        if abs(delta) < 1e-6:
            break
        slack = {a: (hi[a] - cur[a]) if delta > 0 else (cur[a] - lo[a]) for a in acts}
        pool = sum(slack.values())
        if pool <= 1e-9:
            break                      # cannot reach the target inside the bands
        for a in acts:
            cur[a] += delta * slack[a] / pool
        for a in acts:
            cur[a] = min(hi[a], max(lo[a], cur[a]))
    return {a: round(cur[a], 3) for a in acts}


def build_music_map(tracks: Sequence[Track], assignment: Assignment,
                    act_specs: Sequence[dict[str, Any]], *, total_s: float,
                    silence_s: float = 3.0) -> dict[str, Any]:
    """Emit ``work/music/music_map.json`` (spec S02.8).

    ``silence_window`` is the mandatory hard cut to silence after the Act 4
    peak -- the single most powerful move in the film, per the brief -- so it
    is placed at the end of Act 4 rather than left to the assembler.
    """
    by_id = {t.track_id: t for t in tracks}
    durations = allocate_act_durations(act_specs, total_s)

    acts_out: list[dict[str, Any]] = []
    t_cursor = 0.0
    silence: dict[str, float] | None = None

    for spec in sorted(act_specs, key=lambda s: int(s["act"])):
        act = int(spec["act"])
        dur = durations[act]
        track = by_id.get(assignment.by_act.get(act, ""))
        t_start, t_end = t_cursor, t_cursor + dur

        swells: list[float] = []
        beat_grid: list[float] = []
        downbeats: list[float] = []
        if track:
            swells = [round(t_start + float(s["start_s"]), 3)
                      for s in track.sections
                      if s.get("is_swell") and float(s["start_s"]) <= dur]
            beat_grid = [round(t_start + b, 3) for b in track.beats if b <= dur]
            downbeats = [round(t_start + b, 3) for b in track.downbeats if b <= dur]
            if not swells and track.sections:
                # every act needs at least one swell timestamp (acceptance
                # criterion); fall back to the loudest section inside the act
                inside = [s for s in track.sections if float(s["start_s"]) <= dur]
                if inside:
                    peak = max(inside, key=lambda s: float(s.get("energy", 0.0)))
                    swells = [round(t_start + float(peak["start_s"]), 3)]

        acts_out.append({
            "act": act,
            "name": spec.get("name"),
            "track_id": track.track_id if track else None,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "swells": swells,
            "beat_grid": beat_grid,
            "downbeats": downbeats,
        })
        if act == 4:
            silence = {"t_start": round(t_end, 3), "t_end": round(t_end + silence_s, 3)}
        t_cursor = t_end

    return {
        "total_duration_s": round(t_cursor, 3),
        "acts": acts_out,
        "silence_window": silence or {},
        "assignment_cost": assignment.cost if assignment.cost != math.inf else None,
        "callback_bonus": assignment.callback_bonus,
        "assignment_note": assignment.note,
    }


def check_music_map(mmap: dict[str, Any], *, target_s: float,
                    tolerance_s: float = 30.0) -> list[str]:
    """Section S02.8 acceptance: act durations sum to within +/-30 s of target,
    and every act has at least one swell timestamp."""
    problems: list[str] = []
    total = float(mmap.get("total_duration_s", 0.0))
    if abs(total - target_s) > tolerance_s:
        problems.append(f"act durations sum to {total:.0f}s, "
                        f"outside {target_s:.0f}+/-{tolerance_s:.0f}s")
    for a in mmap.get("acts", []):
        if not a.get("swells"):
            problems.append(f"act {a['act']} has no swell timestamp")
        if not a.get("track_id"):
            problems.append(f"act {a['act']} has no track assigned")
    if not mmap.get("silence_window"):
        problems.append("no silence_window after the Act 4 peak")
    return problems


# -- feature extraction (needs librosa) --------------------------------

def analyse_track(path: str | Path, *, n_sections: int = 8,
                  licence: str = "personal") -> Track:
    import librosa

    p = Path(path)
    y, sr = librosa.load(str(p), mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]
    rms = librosa.feature.rms(y=y)[0]
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    artist, title = parse_artist_title(p.name)
    track = Track(
        track_id=p.stem, s3_key=f"raw/music/{p.name}", title=title, artist=artist,
        duration_s=round(duration, 3),
        tempo_bpm=round(float(np.atleast_1d(tempo)[0]), 3),
        key_est=estimate_key(chroma.mean(axis=1)),
        energy_mean=float(rms.mean()),
        energy_p95=float(np.percentile(rms, 95)),
        energy_p10=float(np.percentile(rms, 10)),
        centroid=centroid,
        onset_rate=float((onset > onset.mean()).sum()) / max(duration, 1e-6),
        licence=licence,
        beats=beats,
        # 4/4 is assumed: Telegram-era post-rock and solo piano rarely are not,
        # and a wrong downbeat costs an act transition landing a beat early.
        downbeats=beats[::4],
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    k = max(2, min(n_sections, mfcc.shape[1] // 4)) if mfcc.shape[1] else 2
    bounds = librosa.segment.agglomerative(mfcc, k)
    times = list(librosa.frames_to_time(bounds, sr=sr)) + [duration]
    sections = []
    for i, (a, b) in enumerate(zip(times, times[1:])):
        lo = int(a / duration * len(rms)) if duration else 0
        hi = max(lo + 1, int(b / duration * len(rms)) if duration else 1)
        sections.append({
            "section_id": f"{track.track_id}_s{i:02d}",
            "track_id": track.track_id,
            "start_s": round(float(a), 3), "end_s": round(float(b), 3),
            "energy": float(rms[lo:hi].mean()) if lo < len(rms) else 0.0,
        })
    track.sections = mark_swells(sections)
    return track


def detect_licence(path: Path, licences: dict[str, str] | None = None) -> str:
    """A track is cleared if listed in music/licences.json or filed under a
    directory named 'cleared'."""
    if licences and path.name in licences:
        return licences[path.name]
    return "cleared" if "cleared" in {p.lower() for p in path.parts} else "personal"
