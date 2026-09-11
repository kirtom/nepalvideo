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
import collections
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

# Below this share of total energy in the 300-3400 Hz formant band, the
# syllabic-modulation ratio is measured over noise and means nothing.
MIN_FORMANT_BAND_SHARE = 0.10

# Plain-language description of each act's target, for reports the operator reads.
ACT_CHARACTER = {
    1: "sparse and quiet -- solo piano, domestic, wistful",
    2: "mid-energy and warm -- strings arriving over piano",
    3: "building, with wide dynamic range -- slow post-rock swell or cold orchestral",
    4: "the loudest and brightest thing you have -- it peaks then cuts to silence",
    5: "quiet again, ideally the same artist or key as Act 1 -- the callback",
}

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
    vocal_score: float = 0.0
    artist: str | None = None
    licence: str = "personal"
    sections: list[dict[str, Any]] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    assigned_act: int | None = None

    @property
    def dyn_range(self) -> float:
        return self.energy_p95 - self.energy_p10


def read_tags(path: str | Path) -> tuple[str | None, str | None]:
    """(artist, title) from the file's own metadata, or (None, None).

    Tags beat filenames wherever they exist. The Act 5 callback bonus matches on
    artist, so "03 - some_export_name.mp3" silently costs the film its callback
    while an ID3 frame gets it right. mutagen is optional; without it, or on a
    file carrying no tags, the filename parse still applies.
    """
    try:
        import mutagen
    except ImportError:
        return None, None
    try:
        f = mutagen.File(str(path), easy=True)
    except Exception:                                  # noqa: BLE001 - malformed tags
        return None, None
    if not f:
        return None, None

    def first(*keys: str) -> str | None:
        for k in keys:
            v = f.get(k)
            if v:
                text = str(v[0] if isinstance(v, list) else v).strip()
                if text:
                    return text
        return None

    return first("artist", "albumartist", "performer"), first("title")


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


def vocal_likelihood(y: "np.ndarray", sr: int, *, tempo_bpm: float = 0.0) -> float:
    """0..1 estimate of sung or spoken vocal presence in a track.

    A playlist export hands you Spotify's ``instrumentalness``; an mp3 does not,
    so it has to be measured. This matters here specifically: the narration is
    the Russian round video messages, and sung vocals compete with spoken voice
    in the same frequency range, so a vocal-heavy score fights the thing it is
    supposed to support.

    The measure is syllabic modulation, the standard speech/music discriminator.
    Voice modulates its amplitude at roughly 3-8 Hz -- the rate at which
    syllables arrive -- within the formant band around 300-3400 Hz. Instrumental
    music modulates at the beat rate, typically 1-3 Hz, or holds steady. So the
    harmonic component is isolated (to keep drums out of the envelope), band-
    limited to the formant range, and the energy of its amplitude envelope in
    3-8 Hz is compared against the envelope's total.

    The ratio is only meaningful when there is energy in the formant band to
    measure. A bass-heavy track leaves that band nearly empty, so the envelope
    there is noise and the 3-8 Hz fraction comes out arbitrary -- measured at
    0.744 for a two-tone pad at 220 and 330 Hz, both below the band floor, which
    is indistinguishable from a sung vocal. So the band's share of total energy
    is checked first, and a track with almost nothing there scores 0: it has no
    vocal energy, which is the right answer for the wrong-looking reason.

    Measured on synthetic signals, after both corrections:

        sung/spoken vocal            1.00
        fast arpeggio (4 Hz notes)   0.55     <- the closest false positive
        plucked figure (1-2 Hz)      0.39-0.42
        sustained pad, slow swell    0.00

    So roughly 0.6 separates voice from rhythmically articulated instrumental.
    It is a heuristic, not a classifier -- a solo violin with heavy vibrato
    scores higher than it should, a whispered vocal lower, and a fast arpeggio
    sits uncomfortably close to the line. It is reported as a number for the
    operator to sanity-check and defaults to filtering nothing.
    """
    import librosa

    if y is None or len(y) == 0:
        return 0.0
    # Drums dominate a raw envelope and modulate at the beat rate, which would
    # swamp the syllabic band.
    try:
        harmonic = librosa.effects.harmonic(y, margin=2.0)
    except Exception:                                  # noqa: BLE001
        harmonic = y

    hop = 512
    fmax = min(3400, sr // 2 - 1)
    mel = librosa.feature.melspectrogram(y=harmonic, sr=sr, n_mels=40,
                                         fmin=300, fmax=fmax, hop_length=hop)
    full = librosa.feature.melspectrogram(y=harmonic, sr=sr, n_mels=64,
                                          fmin=20, fmax=sr // 2 - 1, hop_length=hop)
    band_share = float(mel.sum()) / (float(full.sum()) + 1e-12)
    if band_share < MIN_FORMANT_BAND_SHARE:
        return 0.0

    env = mel.sum(axis=0)
    if env.size < 16:
        return 0.0
    level = float(env.mean())
    if level <= 0:
        return 0.0

    # Modulation DEPTH, not a ratio between modulation bands. A ratio asks
    # "of whatever modulation exists, how much is syllabic" -- and when a track
    # has no real modulation at all, that question is answered by noise and
    # lands anywhere: a steady two-tone pad measured 0.744, indistinguishable
    # from a sung vocal. Depth asks the physical question instead: by how much
    # does the amplitude actually swing at syllable rate, relative to its own
    # level. A steady pad swings by nothing, whatever its spectrum looks like.
    frame_rate = sr / hop
    window = np.hanning(env.size)
    spec = np.abs(np.fft.rfft((env - level) * window)) / (env.size / 2)
    freqs = np.fft.rfftfreq(env.size, d=1.0 / frame_rate)

    band = (freqs >= 3.0) & (freqs <= 8.0)
    syllabic = spec[band].copy()
    if syllabic.size == 0:
        return 0.0

    # Rhythmic articulation is not speech. A plucked guitar or a piano figure
    # modulates at harmonics of its note rate, which for anything above about
    # 90 BPM lands inside the syllabic band -- measured as a full 1.0 for a
    # 2 Hz gated tone, identical to a sung vocal. Syllables are irregular;
    # articulation is phase-locked to the beat. So energy sitting on beat
    # harmonics is discounted, using the tempo already measured for the track.
    if tempo_bpm and tempo_bpm > 0:
        beat_hz = tempo_bpm / 60.0
        band_freqs = freqs[band]
        resolution = max(float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.1, 0.15)
        for harmonic in range(1, 12):
            centre = beat_hz * harmonic
            if centre > 8.5:
                break
            near = np.abs(band_freqs - centre) <= resolution * 2
            syllabic[near] *= 0.25

    depth = float(np.sqrt((syllabic ** 2).sum())) / level
    # A fully modulated tone swings between silence and peak; its syllabic
    # component sits near 0.35 of the mean level, so that is taken as 1.0.
    return float(min(1.0, max(0.0, depth / 0.35)))


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
                license_mode: str = "personal",
                feature_mask: Sequence[float] | None = None) -> Assignment:
    """Hungarian assignment of tracks to acts, with the Act 5 callback bonus.

    ``feature_mask`` zeroes dimensions that the feature source cannot supply --
    dynamic range is unavailable from a playlist export, and a masked dimension
    must contribute nothing rather than contribute a fabricated value. The
    remaining dimensions are rescaled so distances stay comparable to a
    full-feature run.

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

    mask = _resolve_mask(feature_mask)

    if len(pool) < len(acts):
        return _assign_with_reuse(pool, acts, license_mode, mask)

    z = znorm(_matrix(pool))
    other_acts = [a for a in acts if a != 1]

    best: Assignment | None = None
    for i1, cand1 in enumerate(pool):
        d1 = float(np.linalg.norm((z[i1] - target_vector(1)) * ACT_WEIGHTS[1] * mask))
        rest_idx = [i for i in range(len(pool)) if i != i1]
        if len(rest_idx) < len(other_acts):
            continue

        cost = np.zeros((len(other_acts), len(rest_idx)))
        bonus_m = np.zeros_like(cost)
        for r, act in enumerate(other_acts):
            tgt, w = target_vector(act), ACT_WEIGHTS[act] * mask
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


def _resolve_mask(feature_mask: Sequence[float] | None) -> np.ndarray:
    """Normalise a feature mask so masking a dimension does not simply shrink
    every distance -- the surviving dimensions carry the full weight."""
    if feature_mask is None:
        return np.ones(len(FEATURES))
    m = np.asarray(feature_mask, dtype=float)
    if m.shape != (len(FEATURES),):
        raise ValueError(f"feature_mask must have {len(FEATURES)} entries")
    live = float(m.sum())
    if live <= 0:
        raise ValueError("feature_mask masks every dimension")
    return m * (len(FEATURES) / live)


def _assign_with_reuse(pool: Sequence[Track], acts: Sequence[int],
                       license_mode: str,
                       mask: np.ndarray | None = None) -> Assignment:
    """Fewer tracks than acts: pick the nearest track per act, allowing reuse,
    and force Act 5 to echo Act 1 where more than one track exists."""
    z = znorm(_matrix(pool))
    mask = np.ones(len(FEATURES)) if mask is None else mask
    by_act: dict[int, str] = {}
    for act in acts:
        tgt, w = target_vector(act), ACT_WEIGHTS[act] * mask
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


def rank_for_act(tracks: Sequence[Track], act: int,
                 feature_mask: Sequence[float] | None = None) -> list[Track]:
    """Tracks ordered by how well they fit one act's target."""
    if not tracks:
        return []
    mask = _resolve_mask(feature_mask)
    z = znorm(_matrix(tracks))
    tgt, w = target_vector(act), ACT_WEIGHTS[act] * mask
    scored = [(float(np.linalg.norm((z[i] - tgt) * w)), t) for i, t in enumerate(tracks)]
    return [t for _, t in sorted(scored, key=lambda kv: kv[0])]


def fill_act(act: int, primary: Track | None, pool: Sequence[Track], duration_s: float,
             *, already_used: set[str] | None = None,
             feature_mask: Sequence[float] | None = None,
             min_segment_s: float = 20.0) -> list[dict[str, Any]]:
    """Lay tracks end to end until the act's runtime is covered.

    One track per act does not cover a 20-minute film. Allocated durations are
    167/287/407/113/227 s, and a typical song is around 210 s, so Act 3 runs
    197 s -- nearly half its length -- past the end of its track. Since S06
    snaps every cut to the beat grid, an act with no grid in its second half
    cannot be cut on the beat there at all.

    The primary track (from the Hungarian assignment) opens the act; the
    remainder is filled with the next best-fitting tracks for the same act
    target, preferring ones not used elsewhere so the film does not repeat
    itself. Reuse is allowed when the library runs out, because a repeated
    track is better than an act with no music.
    """
    used = set(already_used or ())
    segments: list[dict[str, Any]] = []
    cursor = 0.0

    ordered: list[Track] = []
    if primary is not None:
        ordered.append(primary)
    for t in rank_for_act([t for t in pool if t is not primary], act, feature_mask):
        ordered.append(t)

    fresh = [t for t in ordered if t.track_id not in used or t is primary]
    recycled = [t for t in ordered if t not in fresh]

    for candidate in fresh + recycled + ordered:      # ordered again = unrestricted reuse
        if cursor >= duration_s - 1e-6:
            break
        remaining = duration_s - cursor
        if remaining < min_segment_s and segments:
            # stretch the last segment rather than leaving a stub
            segments[-1]["t_end"] = round(duration_s, 3)
            segments[-1]["src_out"] = round(
                segments[-1]["src_in"] + (duration_s - segments[-1]["t_in"]), 3)
            cursor = duration_s
            break
        take = min(candidate.duration_s or remaining, remaining)
        if take <= 0:
            continue
        segments.append({
            "track_id": candidate.track_id,
            "title": candidate.title,
            "artist": candidate.artist,
            "t_in": round(cursor, 3),
            "t_end": round(cursor + take, 3),
            "src_in": 0.0,
            "src_out": round(take, 3),
        })
        used.add(candidate.track_id)
        cursor += take

    if segments and cursor < duration_s - 1e-6:
        # nothing left to add: extend the final segment and let the report say so
        segments[-1]["t_end"] = round(duration_s, 3)
        segments[-1]["src_out"] = round(
            segments[-1]["src_in"] + (duration_s - segments[-1]["t_in"]), 3)
    return segments


def build_music_map(tracks: Sequence[Track], assignment: Assignment,
                    act_specs: Sequence[dict[str, Any]], *, total_s: float,
                    silence_s: float = 3.0,
                    feature_mask: Sequence[float] | None = None) -> dict[str, Any]:
    """Emit ``work/music/music_map.json`` (spec S02.8).

    Each act carries a sequence of track segments rather than a single track,
    because acts are longer than songs. Beat grids and swells are merged across
    the segments and expressed on the film timeline, so S06 has a continuous
    grid to snap cuts to for the whole act.

    ``silence_window`` is the mandatory hard cut to silence after the Act 4
    peak -- the single most powerful move in the film, per the brief -- so it is
    placed at the end of Act 4 rather than left to the assembler.
    """
    by_id = {t.track_id: t for t in tracks}
    durations = allocate_act_durations(act_specs, total_s)

    acts_out: list[dict[str, Any]] = []
    t_cursor = 0.0
    silence: dict[str, float] | None = None
    used: set[str] = set()

    for spec in sorted(act_specs, key=lambda s: int(s["act"])):
        act = int(spec["act"])
        dur = durations[act]
        primary = by_id.get(assignment.by_act.get(act, ""))
        t_start, t_end = t_cursor, t_cursor + dur

        segments = fill_act(act, primary, tracks, dur, already_used=used,
                            feature_mask=feature_mask)
        for seg in segments:
            used.add(seg["track_id"])

        swells: list[float] = []
        beat_grid: list[float] = []
        downbeats: list[float] = []
        for seg in segments:
            t = by_id.get(seg["track_id"])
            if not t:
                continue
            seg_len = seg["t_end"] - seg["t_in"]
            base = t_start + seg["t_in"] - seg["src_in"]
            swells += [round(base + float(x["start_s"]), 3) for x in t.sections
                       if x.get("is_swell")
                       and seg["src_in"] <= float(x["start_s"]) < seg["src_in"] + seg_len]
            beat_grid += [round(base + b, 3) for b in t.beats
                          if seg["src_in"] <= b < seg["src_in"] + seg_len]
            downbeats += [round(base + b, 3) for b in t.downbeats
                          if seg["src_in"] <= b < seg["src_in"] + seg_len]

        if not swells and segments:
            # every act needs at least one swell timestamp (acceptance criterion);
            # fall back to the loudest section of the opening track
            t = by_id.get(segments[0]["track_id"])
            inside = [x for x in (t.sections if t else []) if float(x["start_s"]) <= dur]
            if inside:
                peak = max(inside, key=lambda x: float(x.get("energy", 0.0)))
                swells = [round(t_start + float(peak["start_s"]), 3)]

        covered = segments[-1]["t_end"] if segments else 0.0
        grid_end = max(beat_grid) - t_start if beat_grid else 0.0
        plays = collections.Counter(seg["track_id"] for seg in segments)
        max_repeats = max(plays.values()) if plays else 0
        acts_out.append({
            "act": act,
            "name": spec.get("name"),
            "track_id": segments[0]["track_id"] if segments else None,
            "segments": segments,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "music_covered_s": round(covered, 3),
            "beat_grid_covers_s": round(grid_end, 3),
            "n_distinct_tracks": len(plays),
            "max_track_repeats": max_repeats,
            "swells": sorted(swells),
            "beat_grid": sorted(beat_grid),
            "downbeats": sorted(downbeats),
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
        "n_tracks_available": len(tracks),
        "n_tracks_used": len(used),
        "library_duration_s": round(sum(t.duration_s or 0.0 for t in tracks), 1),
    }


def check_music_map(mmap: dict[str, Any], *, target_s: float,
                    tolerance_s: float = 30.0,
                    max_repeats_per_act: int = 2,
                    min_headroom: float = 2.0) -> list[str]:
    """Section S02.8 acceptance, plus two checks the spec does not state.

    The spec's criteria are that act durations sum to within +/-30 s of target
    and that every act carries a swell. Two more are needed in practice: that
    the beat grid actually spans each act, since S06 snaps cuts to it, and that
    no single track is laid down so many times that the audience hears a loop.
    """
    problems: list[str] = []
    total = float(mmap.get("total_duration_s", 0.0))
    if abs(total - target_s) > tolerance_s:
        problems.append(f"act durations sum to {total:.0f}s, "
                        f"outside {target_s:.0f}+/-{tolerance_s:.0f}s")

    # The clean top-level question: is there enough music for the film at all?
    # Below 1x the film's length, repetition is arithmetically unavoidable.
    # Below the headroom factor there is music enough but little choice, so the
    # act assignment is forced rather than selective.
    library = float(mmap.get("library_duration_s", 0.0))
    if library and total:
        if library < total:
            problems.append(
                f"the library holds {library/60:.0f} min of music for a "
                f"{total/60:.0f} min film -- tracks must repeat. Add at least "
                f"{(total - library)/60:.0f} more minutes.")
        elif library < total * min_headroom:
            problems.append(
                f"the library holds {library/60:.0f} min for a {total/60:.0f} min "
                f"film ({library/total:.1f}x). Below {min_headroom:.1f}x the act "
                f"assignment has little to choose from, so acts get whatever is "
                f"nearest rather than what fits.")
    for a in mmap.get("acts", []):
        if not a.get("swells"):
            problems.append(f"act {a['act']} has no swell timestamp")
        if not a.get("track_id"):
            problems.append(f"act {a['act']} has no track assigned")
        # S06 snaps every cut to the beat grid, so an act whose grid stops early
        # cannot be cut on the beat past that point. With track reuse this
        # rarely fires, which is why repetition is checked as well.
        act_len = float(a.get("t_end", 0)) - float(a.get("t_start", 0))
        grid = float(a.get("beat_grid_covers_s", 0))
        if act_len > 0 and grid < act_len * 0.9:
            problems.append(
                f"act {a['act']}: beat grid covers {grid:.0f}s of {act_len:.0f}s "
                f"({grid/act_len*100:.0f}%) -- cuts past that cannot be beat-snapped")
        # A track laid down several times within one act is a loop the audience
        # will hear, whatever the library's total length.
        repeats = int(a.get("max_track_repeats", 0))
        if repeats > max_repeats_per_act:
            problems.append(
                f"act {a['act']}: one track is laid down {repeats} times to fill "
                f"{act_len:.0f}s, which will be heard as a loop. Act {a['act']} wants "
                + ACT_CHARACTER.get(int(a["act"]), "material"))
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

    # Tags first, filename as the fallback.
    tag_artist, tag_title = read_tags(p)
    name_artist, name_title = parse_artist_title(p.name)
    artist = tag_artist or name_artist
    title = tag_title or name_title
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
        vocal_score=round(vocal_likelihood(
            y, sr, tempo_bpm=float(np.atleast_1d(tempo)[0])), 4),
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
