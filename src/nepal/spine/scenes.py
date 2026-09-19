"""Film v2 step 4 -- scenes and their music targets (spec section 6, task 6).

A scene is the unit S07 hands a track section to: long enough that a cue can
establish itself, short enough that one music target still describes all of
it. Grouping happens before targeting on purpose -- the target formula only
has to read a handful of numbers off something already contiguous, not
re-derive "contiguous" itself from raw per-slot noise.

Pure, like longtake.py and anchors.py: ``attrs`` is the only way this module
touches a slot's measured world, and it is a plain callable the caller
supplies (S05 backs it with the real lookup; a test backs it with a dict).
No database, no ffmpeg, no config reads -- every tunable is a keyword
argument.

``music.scene_targets`` (in config/pipeline.yaml) weights the *distance*
between a scene's target and a candidate track's own features; that
matching is S07's job. This module only says what a stretch of film is
asking for, before anything is compared against it.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from nepal.spine.effort import STEEP_GAIN_M_PER_H

_Attrs = Callable[[Mapping[str, Any]], Mapping[str, Any]]

# -- activity --------------------------------------------------------------

SUMMIT, CITY, TRANSPORT, RESTING = "summit", "city", "transport", "resting"
CROSSING, VILLAGE, CLIMBING = "crossing", "village", "climbing"
DESCENDING, WALKING = "descending", "walking"

# A trailhead city can sit well under a Himalayan pass without ever being
# geocoded as one of the named `cities` -- Kathmandu itself is ~1,400 m, so
# anything this low is almost certainly a city, GPS noise aside.
CITY_ALT_M = 1500.0
# Above walking pace: a jeep or bus, not footage of the trek on foot.
TRANSPORT_SPEED_MS = 6.0
# Below this the party is standing still for the shot, whatever GPS jitter
# would say about a single fix -- a looser bar than effort.py's STOPPED_MS
# is not needed here because a slot already spans several seconds of footage.
RESTING_SPEED_MS = 0.3
# A place name is known but the party is barely moving through it -- a
# tea-house stop, not a march.
VILLAGE_SPEED_MS = 1.0
# Half of effort.py's STEEP_GAIN_M_PER_H (300): that constant flags the
# single hardest hour of the whole trek, but a scene only needs to say the
# trail is visibly tilting -- pinned to the harder bar would leave most of
# the ascent classified as flat walking.
CLIMB_GAIN_M_PER_H = 150.0
# Case-insensitive; casefold (not lower) so the Cyrillic form matches
# regardless of case, the same reasoning as longtake.py's _has_keyword.
CROSSING_KEYWORDS = ("bridge", "мост")


def _num(a: Mapping[str, Any], key: str) -> float | None:
    v = a.get(key)
    return None if v is None else float(v)


def activity_class(a: Mapping[str, Any], *, cities: Sequence[str]) -> str:
    """What kind of moment this slot is, first match wins.

    The order encodes which fact overrides which other: act 4 alone makes
    it the summit push before speed says anything; a stopped shot in a city
    is still a city, not "resting". A ``None`` reading never matches a
    numeric test -- a slot with no speed sample is not thereby "resting".
    """
    speed = _num(a, "speed_ms")
    gain = _num(a, "gain_m_per_h")
    alt = _num(a, "alt_m")
    place = a.get("place_name")

    if a.get("act") == 4:
        return SUMMIT
    if a.get("act") in (1, 5) and (
            (place and any(place.casefold() == c.casefold() for c in cities))
            or (alt is not None and alt < CITY_ALT_M)):
        return CITY
    if speed is not None and speed > TRANSPORT_SPEED_MS:
        return TRANSPORT
    if speed is not None and speed < RESTING_SPEED_MS:
        return RESTING
    if place and any(kw.casefold() in place.casefold() for kw in CROSSING_KEYWORDS):
        return CROSSING
    if place and speed is not None and speed < VILLAGE_SPEED_MS:
        return VILLAGE
    if gain is not None and gain > CLIMB_GAIN_M_PER_H:
        return CLIMBING
    if gain is not None and gain < -CLIMB_GAIN_M_PER_H:
        return DESCENDING
    return WALKING


# -- grouping bands (speed/HR closeness, not the activity itself) ----------

# The same 0.3/1.0/6 cuts activity_class uses for resting/village/transport,
# plus one extra (2.0) between an ordinary walking pace and a brisk one --
# activity_class has no use for that distinction, but two "walking" slots at
# very different paces should still not read as one scene.
SPEED_BAND_CUTS_MS = (RESTING_SPEED_MS, VILLAGE_SPEED_MS, 2.0, TRANSPORT_SPEED_MS)
# Roughly one heart-rate training zone -- two slots 20 bpm apart are working
# at a different intensity even when the trail and the pace look identical.
HR_BAND_WIDTH_BPM = 20.0


def _speed_band(speed_ms: float | None) -> int | None:
    return None if speed_ms is None else bisect.bisect_right(SPEED_BAND_CUTS_MS, speed_ms)


def _hr_band(hr_bpm: float | None) -> int | None:
    return None if hr_bpm is None else int(hr_bpm // HR_BAND_WIDTH_BPM)


# -- Scene -------------------------------------------------------------

@dataclass(frozen=True)
class Scene:
    """A contiguous stretch of the film with one music target.

    ``gain_m_per_h`` is not something S07 reads off a scene directly, but
    ``scene_target``'s effort term needs it as the fallback for scenes with
    no heart-rate sample, so it is carried the same duration-weighted way
    as the other numeric attributes rather than re-derived later.
    """
    scene_id: int
    act: int
    t_in: float
    t_out: float
    slot_indices: tuple[int, ...]
    activity: str
    speed_ms: float | None
    hr_bpm: float | None
    gain_m_per_h: float | None
    alt_m: float | None
    hour: float | None
    voice_share: float
    levity: bool
    picture_energy: float | None


def _slot_position(slot: Mapping[str, Any], position: int) -> int:
    """The slot's own ``slot_index`` when it has one, else its position in
    the list this call was handed."""
    si = slot.get("slot_index")
    return position if si is None else int(si)


def _span(group: Mapping[str, Any], slots: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    positions = group["positions"]
    return slots[positions[0]]["t_in"], slots[positions[-1]]["t_out"]


def _duration(group: Mapping[str, Any], slots: Sequence[Mapping[str, Any]]) -> float:
    t_in, t_out = _span(group, slots)
    return t_out - t_in


def _merge_pair(groups: list[dict], slots: Sequence[Mapping[str, Any]],
                lo: int, hi: int) -> list[dict]:
    """Merge adjacent groups[lo] and groups[hi], keeping the longer part's
    activity -- a short village stop folded into a long walk should not
    relabel the whole scene "village", and vice versa."""
    left, right = groups[lo], groups[hi]
    winner = left if _duration(left, slots) >= _duration(right, slots) else right
    merged = {"positions": left["positions"] + right["positions"],
             "act": left["act"], "activity": winner["activity"]}
    return groups[:lo] + [merged] + groups[hi + 1:]


def _merge_short_scenes(groups: list[dict], slots: Sequence[Mapping[str, Any]], *,
                        min_scene_s: float) -> list[dict]:
    i = 0
    while i < len(groups):
        if _duration(groups[i], slots) >= min_scene_s:
            i += 1
            continue
        act = groups[i]["act"]
        candidates = []                     # (duration, neighbour_index)
        if i > 0 and groups[i - 1]["act"] == act:
            candidates.append((_duration(groups[i - 1], slots), i - 1))
        if i + 1 < len(groups) and groups[i + 1]["act"] == act:
            candidates.append((_duration(groups[i + 1], slots), i + 1))
        if not candidates:
            i += 1                          # no same-act neighbour -- keep as is
            continue
        _, j = min(candidates)
        groups = _merge_pair(groups, slots, min(i, j), max(i, j))
        i = min(i, j)                        # re-examine the merged scene in place
    return groups


def _inside_any_span(t: float, beats_spans: Sequence[tuple[float, float]]) -> bool:
    return any(lo < t < hi for lo, hi in beats_spans)


def _split_one(group: dict, slots: Sequence[Mapping[str, Any]], *,
               max_scene_s: float, beats_spans: Sequence[tuple[float, float]]) -> list[dict]:
    positions = group["positions"]
    if _duration(group, slots) <= max_scene_s or len(positions) < 2:
        return [group]
    t_in, t_out = _span(group, slots)
    mid = (t_in + t_out) / 2.0
    boundaries = []                          # (distance to mid, split point)
    for k in range(1, len(positions)):
        boundary_t = slots[positions[k]]["t_in"]
        if _inside_any_span(boundary_t, beats_spans):
            continue
        boundaries.append((abs(boundary_t - mid), k))
    if not boundaries:
        return [group]                       # no eligible boundary -- leave it long
    _, k = min(boundaries)
    left = {"positions": positions[:k], "act": group["act"], "activity": group["activity"]}
    right = {"positions": positions[k:], "act": group["act"], "activity": group["activity"]}
    return (_split_one(left, slots, max_scene_s=max_scene_s, beats_spans=beats_spans) +
           _split_one(right, slots, max_scene_s=max_scene_s, beats_spans=beats_spans))


def _weighted_mean(pairs: Sequence[tuple[float | None, float]]) -> float | None:
    total_w = sum(w for v, w in pairs if v is not None)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in pairs if v is not None) / total_w


def _build_scene(scene_id: int, group: dict, slots: Sequence[Mapping[str, Any]],
                 attrs: _Attrs) -> Scene:
    positions = group["positions"]
    t_in, t_out = _span(group, slots)
    member_slots = [slots[p] for p in positions]
    member_attrs = [attrs(s) for s in member_slots]
    durations = [s["t_out"] - s["t_in"] for s in member_slots]

    def agg(key: str) -> float | None:
        return _weighted_mean([(_num(a, key), d) for a, d in zip(member_attrs, durations)])

    total_duration = sum(durations)
    voice_duration = sum(d for a, d in zip(member_attrs, durations) if a.get("under_speech"))
    voice_share = voice_duration / total_duration if total_duration > 0 else 0.0
    levity = any(a.get("levity") for a in member_attrs)
    slot_indices = tuple(_slot_position(s, p) for s, p in zip(member_slots, positions))

    return Scene(scene_id=scene_id, act=group["act"], t_in=t_in, t_out=t_out,
                slot_indices=slot_indices, activity=group["activity"],
                speed_ms=agg("speed_ms"), hr_bpm=agg("hr_bpm"),
                gain_m_per_h=agg("gain_m_per_h"), alt_m=agg("alt_m"), hour=agg("hour"),
                voice_share=voice_share, levity=levity, picture_energy=agg("motion_mag"))


def group_scenes(slots: Sequence[Mapping[str, Any]], attrs: _Attrs, *,
                 min_scene_s: float, max_scene_s: float,
                 beats_spans: Sequence[tuple[float, float]],
                 cities: Sequence[str]) -> list[Scene]:
    """Slots into scenes: consecutive runs of matching act/activity/speed-band
    /HR-band, folded up to a floor and cut down to a ceiling.

    ``cities`` is not part of the brief's signature for this function, only
    for ``activity_class`` -- but grouping has to know each slot's activity
    to compare consecutive ones, and ``activity_class`` cannot answer that
    without the city list, so it is threaded through here as well.
    """
    if not slots:
        return []

    groups: list[dict] = []
    for i, slot in enumerate(slots):
        a = attrs(slot)
        act = slot["act"]
        # The slot is authoritative for act, not attrs()'s own copy of it --
        # a missing/stale act in attrs must not silently disable summit/city
        # classification.
        activity = activity_class({**a, "act": act}, cities=cities)
        band = (_speed_band(_num(a, "speed_ms")), _hr_band(_num(a, "hr_bpm")))
        if groups and groups[-1]["act"] == act and groups[-1]["activity"] == activity \
                and groups[-1]["band"] == band:
            groups[-1]["positions"].append(i)
        else:
            groups.append({"positions": [i], "act": act, "activity": activity, "band": band})

    groups = _merge_short_scenes(groups, slots, min_scene_s=min_scene_s)
    split: list[dict] = []
    for g in groups:
        split.extend(_split_one(g, slots, max_scene_s=max_scene_s, beats_spans=beats_spans))

    return [_build_scene(scene_id, g, slots, attrs) for scene_id, g in enumerate(split, start=1)]


# -- scene_target ------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# No shot in this film is truly inert -- even the calmest walk earns a
# track -- so the floor sits above zero; the three terms below carry the
# rest of the 0..1 range.
ENERGY_BASE = 0.3
# Weight of the hr-or-gain effort term -- the single biggest lever, since
# effort is what the film's energy curve is actually tracking.
ENERGY_EFFORT_WEIGHT = 0.4
# A scene in the climb or the summit act leans forward even before its own
# numbers are read, and a scene already marked calm leans back by the same
# amount -- so a climbing shot cut into the wrong act still reads urgent,
# and a village stop inside Act 3 still reads calm.
ENERGY_ACT_ADJUST = 0.2
DRAMATIC_ACTS = (3, 4)
CALM_ACTIVITIES = (VILLAGE, RESTING, CITY)

# A still scene sits at a resting pulse's tempo (60 bpm); the same span
# again (another 60) is what a scene moving at TEMPO_SPEED_NORM_MS reaches --
# the trek's own walking cadence, roughly 120 bpm -- before the cap takes
# over for anything faster.
TEMPO_BASE_BPM = 60.0
TEMPO_SPAN_BPM = 60.0
# Normalises around a brisk walking pace; S07 is told it may accept half or
# double time downstream, so the 1.5 cap only has to put a fast scene in the
# right neighbourhood, not compute an exact tempo.
TEMPO_SPEED_NORM_MS = 1.4
TEMPO_SPEED_CAP = 1.5

# The swell is the point of the climb -- a high dynamic range is what makes
# the act's music sound like it costs something.
DYNAMICS_CLIMB = 0.8
# Quiet and compressed enough that the cue does not fight what is being said.
DYNAMICS_SPEECH = 0.3
# The unmarked middle: neither the climb's swell nor speech's flattened bed.
DYNAMICS_NEUTRAL = 0.5
# A scene the audience is meant to be listening through, not over.
VOICE_SHARE_THRESHOLD = 0.5

# Dark: a pre-dawn start and thin high-altitude air both call for a cue that
# does not sound sunlit.
BRIGHTNESS_LOW = 0.2
# The brightest picture the film has -- a village or city seen by daylight --
# so the cue is allowed to sound the most open here.
BRIGHTNESS_HIGH = 0.9
# Everything else: neither a golden village noon nor a pre-dawn climb.
BRIGHTNESS_NEUTRAL = 0.5
# effort.py's light_quality treats 3:00-5:00 as the alpine start and 5:00-7:00
# as sunrise; a music target only needs the coarser "before real light"
# cut, so pre-dawn spans both of those bands.
PRE_DAWN_HOUR_MAX = 6.0
# The far edges of light_quality's own sunrise (ends 7:00) and dusk (starts
# 18:30) bands -- daytime is what is left once both twilight bands are cut.
DAYTIME_HOUR_MIN = 7.0
DAYTIME_HOUR_MAX = 18.5
# High enough that only genuine high-altitude terrain qualifies, roughly the
# trek's own highest passes, not just a high camp.
HIGH_ALTITUDE_BRIGHTNESS_M = 4500.0


def _effort_term(scene: Scene, *, hr_rest: float, hr_max: float) -> float:
    """The pulse is half the answer wherever the watch recorded one
    (effort.py's own reasoning); without it, gain is the next best fact a
    scene carries about how hard the ground was."""
    if scene.hr_bpm is not None:
        return _clamp01((scene.hr_bpm - hr_rest) / (hr_max - hr_rest))
    gain = scene.gain_m_per_h if scene.gain_m_per_h is not None else 0.0
    return _clamp01(max(0.0, gain) / STEEP_GAIN_M_PER_H)


def scene_target(scene: Scene, *, hr_rest: float, hr_max: float) -> dict[str, float]:
    """What the music should be doing for this scene, before any track is
    matched to it. ``music.scene_targets`` (config) weights the distance to
    a candidate track's own features -- that comparison is S07's job, not
    this one."""
    energy = ENERGY_BASE + ENERGY_EFFORT_WEIGHT * _effort_term(scene, hr_rest=hr_rest, hr_max=hr_max)
    if scene.act in DRAMATIC_ACTS:
        energy += ENERGY_ACT_ADJUST
    if scene.activity in CALM_ACTIVITIES:
        energy -= ENERGY_ACT_ADJUST
    energy = _clamp01(energy)

    # A scene with no GPS fix is not thereby the film's slowest moment -- it
    # takes the walking norm the tempo formula is itself calibrated to, the
    # cadence the film follows by default, not silence.
    speed = scene.speed_ms if scene.speed_ms is not None else TEMPO_SPEED_NORM_MS
    tempo_bpm = TEMPO_BASE_BPM + TEMPO_SPAN_BPM * min(speed / TEMPO_SPEED_NORM_MS, TEMPO_SPEED_CAP)

    if scene.activity in (CLIMBING, SUMMIT):
        dynamics = DYNAMICS_CLIMB
    elif scene.voice_share > VOICE_SHARE_THRESHOLD:
        dynamics = DYNAMICS_SPEECH
    else:
        dynamics = DYNAMICS_NEUTRAL

    pre_dawn = scene.hour is not None and scene.hour < PRE_DAWN_HOUR_MAX
    daytime = scene.hour is not None and DAYTIME_HOUR_MIN <= scene.hour < DAYTIME_HOUR_MAX
    high_alt = scene.alt_m is not None and scene.alt_m > HIGH_ALTITUDE_BRIGHTNESS_M
    if pre_dawn or high_alt:
        brightness = BRIGHTNESS_LOW
    elif scene.activity in (VILLAGE, CITY) and daytime:
        brightness = BRIGHTNESS_HIGH
    else:
        brightness = BRIGHTNESS_NEUTRAL

    return {"energy": energy, "tempo_bpm": tempo_bpm, "dynamics": dynamics, "brightness": brightness}
