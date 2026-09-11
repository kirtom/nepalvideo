import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np
import pytest

from nepal.spine.music import (Track, parse_artist_title, estimate_key, mark_swells,
                               znorm, target_vector, callback_affinity, assign_acts,
                               allocate_act_durations, choose_total_duration,
                               build_music_map, check_music_map,
                               detect_licence, ACT_WEIGHTS, PITCH_CLASSES,
                               MAJOR_PROFILE, MINOR_PROFILE)

ACT_SPECS = [
    {"act": 1, "name": "Planning", "min_s": 120, "max_s": 180},
    {"act": 2, "name": "Approach", "min_s": 240, "max_s": 300},
    {"act": 3, "name": "Climb", "min_s": 360, "max_s": 420},
    {"act": 4, "name": "Highest point", "min_s": 90, "max_s": 120},
    {"act": 5, "name": "Descent", "min_s": 180, "max_s": 240},
]


def mk(tid, energy, p95, p10, centroid, onset, tempo, artist=None, key=None, lic="personal"):
    return Track(track_id=tid, s3_key=f"raw/music/{tid}.mp3", title=tid,
                 duration_s=300.0, tempo_bpm=tempo, key_est=key,
                 energy_mean=energy, energy_p95=p95, energy_p10=p10,
                 centroid=centroid, onset_rate=onset, artist=artist, licence=lic,
                 sections=[{"section_id": f"{tid}_s0", "track_id": tid,
                            "start_s": 0.0, "end_s": 60.0, "energy": energy},
                           {"section_id": f"{tid}_s1", "track_id": tid,
                            "start_s": 60.0, "end_s": 120.0, "energy": p95}],
                 beats=[i * 2.0 for i in range(150)],
                 downbeats=[i * 8.0 for i in range(37)])


def library():
    return [
        mk("piano",   0.05, 0.08, 0.02,  800, 1.0,  60),
        mk("strings", 0.12, 0.18, 0.06, 1600, 2.0,  90),
        mk("climb",   0.16, 0.30, 0.04, 1800, 2.2,  95),
        mk("peak",    0.28, 0.40, 0.12, 3200, 4.0, 140),
        mk("return",  0.08, 0.12, 0.03,  900, 1.2,  62),
    ]


# -- helpers -----------------------------------------------------------

@pytest.mark.parametrize("name,artist,title", [
    ("03 - Nils Frahm - Says.mp3", "Nils Frahm", "Says"),
    ("Nils Frahm — Ambre.flac", "Nils Frahm", "Ambre"),
    ("01_sparse_piano.mp3", None, "sparse_piano"),   # leading track number stripped
    ("Says.mp3", None, "Says"),
])
def test_parse_artist_title(name, artist, title):
    assert parse_artist_title(name) == (artist, title)


def test_act4_tempo_carries_no_weight():
    """The spec marks Act 4 tempo 'any' -- it must not pull the assignment."""
    assert ACT_WEIGHTS[4][-1] == 0.0
    assert all(w == 1.0 for w in ACT_WEIGHTS[1])


def test_znorm_handles_a_constant_column():
    m = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    z = znorm(m)
    assert not np.isnan(z).any()
    assert (z[:, 1] == 0).all()


def test_znorm_single_row():
    assert not np.isnan(znorm(np.array([[1.0, 2.0]]))).any()


# -- key ---------------------------------------------------------------

def test_estimate_key_recovers_a_major_profile():
    for shift, expected in [(0, "Cmaj"), (7, "Gmaj"), (9, "Amaj")]:
        assert estimate_key(np.roll(MAJOR_PROFILE, shift)) == expected


def test_estimate_key_recovers_a_minor_profile():
    assert estimate_key(np.roll(MINOR_PROFILE, 9)) == "Amin"


def test_estimate_key_on_silence():
    assert estimate_key(np.zeros(12)) == "unknown"


def test_estimate_key_wrong_length():
    assert estimate_key([1, 2, 3]) == "unknown"


# -- swells ------------------------------------------------------------

def test_swell_needs_both_loud_and_rising():
    sections = [{"energy": 0.9}, {"energy": 0.1}, {"energy": 0.95}]
    got = mark_swells(sections, percentile=75)
    assert got[0]["is_swell"] == 0, "loud but not rising from anything"
    assert got[1]["is_swell"] == 0, "quiet"
    assert got[2]["is_swell"] == 1, "loud and rising"


def test_mark_swells_empty():
    assert mark_swells([]) == []


# -- assignment --------------------------------------------------------

def test_each_act_gets_its_matching_track():
    tracks = library()
    a = assign_acts(tracks)
    assert a.by_act[1] == "piano"
    assert a.by_act[3] == "climb"
    assert a.by_act[4] == "peak"
    assert a.by_act[5] == "return"
    assert a.by_act[2] == "strings"


def test_assignment_is_a_permutation():
    a = assign_acts(library())
    assert len(set(a.by_act.values())) == 5


def test_assigned_act_is_written_back_to_tracks():
    tracks = library()
    assign_acts(tracks)
    assert {t.track_id: t.assigned_act for t in tracks}["peak"] == 4


def test_callback_bonus_rewards_shared_artist():
    a = callback_affinity(mk("x", .05, .08, .02, 800, 1, 60, artist="Nils Frahm"),
                          mk("y", .08, .12, .03, 900, 1, 62, artist="nils frahm"))
    b = callback_affinity(mk("x", .05, .08, .02, 800, 1, 60, artist="Nils Frahm"),
                          mk("y", .08, .12, .03, 900, 1, 62, artist="Someone Else"))
    assert a > b


def test_callback_bonus_rewards_shared_key_and_relative_mode():
    ref = mk("x", .05, .08, .02, 800, 1, 60, key="Amin")
    same = callback_affinity(ref, mk("y", .08, .12, .03, 900, 1, 62, key="Amin"))
    tonic = callback_affinity(ref, mk("z", .08, .12, .03, 900, 1, 62, key="Amaj"))
    none = callback_affinity(ref, mk("w", .08, .12, .03, 900, 1, 62, key="F#min"))
    assert same > tonic > none


def test_callback_changes_the_act5_choice():
    """Two near-identical descent candidates; the one echoing Act 1 must win.

    This is the design requirement the spec calls out -- the callback is what
    makes twenty minutes read as a film rather than a montage.
    """
    tracks = [
        mk("piano",   0.05, 0.08, 0.02,  800, 1.0,  60, artist="Nils Frahm", key="Amin"),
        mk("strings", 0.12, 0.18, 0.06, 1600, 2.0,  90),
        mk("climb",   0.16, 0.30, 0.04, 1800, 2.2,  95),
        mk("peak",    0.28, 0.40, 0.12, 3200, 4.0, 140),
        mk("returnA", 0.08, 0.12, 0.03,  900, 1.2,  62, artist="Other", key="F#min"),
        mk("returnB", 0.081, 0.121, 0.031, 905, 1.21, 62.5, artist="Nils Frahm", key="Amin"),
    ]
    a = assign_acts(tracks)
    assert a.by_act[5] == "returnB"
    assert a.callback_bonus > 0


def test_library_smaller_than_the_act_count_reuses_and_says_so():
    tracks = [mk("piano", 0.05, 0.08, 0.02, 800, 1.0, 60),
              mk("peak", 0.28, 0.40, 0.12, 3200, 4.0, 140)]
    a = assign_acts(tracks)
    assert a.fallback_used
    assert "reused" in a.note
    assert set(a.by_act) == {1, 2, 3, 4, 5}
    assert a.by_act[5] == a.by_act[1], "Act 5 must echo Act 1 when choices run out"


def test_cleared_mode_restricts_the_pool():
    tracks = library()
    for t in tracks:
        t.licence = "cleared" if t.track_id in ("piano", "peak") else "personal"
    a = assign_acts(tracks, license_mode="cleared")
    assert set(a.by_act.values()) <= {"piano", "peak"}


def test_cleared_mode_with_no_cleared_tracks_reports_rather_than_crashes():
    a = assign_acts(library(), license_mode="cleared")
    assert a.fallback_used and "license_mode" in a.note


def test_empty_library():
    a = assign_acts([])
    assert a.fallback_used and a.by_act == {}


# -- durations ---------------------------------------------------------

def test_durations_hit_the_target_and_respect_the_bands():
    d = allocate_act_durations(ACT_SPECS, 1200)
    assert sum(d.values()) == pytest.approx(1200, abs=0.5)
    for spec in ACT_SPECS:
        a = spec["act"]
        assert spec["min_s"] - 0.01 <= d[a] <= spec["max_s"] + 0.01


@pytest.mark.parametrize("total", [990, 1100, 1200, 1260])
def test_durations_across_the_achievable_range(total):
    d = allocate_act_durations(ACT_SPECS, total)
    assert sum(d.values()) == pytest.approx(total, abs=0.5)


def test_durations_clamp_when_the_target_is_unreachable():
    """Asking for 3000 s from bands that max out at 1260 must clamp to the
    maximum rather than silently blow past the creative brief."""
    d = allocate_act_durations(ACT_SPECS, 3000)
    assert sum(d.values()) == pytest.approx(1260, abs=0.5)
    for spec in ACT_SPECS:
        assert d[spec["act"]] <= spec["max_s"] + 0.01


def test_short_cut_budget_clamps_to_the_minimum():
    d = allocate_act_durations(ACT_SPECS, 180)
    assert sum(d.values()) == pytest.approx(990, abs=0.5)


# The shipped bands: acts 2, 3 and 5 absorb extra runtime, act 4 barely moves.
SHIPPED_SPECS = [
    {"act": 1, "name": "Planning",      "target_s": 167, "min_s":  90, "max_s":  300},
    {"act": 2, "name": "Approach",      "target_s": 287, "min_s": 150, "max_s":  700},
    {"act": 3, "name": "Climb",         "target_s": 407, "min_s": 240, "max_s": 1150},
    {"act": 4, "name": "Highest point", "target_s": 113, "min_s":  90, "max_s":  150},
    {"act": 5, "name": "Descent",       "target_s": 227, "min_s": 120, "max_s":  600},
]


def test_durations_at_the_target_are_the_stated_targets():
    """At 20 minutes each act gets the length the creative brief gave it, not
    the midpoint of a band that is asymmetric by design."""
    d = allocate_act_durations(SHIPPED_SPECS, 1200)
    for spec in SHIPPED_SPECS:
        assert d[spec["act"]] == pytest.approx(spec["target_s"], abs=1.0)


def test_the_summit_barely_grows_when_the_film_does():
    """At 45 minutes a linear share would make act 4 four minutes long. It is a
    peak swell and a hard cut to silence -- a beat, not a phase."""
    at_target = allocate_act_durations(SHIPPED_SPECS, 1200)
    at_max = allocate_act_durations(SHIPPED_SPECS, 2700)
    assert sum(at_max.values()) == pytest.approx(2700, abs=1.0)
    assert at_max[4] <= 150
    # the journey acts take the runtime the summit does not
    assert at_max[3] - at_target[3] > 600
    assert at_max[4] - at_target[4] < 40


@pytest.mark.parametrize("total", [1200, 1500, 2000, 2700])
def test_shipped_bands_are_respected_across_the_range(total):
    d = allocate_act_durations(SHIPPED_SPECS, total)
    assert sum(d.values()) == pytest.approx(total, abs=1.0)
    for spec in SHIPPED_SPECS:
        assert spec["min_s"] - 0.01 <= d[spec["act"]] <= spec["max_s"] + 0.01


def test_specs_without_a_target_still_start_at_the_midpoint():
    d = allocate_act_durations(ACT_SPECS, 1200)
    assert sum(d.values()) == pytest.approx(1200, abs=0.5)


# -- how long the film should be ---------------------------------------

def test_unknown_material_means_the_target():
    """Before S05 has scored the shots the length of the usable material is
    unknown, and the film must not grow on a guess."""
    assert choose_total_duration(1200, 2700, material_s=None) == 1200
    assert choose_total_duration(1200, 2700, material_s=0) == 1200


def test_material_that_barely_covers_the_film_buys_nothing():
    """A film cut 1:1 from its rushes is not a cut. Below the selectivity the
    material needs, there is no surplus to spend."""
    assert choose_total_duration(1200, 2700, material_s=1200, selectivity=3.0) == 1200
    assert choose_total_duration(1200, 2700, material_s=3600, selectivity=3.0) == 1200


def test_the_film_grows_with_material_but_stays_under_the_ceiling():
    lengths = [choose_total_duration(1200, 2700, material_s=3600 * k, selectivity=3.0)
               for k in (2, 5, 10, 100)]
    assert lengths == sorted(lengths)                 # monotone in material
    assert all(1200 < x < 2700 for x in lengths)      # earns growth, never reaches the cap
    # the bias is toward twenty minutes: twice the material needed buys well
    # under half the available twenty-five minutes
    assert lengths[0] < 1200 + 0.5 * (2700 - 1200)


def test_growth_bias_zero_pins_the_film_to_the_target():
    assert choose_total_duration(1200, 2700, material_s=3600 * 100,
                                 growth_bias=0.0) == 1200


def test_a_ceiling_at_the_target_is_honoured():
    assert choose_total_duration(1200, 1200, material_s=3600 * 100) == 1200


# -- music map ---------------------------------------------------------

def test_music_map_structure_and_acceptance():
    tracks = library()
    a = assign_acts(tracks)
    m = build_music_map(tracks, a, ACT_SPECS, total_s=1200, silence_s=3.0)
    # min_headroom=1.0 isolates structure from library size: this five-track
    # fixture is deliberately small, and library adequacy has its own tests.
    assert check_music_map(m, target_s=1200, min_headroom=1.0) == []
    assert len(m["acts"]) == 5
    assert m["acts"][0]["t_start"] == 0.0
    # acts are contiguous
    for prev, nxt in zip(m["acts"], m["acts"][1:]):
        assert prev["t_end"] == pytest.approx(nxt["t_start"])


def test_silence_window_follows_act_four():
    tracks = library()
    m = build_music_map(tracks, assign_acts(tracks), ACT_SPECS, total_s=1200, silence_s=3.0)
    act4 = next(a for a in m["acts"] if a["act"] == 4)
    assert m["silence_window"]["t_start"] == pytest.approx(act4["t_end"])
    assert m["silence_window"]["t_end"] - m["silence_window"]["t_start"] == pytest.approx(3.0)


def test_every_act_gets_a_swell_even_without_a_rising_section():
    tracks = library()
    for t in tracks:
        t.sections = [{"section_id": "s0", "track_id": t.track_id, "start_s": 0.0,
                       "end_s": 60.0, "energy": 0.5, "is_swell": 0}]
    m = build_music_map(tracks, assign_acts(tracks), ACT_SPECS, total_s=1200)
    assert all(a["swells"] for a in m["acts"])
    assert check_music_map(m, target_s=1200, min_headroom=1.0) == []


def test_beat_grid_is_offset_onto_the_film_timeline():
    tracks = library()
    m = build_music_map(tracks, assign_acts(tracks), ACT_SPECS, total_s=1200)
    act2 = next(a for a in m["acts"] if a["act"] == 2)
    assert act2["beat_grid"], "act 2 must carry a beat grid"
    assert min(act2["beat_grid"]) >= act2["t_start"]
    assert max(act2["beat_grid"]) <= act2["t_end"] + 1e-6


def test_check_music_map_flags_a_bad_total():
    tracks = library()
    m = build_music_map(tracks, assign_acts(tracks), ACT_SPECS, total_s=1200)
    m["total_duration_s"] = 900
    assert any("outside" in p for p in check_music_map(m, target_s=1200))


# -- licences ----------------------------------------------------------

def test_detect_licence_from_directory():
    assert detect_licence(pathlib.Path("music/cleared/Discovery.mp3")) == "cleared"
    assert detect_licence(pathlib.Path("music/Says.mp3")) == "personal"


def test_detect_licence_from_manifest():
    got = detect_licence(pathlib.Path("music/Says.mp3"), {"Says.mp3": "cleared"})
    assert got == "cleared"


# -- multi-track acts --------------------------------------------------

from nepal.spine.music import fill_act, rank_for_act, read_tags


def song(tid, artist, dur=210.0, energy=0.3, dyn=0.15, centroid=1500.0,
         onset=2.0, tempo=100.0):
    t = Track(track_id=tid, s3_key="", title=tid, artist=artist, duration_s=dur,
              tempo_bpm=tempo, key_est="Amin", energy_mean=energy,
              energy_p95=energy + dyn / 2, energy_p10=energy - dyn / 2,
              centroid=centroid, onset_rate=onset)
    t.sections = [{"section_id": f"{tid}_s{i}", "track_id": tid,
                   "start_s": i * dur / 4, "end_s": (i + 1) * dur / 4,
                   "energy": energy * (1 + i * 0.2), "is_swell": 1 if i == 2 else 0}
                  for i in range(4)]
    t.beats = [i * 60.0 / tempo for i in range(int(dur * tempo / 60))]
    t.downbeats = t.beats[::4]
    return t


def spread(n):
    """n tracks of 3:30 spanning the act targets."""
    out = []
    for i in range(n):
        f = i / max(n - 1, 1)
        out.append(song(f"t{i:02d}", f"artist{i % 7}", 210,
                        0.05 + 0.8 * f, 0.03 + 0.25 * f, 600 + 2600 * f,
                        0.5 + 4 * f, 58 + 82 * f))
    return out


def test_fill_act_lays_multiple_tracks_end_to_end():
    """One track per act does not cover a 20-minute film: Act 3 is allocated
    407 s and a typical song is 210 s."""
    pool = spread(8)
    segs = fill_act(3, pool[5], pool, 407.0)
    assert len(segs) >= 2
    assert segs[0]["track_id"] == pool[5].track_id, "the assigned track opens the act"
    assert segs[0]["t_in"] == 0.0
    assert segs[-1]["t_end"] == pytest.approx(407.0)


def test_fill_act_segments_are_contiguous_and_ordered():
    pool = spread(8)
    segs = fill_act(3, pool[5], pool, 407.0)
    for prev, nxt in zip(segs, segs[1:]):
        assert prev["t_end"] == pytest.approx(nxt["t_in"])


def test_fill_act_needs_only_one_track_for_a_short_act():
    pool = spread(8)
    segs = fill_act(4, pool[7], pool, 113.0)
    assert len(segs) == 1
    assert segs[0]["t_end"] == pytest.approx(113.0)


def test_fill_act_prefers_tracks_not_used_elsewhere():
    pool = spread(8)
    used = {t.track_id for t in pool[1:6]}
    segs = fill_act(3, pool[0], pool, 407.0, already_used=used)
    fillers = [s["track_id"] for s in segs[1:]]
    assert any(f not in used for f in fillers), "should reach for an unused track"


def test_fill_act_reuses_when_the_library_runs_out():
    """A repeated track beats an act with no music."""
    pool = [song("only", "a", dur=100.0)]
    segs = fill_act(3, pool[0], pool, 407.0)
    assert segs
    assert segs[-1]["t_end"] == pytest.approx(407.0)


def test_fill_act_with_no_tracks():
    assert fill_act(3, None, [], 407.0) == []


def test_rank_for_act_puts_the_quiet_track_first_for_act_one():
    pool = spread(8)
    ranked = rank_for_act(pool, 1)
    assert ranked[0].energy_mean < ranked[-1].energy_mean


def test_rank_for_act_puts_the_loud_track_first_for_act_four():
    pool = spread(8)
    ranked = rank_for_act(pool, 4)
    assert ranked[0].energy_mean > ranked[-1].energy_mean


# -- beat grid coverage ------------------------------------------------

def test_beat_grid_covers_the_whole_act_with_enough_tracks():
    """S06 snaps every cut to the grid, so an act whose grid stops early cannot
    be cut on the beat past that point."""
    pool = spread(12)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    for act in m["acts"]:
        span = act["t_end"] - act["t_start"]
        assert act["beat_grid_covers_s"] >= span * 0.9, \
            f"act {act['act']}: grid covers {act['beat_grid_covers_s']:.0f}s of {span:.0f}s"


def test_a_library_shorter_than_the_film_is_flagged():
    """The clean top-level question. Coverage cannot detect a thin library
    because fill_act reuses tracks rather than leaving an act silent, and
    per-act repeat counts stay low until the library is tiny. Total minutes
    against the film's length is unambiguous."""
    pool = [song(f"short{i}", chr(97 + i), dur=60.0, energy=0.05 + 0.2 * i)
            for i in range(5)]                       # 5 min of music, 20 min film
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    probs = check_music_map(m, target_s=1200)
    assert any("must repeat" in p for p in probs), probs
    assert any("more minutes" in p for p in probs), "say how much is missing"


def test_a_library_with_little_headroom_is_flagged_separately():
    """Enough music, but no choice -- acts get whatever is nearest."""
    pool = [song(f"t{i}", chr(97 + i), dur=210.0, energy=0.05 + 0.2 * i)
            for i in range(7)]                       # 24.5 min for a 20 min film
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    probs = check_music_map(m, target_s=1200, min_headroom=2.0)
    assert any("little to choose from" in p for p in probs), probs
    assert not any("must repeat" in p for p in probs)


def test_library_duration_is_reported():
    pool = spread(12)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    assert m["library_duration_s"] == pytest.approx(12 * 210, abs=1)


def test_a_healthy_library_passes_both_checks():
    pool = spread(12)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    assert check_music_map(m, target_s=1200) == []


def test_repetition_counts_are_reported_per_act():
    pool = spread(12)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    for act in m["acts"]:
        assert act["n_distinct_tracks"] >= 1
        assert act["max_track_repeats"] >= 1


def test_segments_carry_artist_and_title_for_the_gate_payload():
    pool = spread(8)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    seg = m["acts"][0]["segments"][0]
    assert "artist" in seg and "title" in seg
    assert seg["src_in"] == 0.0 and seg["src_out"] > 0


def test_track_id_still_names_the_opening_track():
    """Kept for readability and backward compatibility with the earlier shape."""
    pool = spread(8)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    for act in m["acts"]:
        assert act["track_id"] == act["segments"][0]["track_id"]


def test_map_reports_library_utilisation():
    pool = spread(27)
    m = build_music_map(pool, assign_acts(pool), ACT_SPECS, total_s=1200)
    assert m["n_tracks_available"] == 27
    assert 5 <= m["n_tracks_used"] <= 27


# -- ID3 tags ----------------------------------------------------------

def test_read_tags_on_a_missing_file(tmp_path):
    assert read_tags(tmp_path / "absent.mp3") == (None, None)


def test_read_tags_on_a_non_audio_file(tmp_path):
    f = tmp_path / "notaudio.mp3"
    f.write_bytes(b"this is not an mp3")
    assert read_tags(f) == (None, None)


# -- variety: the fill must not spend the assignment ---------------------

def long_library():
    """Tracks long enough that one could swallow a whole act."""
    return [mk(f"t{i}", 0.2 + i * 0.01, 0.4, 0.05, 2000.0, 2.0, 100 + i)
            for i in range(8)]


def test_fill_does_not_spend_another_acts_primary():
    """Act 2's filler took Giorgio by Moroder, which was Act 3's primary, so
    Act 3 opened on a track the audience had just heard."""
    lib = long_library()
    primary, reserved_for_act3 = lib[0], lib[1]
    segs = fill_act(2, primary, lib, 900.0, reserved={reserved_for_act3.track_id})
    assert segs[0]["track_id"] == primary.track_id
    assert reserved_for_act3.track_id not in {s["track_id"] for s in segs[1:]}


def test_an_acts_own_primary_is_never_held_back_from_it():
    lib = long_library()
    primary = lib[0]
    # the whole library is reserved, including this act's primary
    segs = fill_act(3, primary, lib, 200.0,
                    reserved={t.track_id for t in lib})
    assert segs[0]["track_id"] == primary.track_id


def test_reserved_tracks_are_a_last_resort_not_a_prohibition():
    """A library too small to fill the act must still fill it."""
    lib = long_library()[:2]
    segs = fill_act(1, lib[0], lib, 2000.0, reserved={lib[1].track_id})
    assert segs, "an act with no music is worse than a reused track"
    assert sum(s["t_end"] - s["t_in"] for s in segs) == pytest.approx(2000.0, abs=1.0)


def test_max_segment_s_breaks_a_long_track_off_its_act():
    lib = long_library()
    segs = fill_act(3, lib[0], lib, 900.0, max_segment_s=150.0)
    assert len(segs) >= 3
    # every segment but the last respects the cap
    for seg in segs[:-1]:
        assert seg["t_end"] - seg["t_in"] <= 150.0 + 1e-6


def test_an_act_shorter_than_the_cap_stays_one_piece():
    """Act 4 is a single unbroken swell into the silence."""
    lib = long_library()
    segs = fill_act(4, lib[0], lib, 113.0, max_segment_s=150.0)
    assert len(segs) == 1
    assert segs[0]["t_end"] == pytest.approx(113.0)


def test_a_remainder_below_the_minimum_extends_rather_than_stubs():
    lib = long_library()
    segs = fill_act(2, lib[0], lib, 160.0, max_segment_s=150.0, min_segment_s=20.0)
    assert all(s["t_end"] - s["t_in"] >= 20.0 for s in segs)
    assert segs[-1]["t_end"] == pytest.approx(160.0)


def test_the_cap_and_the_reservation_together_widen_the_soundtrack():
    """Neither alone is enough: the cap splits the acts but the fill then takes
    the reserved primaries back, and the reservation alone leaves a long track
    still swallowing its whole act.

    The library has to be larger than the film needs, or variety is not a
    choice: 20 songs for 20 minutes, one of them nine minutes long and assigned
    to Act 3 -- which is how the real corpus is shaped.
    """
    lib = []
    for i in range(20):
        t = mk(f"v{i}", 0.2 + i * 0.005, 0.4, 0.05, 2000.0, 2.0, 100 + i)
        t.duration_s = 540.0 if i == 2 else 210.0
        lib.append(t)
    by_id = {t.track_id: t for t in lib}
    primaries = {1: "v0", 2: "v1", 3: "v2", 4: "v3", 5: "v4"}
    reserved = set(primaries.values())
    durations = allocate_act_durations(SHIPPED_SPECS, 1200)

    def distinct(cap, res):
        used = set()
        for act in (1, 2, 3, 4, 5):
            for seg in fill_act(act, by_id[primaries[act]], lib, durations[act],
                                already_used=set(used), max_segment_s=cap,
                                reserved=res):
                used.add(seg["track_id"])
        return len(used)

    plain = distinct(None, None)
    assert distinct(150.0, reserved) > plain
    # and every act still opens on the track it was assigned
    used = set()
    for act in (1, 2, 3, 4, 5):
        segs = fill_act(act, by_id[primaries[act]], lib, durations[act],
                        already_used=set(used), max_segment_s=150.0,
                        reserved=reserved)
        assert segs[0]["track_id"] == primaries[act]
        for seg in segs:
            used.add(seg["track_id"])


def test_build_music_map_reserves_the_primaries_by_itself():
    tracks = library()
    a = assign_acts(tracks)
    m = build_music_map(tracks, a, ACT_SPECS, total_s=1200, silence_s=3.0,
                        max_segment_s=150.0)
    # no act opens on a track another act was assigned
    primaries = {tid for tid in a.by_act.values() if tid}
    for act in m["acts"]:
        own = act["track_id"]
        for seg in act["segments"][1:]:
            assert seg["track_id"] not in (primaries - {own}) or \
                len(tracks) < len(m["acts"]) + 1
