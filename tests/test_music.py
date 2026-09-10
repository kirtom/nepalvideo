import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np
import pytest

from nepal.spine.music import (Track, parse_artist_title, estimate_key, mark_swells,
                               znorm, target_vector, callback_affinity, assign_acts,
                               allocate_act_durations, build_music_map, check_music_map,
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


# -- music map ---------------------------------------------------------

def test_music_map_structure_and_acceptance():
    tracks = library()
    a = assign_acts(tracks)
    m = build_music_map(tracks, a, ACT_SPECS, total_s=1200, silence_s=3.0)
    assert check_music_map(m, target_s=1200) == []
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
    assert check_music_map(m, target_s=1200) == []


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
