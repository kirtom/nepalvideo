"""S06 -- selection, constraints and the beat grid."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal import db
from nepal.process import assemble


def shot(sid, score=0.5, **kw):
    d = {"shot_id": sid, "score_total": score, "act": 3, "start_s": 0.0}
    d.update(kw)
    return d


# -- slot budget -------------------------------------------------------

def test_the_budget_is_the_act_runtime_over_a_mid_shot():
    assert assemble.slot_budget(100.0, [4.0, 6.0]) == 20     # 100 / 5
    assert assemble.slot_budget(83.0, [3.0, 5.0]) == 21      # 83 / 4


def test_an_act_always_gets_at_least_one_slot():
    assert assemble.slot_budget(1.0, [10.0, 20.0]) == 1


# -- beat snapping -----------------------------------------------------

def test_a_cut_snaps_to_the_nearest_beat_not_the_next_one():
    """Snapping forward would push every following shot and accumulate."""
    beats = [0.0, 1.0, 2.0, 3.0]
    assert assemble.snap_to_beat(1.9, beats) == 2.0
    assert assemble.snap_to_beat(2.1, beats) == 2.0


def test_an_act_transition_snaps_to_a_downbeat():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    downbeats = [0.0, 2.0]
    assert assemble.snap_to_beat(1.4, beats) == 1.5
    assert assemble.snap_to_beat(1.4, beats, downbeats=downbeats) == 2.0


def test_no_grid_leaves_the_time_alone():
    assert assemble.snap_to_beat(3.7, []) == 3.7


# -- MMR ---------------------------------------------------------------

def _emb(mapping):
    return {k: np.array(v, dtype=np.float32) / np.linalg.norm(v)
            for k, v in mapping.items()}


_FALLBACK_W = {"same_recording_60s": 1.0, "same_recording": 0.6, "same_place_hour": 0.3}


def _sim(embeddings):
    return assemble.make_similarity(embeddings, fallback_weights=_FALLBACK_W)


def test_without_diversity_pressure_the_best_shots_win():
    cands = [shot("a", 0.9), shot("b", 0.8), shot("c", 0.7)]
    got = assemble.mmr_select(cands, budget=2, similarity=_sim({}), lam=0.0)
    assert [s["shot_id"] for s in got] == ["a", "b"]


def test_mmr_refuses_a_second_shot_that_looks_like_the_first():
    """Three shots: a and b are the same ridge, c is different and scores
    slightly lower. A plain greedy fill takes a then b; MMR takes a then c."""
    E = _emb({"a": [1, 0], "b": [0.99, 0.01], "c": [0, 1]})
    cands = [shot("a", 0.90), shot("b", 0.89), shot("c", 0.80)]
    plain = assemble.mmr_select(cands, budget=2, similarity=_sim(E), lam=0.0)
    diverse = assemble.mmr_select(cands, budget=2, similarity=_sim(E), lam=0.5)
    assert [s["shot_id"] for s in plain] == ["a", "b"]
    assert [s["shot_id"] for s in diverse] == ["a", "c"]


def test_a_seeded_pass_is_penalised_for_resembling_what_came_before():
    """A relaxed second pass must not re-pick the first pass's ridge: with
    the first pass's picks as ``seed``, the candidate identical to one of
    them ranks last under any diversity pressure, though the returned list
    holds this call's picks alone. Without a seed nothing changes."""
    E = _emb({"a": [1, 0], "b": [0.99, 0.01], "c": [0, 1], "d": [0.7, 0.7]})
    first = [shot("a", 0.90)]
    later = [shot("b", 0.89), shot("c", 0.80), shot("d", 0.79)]
    # seeded with a: b is a's twin and drops to 0.39; c (0.80) then d (0.44) go first
    seeded = assemble.mmr_select(later, budget=3, similarity=_sim(E), lam=0.5, seed=first)
    assert [s["shot_id"] for s in seeded] == ["c", "d", "b"]
    assert [s["shot_id"] for s in assemble.mmr_select(later, budget=3, similarity=_sim(E), lam=0.5)] \
        == ["b", "c", "d"]


def test_an_unknown_similarity_does_not_veto_a_shot():
    """A shot missing from the embedding index, with no fallback signal
    either (no shared recording or place), must still be selectable -- it is
    footage we know less about, not footage that repeats."""
    cands = [shot("a", 0.9), shot("nope", 0.8)]
    got = assemble.mmr_select(cands, budget=2, similarity=_sim(_emb({"a": [1, 0]})), lam=0.9)
    assert [s["shot_id"] for s in got] == ["a", "nope"]


def test_the_fill_stops_rather_than_forcing_an_inadmissible_shot():
    cands = [shot("a", 0.9), shot("b", 0.8)]
    got = assemble.mmr_select(cands, budget=2, similarity=_sim({}),
                              admissible=lambda c, chosen: not chosen)
    assert len(got) == 1


def test_unscored_candidates_are_never_selected():
    cands = [shot("a", None), shot("b", 0.1)]
    got = assemble.mmr_select(cands, budget=5, similarity=_sim({}))
    assert [s["shot_id"] for s in got] == ["b"]


# -- similarity without embeddings, and the constraint ladder ----------

def test_fallback_similarity_is_deterministic_and_graded():
    w = {"same_recording_60s": 1.0, "same_recording": 0.6, "same_place_hour": 0.3}
    a = {"recording_id": "r1", "start_utc": "2024-05-01T04:00:00+00:00", "place_name": "Deng"}
    b = {"recording_id": "r1", "start_utc": "2024-05-01T04:00:30+00:00", "place_name": "Deng"}
    c = {"recording_id": "r1", "start_utc": "2024-05-01T05:30:00+00:00", "place_name": "Deng"}
    d = {"recording_id": "r2", "start_utc": "2024-05-01T04:20:00+00:00", "place_name": "Deng"}
    e = {"recording_id": "r3", "start_utc": "2024-05-02T04:20:00+00:00", "place_name": "Bihi"}
    assert assemble.fallback_similarity(a, b, weights=w) == 1.0
    assert assemble.fallback_similarity(a, c, weights=w) == 0.6
    assert assemble.fallback_similarity(a, d, weights=w) == 0.3
    assert assemble.fallback_similarity(a, e, weights=w) == 0.0


def test_the_fill_breaks_up_a_run_of_one_recording_without_embeddings():
    """The first draft's runs of one recording: with no CLIP, similarity was
    0.0 and MMR was score order. The fallback alone prevents it."""
    cands = [{"shot_id": f"r1#{i}", "recording_id": "r1", "score_total": 0.9 - i * 0.01,
              "start_utc": f"2024-05-01T04:0{i}:00+00:00", "place_name": "Deng"} for i in range(4)]
    cands += [{"shot_id": "r2#0", "recording_id": "r2", "score_total": 0.5,
               "start_utc": "2024-05-01T06:00:00+00:00", "place_name": "Bihi"}]
    sim = assemble.make_similarity({}, fallback_weights={"same_recording_60s": 1.0,
                                                          "same_recording": 0.6, "same_place_hour": 0.3})
    picked = assemble.mmr_select(cands, budget=3, similarity=sim, lam=0.5)
    assert "r2#0" in {p["shot_id"] for p in picked}


def test_recording_run_limit_and_source_alternation():
    chosen = [{"recording_id": "r1", "source": "phone_keller"}] * 2
    assert not assemble.recording_run_ok({"recording_id": "r1"}, chosen, limit=2)
    assert assemble.recording_run_ok({"recording_id": "r9"}, chosen, limit=2)
    chosen3 = [{"source": "camera"}] * 3
    assert assemble.source_alternation_bonus({"source": "phone_keller"}, chosen3, after=3) > 0
    assert assemble.source_alternation_bonus({"source": "camera"}, chosen3, after=3) == 0


def test_source_share_repair_gives_the_starved_phone_its_quarter():
    chosen = [{"shot_id": f"k{i}", "source": "phone_kulikov", "score_total": 0.9 - i * 0.05}
              for i in range(8)]
    pool = [{"shot_id": f"e{i}", "source": "phone_keller", "score_total": 0.4} for i in range(3)]
    out = assemble.source_share_repair(chosen, pool, min_share=0.25)
    keller = [s for s in out if s["source"] == "phone_keller"]
    assert len(out) == 8 and len(keller) == 2          # 2 of 8 = 0.25
    assert {s["shot_id"] for s in out} >= {"k0", "k1", "k2"}   # the best of kulikov stay


def test_slot_keys_matches_the_timeline_v2_schema_plus_locked():
    assert assemble.SLOT_KEYS == db.TIMELINE_V2_COLUMNS + ("locked",)


# -- constraints -------------------------------------------------------

def test_chronology_is_by_capture_time():
    rows = [shot("late", start_utc="2024-05-02T10:00:00"),
            shot("early", start_utc="2024-05-01T10:00:00")]
    assert [s["shot_id"] for s in assemble.chronological(rows)] == ["early", "late"]


def test_a_shot_with_no_time_is_kept_at_the_end_not_dropped():
    rows = [shot("no_time"), shot("timed", start_utc="2024-05-01T10:00:00")]
    got = assemble.chronological(rows)
    assert [s["shot_id"] for s in got] == ["timed", "no_time"]


def test_a_place_cannot_take_more_than_its_share_of_an_act():
    chosen = [shot("a", place_name="Namche"), shot("b", place_name="Namche")]
    assert assemble.place_count_ok(shot("c", place_name="Namche"), chosen, limit=3)
    chosen.append(shot("c", place_name="Namche"))
    assert not assemble.place_count_ok(shot("d", place_name="Namche"), chosen, limit=3)


def test_an_unnamed_place_never_blocks_a_shot():
    chosen = [shot(str(i)) for i in range(9)]
    assert assemble.place_count_ok(shot("x"), chosen, limit=3)


def test_the_subject_quota_tracks_runtime():
    chosen = [shot("a", has_face=1)]
    assert not assemble.needs_subject(chosen, runtime_s=40, every_s=40)
    assert assemble.needs_subject(chosen, runtime_s=120, every_s=40)


def test_levity_is_owed_until_it_is_present():
    assert assemble.missing_levity([], minimum=1) == 1
    assert assemble.missing_levity([shot("a", tag_levity=1)], minimum=1) == 0


# -- speech is the spine, but not here -----------------------------------

def test_speech_is_no_longer_a_claim_on_a_slot_in_the_fill():
    """Film v2 section 4.3: `speech_first` is gone. The voice of the film is
    chosen by reading (the beat sheet); the fill does not reorder on a flag
    that two whisper hallucinations once carried to the front of their acts."""
    assert not hasattr(assemble, "speech_first")


# -- layout ------------------------------------------------------------

def test_lay_out_emits_every_slot_key_except_slot_index():
    beats = [i * 0.5 for i in range(40)]
    out = assemble.lay_out([shot("a", 0.5), shot("p", 0.5, media_kind="photo")],
                           start_s=0.0, duration_range=[4.0, 6.0], beats=beats)
    expected = set(assemble.SLOT_KEYS) - {"slot_index"}
    for row in out:
        assert set(row.keys()) == expected
    video_row, photo_row = out[0], out[1]
    assert video_row["kind"] == "video"
    assert photo_row["kind"] == "photo"
    for row in out:
        assert row["locked"] == 0
        assert row["speed"] == 1.0
        assert row["transition"] == "cut"
        for key in ("secondary_shot_id", "secondary_src_in", "motion",
                   "beat_id", "scene_id", "msg_id"):
            assert row[key] is None


def test_shots_are_laid_end_to_end_on_the_grid():
    beats = [i * 0.5 for i in range(40)]
    out = assemble.lay_out([shot("a", 0.5), shot("b", 0.5)], start_s=0.0,
                           duration_range=[4.0, 6.0], beats=beats)
    assert out[0]["t_in"] == 0.0
    assert out[1]["t_in"] == out[0]["t_out"], "no gap and no overlap"
    for s in out:
        assert 3.0 <= s["t_out"] - s["t_in"] <= 7.0


def test_a_stronger_shot_is_held_longer_but_stays_inside_the_act_range():
    beats = [i * 0.1 for i in range(400)]
    weak = assemble.lay_out([shot("w", 0.0)], start_s=0, duration_range=[4.0, 6.0], beats=beats)
    strong = assemble.lay_out([shot("s", 1.0)], start_s=0, duration_range=[4.0, 6.0], beats=beats)
    assert strong[0]["t_out"] > weak[0]["t_out"]
    assert weak[0]["t_out"] >= 3.9 and strong[0]["t_out"] <= 6.1


def test_the_source_window_matches_the_timeline_window():
    beats = [i * 0.5 for i in range(40)]
    out = assemble.lay_out([shot("a", 0.5, start_s=12.0)], start_s=0.0,
                           duration_range=[4.0, 6.0], beats=beats)
    row = out[0]
    assert row["src_in"] == 12.0
    assert row["src_out"] - row["src_in"] == pytest.approx(row["t_out"] - row["t_in"])


def test_a_coarse_grid_never_produces_a_zero_length_shot():
    out = assemble.lay_out([shot("a"), shot("b")], start_s=0.0,
                           duration_range=[4.0, 6.0], beats=[0.0])
    assert all(s["t_out"] > s["t_in"] for s in out)


# -- a slot may never be longer than its shot --------------------------

def test_a_slot_never_claims_more_footage_than_the_shot_has():
    """The first draft's 126 video slots asked for 664.9 s and the render
    delivered 593.3 s: ffmpeg simply stops at the end of the source. Every
    act-length and runtime number is computed from the claim, so the claim
    has to be true."""
    shots = [{"shot_id": "short", "start_s": 0.0, "end_s": 1.87,
              "media_kind": "video", "score_total": 1.0}]
    row = assemble.lay_out(shots, start_s=0.0, duration_range=[5.0, 8.0], beats=[])[0]
    assert row["t_out"] - row["t_in"] <= 1.87 + 1e-6
    assert row["src_out"] <= 1.87 + 1e-6


def test_a_shot_near_the_end_of_its_recording_is_clamped_too():
    """src_in + duration is what the render seeks to; the limit is the shot's
    own end, not zero-plus-the-act-range."""
    shots = [{"shot_id": "tail", "start_s": 37.64, "end_s": 40.77,
              "media_kind": "video", "score_total": 1.0}]
    row = assemble.lay_out(shots, start_s=0.0, duration_range=[5.0, 8.0], beats=[])[0]
    assert row["t_out"] - row["t_in"] == pytest.approx(3.13, abs=0.01)
    assert row["src_out"] == pytest.approx(40.77, abs=0.01)


def test_a_photograph_is_not_clamped_because_a_still_has_no_end():
    shots = [{"shot_id": "p", "media_kind": "photo", "score_total": 1.0}]
    row = assemble.lay_out(shots, start_s=0.0, duration_range=[5.0, 8.0], beats=[])[0]
    assert row["t_out"] - row["t_in"] == pytest.approx(8.0)


def test_clamping_still_prefers_a_beat_when_one_fits():
    """Dropping off the grid is a worse cut than a slightly shorter one."""
    shots = [{"shot_id": "s", "start_s": 0.0, "end_s": 2.0,
              "media_kind": "video", "score_total": 1.0}]
    row = assemble.lay_out(shots, start_s=0.0, duration_range=[5.0, 8.0],
                      beats=[0.5, 1.0, 1.5, 1.8, 2.5, 3.0])[0]
    assert row["t_out"] == pytest.approx(1.8), "the latest beat that fits"


def test_a_shot_shorter_than_the_beat_gap_falls_back_to_its_own_length():
    shots = [{"shot_id": "s", "start_s": 0.0, "end_s": 0.4,
              "media_kind": "video", "score_total": 1.0}]
    row = assemble.lay_out(shots, start_s=0.0, duration_range=[5.0, 8.0],
                      beats=[2.0, 4.0, 6.0])[0]
    assert row["t_out"] == pytest.approx(0.4)


def test_the_timeline_stays_contiguous_after_clamping():
    """A clamp must shorten the slot, not leave a hole where the rest of it was."""
    shots = [{"shot_id": "a", "start_s": 0.0, "end_s": 1.5, "media_kind": "video",
              "score_total": 1.0},
             {"shot_id": "b", "start_s": 0.0, "end_s": 30.0, "media_kind": "video",
              "score_total": 0.5}]
    rows = assemble.lay_out(shots, start_s=0.0, duration_range=[4.0, 6.0], beats=[])
    assert rows[1]["t_in"] == pytest.approx(rows[0]["t_out"])
