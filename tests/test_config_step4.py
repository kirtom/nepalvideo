import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nepal.config import Config


def test_the_step4_keys_exist_with_the_spec_values():
    cfg = Config.load(None)
    assert cfg.get("film.target_duration_s") == 1500 and cfg.get("film.growth_bias") == 0.6
    assert cfg.get("film.cold_open_s") == [15, 25] and cfg.get("film.cold_open_card")
    assert cfg.get("beats.face_hold_s") == 2.5 and cfg.get("beats.pre_roll_s") == 0.4
    assert cfg.get("beats.broll_window_s") == 7200
    a = cfg.get("assemble")
    for k in ("pair_window_s", "pairs_per_act", "max_consecutive_recording",
              "source_alternation_after", "source_share_min", "long_take_s",
              "long_take_keywords", "rhythm", "similarity_fallback", "act4_held_shot_s",
              "natural_sound_windows", "natural_sound_window_s", "subject_shot_every_s",
              "levity_min_per_act"):
        assert k in a, k
    assert cfg.get("music.assignment") == "scene" and cfg.get("music.min_scene_s") == 45
    assert cfg.get("music.reuse_gap_s") == 300
    r = cfg.get("render")
    for k in ("speech_lufs", "location_full_lufs", "location_under_speech_lufs",
              "music_xfade_s", "window_fade_s", "final_lufs", "true_peak_db"):
        assert k in r, k
