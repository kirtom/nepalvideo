"""The beat sheet's rules, one at a time, and a good sheet passing them all."""
import copy
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.config import Config
from nepal.story import beats_schema as bs

RULES = bs.Rules(min_speech=3, max_speech=5, max_speech_s=25, min_quotes=1, max_quotes=2,
                 max_act4=1, max_pairs=2, cut_tolerance_s=0.05, snap_s=0.5,
                 card_max_chars=40, title_max_chars=30)

SHOTS = {
    "r#0001": {"act": 2, "utc": "2024-04-28T04:00:00+00:00", "cuts": [0.0, 5.0, 10.0],
               "text": "Мы приехали. Тут жарко."},
    "r#0002": {"act": 3, "utc": "2024-05-02T04:00:00+00:00", "cuts": [40.0, 48.0, 52.8],
               "text": "Я вот на этом курумнике прям сдох. Серьёзно."},
    "r#0003": {"act": 3, "utc": "2024-05-03T04:00:00+00:00", "cuts": [0.0, 20.0],
               "text": "Ну что, идём дальше, чай был отличный"},
    "r#0004": {"act": 4, "utc": "2024-05-06T01:00:00+00:00", "cuts": [0.0, 3.0],
               "text": "Покорена!"},
    "r#0005": {"act": 4, "utc": "2024-05-06T01:30:00+00:00", "cuts": [0.0, 3.0],
               "text": "Пять тысяч сто"},
}
MSGS = {"m1": {"phase": "planning", "text": "20 км в день это вообще не проблема"},
        "m2": {"phase": "planning", "text": "ну посмотрим"},
        "m4": {"phase": "after", "text": "Скучаю по горам"}}


def _beat(bid, shot, t_in, t_out, act, text, levity=False):
    return {"beat_id": bid, "kind": "speech", "act": act, "shot_id": shot, "msg_id": None,
            "src_in": t_in, "src_out": t_out, "text": text, "levity": levity,
            "effect": "none", "rank": 1, "rationale": "because"}


def good():
    return {"title": "Перевал", "beats": [
        _beat("b01", "r#0001", 0.0, 5.0, 2, "Мы приехали.", levity=True),
        _beat("b02", "r#0002", 40.0, 48.0, 3, "Я вот на этом курумнике прям сдох", levity=True),
        _beat("b03", "r#0003", 0.0, 20.0, 3, "чай был отличный"),
        _beat("b04", "r#0004", 0.0, 3.0, 4, "Покорена"),
        {"beat_id": "q01", "kind": "quote", "act": 1, "shot_id": None, "msg_id": "m1",
         "src_in": None, "src_out": None, "text": "20 км в день это вообще не проблема",
         "levity": True, "effect": "none", "rank": 5, "rationale": "confidence"}],
        "pairs": [{"planning_msg_id": "m1", "trek_beat_id": "b02", "why": "20 km a day"}],
        "closing": {"msg_id": "m4", "text": "Скучаю по горам"},
        "act_notes": [{"act": 3, "note": "slower"}], "stat_card_ideas": ["gain"],
        "trailer": {"hook_beat_id": "b02", "cliffhanger_beat_id": "b04", "cards": ["5,100 m"]}}


def _errs(mutate):
    doc = good()
    mutate(doc)
    return bs.validate(doc, shot_info=SHOTS, msg_info=MSGS, rules=RULES)


def test_a_good_sheet_passes():
    assert bs.validate(good(), shot_info=SHOTS, msg_info=MSGS, rules=RULES) == []


def test_rules_come_from_config():
    cfg = Config({"beats": {"min_speech_beats": 12, "max_speech_s": 20, "snap_s": 0.3},
                  "spine": {"card_max_chars": 150}})
    r = bs.Rules.from_cfg(cfg)
    assert (r.min_speech, r.max_speech_s, r.snap_s, r.card_max_chars) == (12, 20.0, 0.3, 150)
    assert r.as_prompt_dict()["acts_from"] == 2


def test_counts_and_chronology():
    def two_fewer(d):
        d["beats"].pop(0)
        d["beats"].pop(0)
    assert "2 speech beats; the film needs 3 to 5" in _errs(two_fewer)
    assert "speech beats are not in chronological order" in \
        _errs(lambda d: d["beats"].insert(0, d["beats"].pop(2)))


def test_every_act_from_two_needs_a_beat_and_a_light_one():
    def drop_levity(d):
        d["beats"][1]["levity"] = False
    assert "act 3 has no beat with levity" in _errs(drop_levity)

    def drop_act2(d):
        d["beats"][0] = _beat("b01", "r#0003", 0.0, 20.0, 3, "идём дальше", levity=True)
    assert "act 2 has no speech beat" in _errs(drop_act2)


def test_act_four_is_a_beat_not_a_phase():
    def two_summit_beats(d):
        d["beats"].insert(4, _beat("b05", "r#0005", 0.0, 3.0, 4, "Пять тысяч сто"))
    assert any(e.startswith("act 4 has 2") for e in _errs(two_summit_beats))


def test_times_sit_on_cut_points_or_are_snapped_onto_them():
    def rounding(d):
        d["beats"][1]["src_out"] = 47.9                     # within snap_s of 48.0
    doc = good()
    rounding(doc)
    assert bs.validate(doc, shot_info=SHOTS, msg_info=MSGS, rules=RULES) == []
    assert doc["beats"][1]["src_out"] == 48.0

    def off(d):
        d["beats"][1]["src_out"] = 45.0
    assert any("is not a cut point" in e for e in _errs(off))

    def backwards(d):
        d["beats"][1]["src_in"], d["beats"][1]["src_out"] = 48.0, 40.0
    assert "b02: src_out must be after src_in" in _errs(backwards)

    def too_long(d):
        d["beats"][2]["src_out"] = 20.0
        SHOTS["r#0003"]["cuts"].append(30.0)
        d["beats"][2]["src_out"] = 30.0
    errs = _errs(too_long)
    SHOTS["r#0003"]["cuts"].remove(30.0)
    assert any("longer than 25 s" in e for e in errs)


def test_text_must_be_verbatim_and_the_act_fixed_by_the_shot():
    def paraphrase(d):
        d["beats"][1]["text"] = "я тут умер"
    assert "b02: the text is not a verbatim part of r#0002's transcript" in _errs(paraphrase)

    def punctuation_is_not_a_difference(d):
        d["beats"][1]["text"] = "я вот, на этом курумнике прям СДОХ"
    assert _errs(punctuation_is_not_a_difference) == []

    def moved(d):
        d["beats"][1]["act"] = 2
    assert "b02: shot r#0002 is in act 3, not 2" in _errs(moved)

    def unknown(d):
        d["beats"][1]["shot_id"] = "r#0099"
    assert any("r#0099" in e for e in _errs(unknown))


def test_quotes_and_the_closing_line_come_from_the_right_phase_and_fit_a_card():
    def wrong_phase(d):
        d["beats"][4]["msg_id"] = "m4"
        d["beats"][4]["text"] = "Скучаю по горам"
    assert any("after phase, not planning" in e for e in _errs(wrong_phase))

    def too_many(d):
        for i in range(2):
            q = copy.deepcopy(d["beats"][4])
            q["beat_id"], q["msg_id"], q["text"] = f"q0{i + 2}", "m2", "ну посмотрим"
            d["beats"].append(q)
    assert any("act 1 quotes" in e for e in _errs(too_many))

    def long_card(d):
        MSGS["m2"]["text"] = "x" * 60
        d["closing"] = {"msg_id": "m2", "text": "x" * 60}
    errs = _errs(long_card)
    MSGS["m2"]["text"] = "ну посмотрим"
    assert any("closing: 60 characters" in e for e in errs)
    assert any("closing: message m2 is from the planning phase" in e for e in errs)

    def invented_closing(d):
        d["closing"]["text"] = "Скучаю по морю"
    assert "closing: the text is not a verbatim part of message m4" in _errs(invented_closing)


def test_ids_pairs_trailer_and_title():
    def dup(d):
        d["beats"][1]["beat_id"] = "b01"
    assert "beat_id b01 is used twice" in _errs(dup)

    def bad_pair(d):
        d["pairs"] = [{"planning_msg_id": "m4", "trek_beat_id": "q01", "why": ""}] * 3
    errs = _errs(bad_pair)
    assert any("3 pairs" in e for e in errs)
    assert any("not a planning-phase message" in e for e in errs)
    assert any("not one of the speech beats" in e for e in errs)

    def bad_trailer(d):
        d["trailer"]["hook_beat_id"] = "zz"
        d["trailer"]["cards"] = ["y" * 41]
    errs = _errs(bad_trailer)
    assert any("trailer.hook_beat_id" in e for e in errs)
    assert any("trailer card 0" in e for e in errs)

    def no_title(d):
        d["title"] = " "
    assert any("title" in e for e in _errs(no_title))
