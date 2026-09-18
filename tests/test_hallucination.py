"""The hallucination filter: deterministic, reads what is stored."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.config import Config
from nepal.process import hallucination as h


def test_a_subtitle_credit_is_a_hallucination_whatever_its_case():
    assert h.segment_reasons({"text": "субтитры сделал DimaTorzok"}) == ["phrase:Субтитры сделал"]
    assert h.segment_reasons({"text": "Продолжение следует..."}) == ["phrase:Продолжение следует"]


def test_a_looped_phrase_is_a_hallucination_and_a_repeated_word_is_not():
    loop = "ну вот так ну вот так ну вот так и всё"
    assert h.repeated_ngram(loop) == "ну вот так"
    assert h.segment_reasons({"text": loop}) == ["loop:ну вот так"]
    assert h.repeated_ngram("да да да да да, пошли") is None      # words, not phrases
    assert h.segment_reasons({"text": "Я вот на этом курумнике прям сдох"}) == []


def test_whisper_confidence_is_read_where_it_was_stored():
    assert h.segment_reasons({"text": "хм", "no_speech_prob": 0.91}) == ["no_speech_prob:0.91"]
    assert h.segment_reasons({"text": "хм", "compression_ratio": 3.1}) == ["compression_ratio:3.10"]
    assert h.segment_reasons({"text": "хм", "no_speech_prob": 0.2, "compression_ratio": 1.1}) == []
    # a transcript from before the numbers were kept is judged on its text
    assert h.segment_reasons({"text": "нормальная фраза"}) == []


def test_a_shot_is_hallucinated_only_when_every_segment_is():
    segs = [{"text": "Спасибо за просмотр"}, {"text": "мы почти на перевале"}]
    flag, why = h.shot_hallucinated(segs, speech_s=6.0, speech_min_s=0.6)
    assert flag is False and why == [["phrase:Спасибо за просмотр"], []]
    flag, why = h.shot_hallucinated(segs[:1], speech_s=6.0, speech_min_s=0.6)
    assert flag is True and why == [["phrase:Спасибо за просмотр"]]


def test_text_over_a_window_the_vad_heard_nothing_in_is_a_hallucination():
    segs = [{"text": "совершенно нормальная фраза"}]
    flag, why = h.shot_hallucinated(segs, speech_s=0.1, speech_min_s=0.6)
    assert flag is True and why == [["no_vad_speech"]]
    # no segments at all is silence, not hallucination
    assert h.shot_hallucinated([], speech_s=6.0, speech_min_s=0.6) == (False, [])


def test_rules_come_from_the_config_block():
    cfg = Config({"asr": {"hallucination_phrases": ["Корректор"], "max_no_speech_prob": 0.5,
                          "max_compression_ratio": 2.0, "ngram_n": 2, "ngram_min_repeats": 4}})
    rules = h.rules_from_cfg(cfg)
    assert rules["phrases"] == ("Корректор",) and rules["ngram_n"] == 2
    assert h.segment_reasons({"text": "Корректор А. Егорова"}, **rules) == ["phrase:Корректор"]
    assert h.segment_reasons({"text": "Субтитры сделал"}, **rules) == []   # not in this list
