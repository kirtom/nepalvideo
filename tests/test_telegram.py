import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone
import json
import pytest

from nepal.spine.telegram import (Message, flatten_text, message_time, parse_export,
                                  classify_phase, mark_notable, is_card_candidate,
                                  mark_cards, extract_vocabulary, resolve_media_path)

UTC = timezone.utc


def t(h, day=15):
    return datetime(2023, 10, day, h, tzinfo=UTC)


# -- text flattening ---------------------------------------------------

def test_flatten_plain_string():
    assert flatten_text("привет") == "привет"


def test_flatten_entity_list():
    v = ["смотри ", {"type": "link", "text": "http://x"}, " вот"]
    assert flatten_text(v) == "смотри http://x вот"


def test_flatten_single_entity_dict():
    assert flatten_text({"type": "bold", "text": "важно"}) == "важно"


def test_flatten_none_and_empty():
    assert flatten_text(None) == "" and flatten_text([]) == ""


# -- timestamps --------------------------------------------------------

def test_prefers_unixtime_over_naive_date():
    """The trap: `date` is written in the exporter's local zone with no
    offset recorded, so trusting it shifts the entire chat."""
    raw = {"date": "2023-10-15T15:45:00", "date_unixtime": "1697356800"}
    got = message_time(raw)
    assert got == datetime(2023, 10, 15, 8, 0, tzinfo=UTC)
    assert got.hour != 15, "must not have trusted the zone-less local string"


def test_falls_back_to_date_when_unixtime_absent():
    assert message_time({"date": "2023-10-15T10:00:00"}) == t(10)


def test_undated_message_returns_none():
    assert message_time({}) is None


def test_malformed_unixtime_falls_through():
    raw = {"date_unixtime": "not-a-number", "date": "2023-10-15T10:00:00"}
    assert message_time(raw) == t(10)


# -- export parsing ----------------------------------------------------

EXPORT = {
    "name": "Nepal", "type": "personal_chat", "id": 1,
    "messages": [
        {"id": 1, "type": "service", "action": "create_group", "date": "2023-09-01T10:00:00"},
        {"id": 2, "type": "message", "date_unixtime": "1697356800", "from": "Keller",
         "text": "Смотри какой маршрут через Гокио"},
        {"id": 3, "type": "message", "date_unixtime": "1697360400", "from": "Kulikov",
         "text": ["Пермиты сделаем в ", {"type": "bold", "text": "Катманду"}]},
        {"id": 4, "type": "message", "date_unixtime": "1697364000", "from": "Kulikov",
         "text": "", "photo": "photos/photo_1@15-10-2023.jpg"},
        {"id": 5, "type": "message", "date_unixtime": "1697367600", "from": "Kulikov",
         "text": "", "media_type": "video_message",
         "file": "round_video_messages/video_1.mp4", "duration_seconds": 12},
        {"id": 6, "type": "message", "date": "bad", "from": "X", "text": "dropped"},
    ],
}


def test_parses_and_skips_service_messages():
    msgs = parse_export(EXPORT)
    assert [m.msg_id for m in msgs] == ["2", "3", "4", "5"]


def test_parse_sorts_by_time():
    msgs = parse_export(EXPORT)
    assert msgs == sorted(msgs, key=lambda m: m.ts_utc)


def test_media_and_type_are_captured():
    by_id = {m.msg_id: m for m in parse_export(EXPORT)}
    assert by_id["4"].media_path == "photos/photo_1@15-10-2023.jpg"
    assert by_id["5"].media_type == "video_message"
    assert by_id["5"].duration_s == 12.0
    assert by_id["5"].has_media


def test_entity_text_is_flattened_on_parse():
    by_id = {m.msg_id: m for m in parse_export(EXPORT)}
    assert by_id["3"].text == "Пермиты сделаем в Катманду"


def test_parse_from_file(tmp_path):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(EXPORT, ensure_ascii=False), encoding="utf-8")
    assert len(parse_export(p)) == 4


# -- phase -------------------------------------------------------------

def test_phase_classification():
    start, end = t(6, day=15), t(18, day=22)
    assert classify_phase(t(10, day=1), start, end) == "planning"
    assert classify_phase(t(10, day=17), start, end) == "trek"
    assert classify_phase(t(10, day=30), start, end) == "after"


def test_phase_without_a_track_is_planning():
    assert classify_phase(t(10), None, None) == "planning"


# -- notability --------------------------------------------------------

def _m(mid, hour, text="", media=None):
    return Message(str(mid), t(hour), "Keller", text, media_path=media)


def test_media_messages_are_notable():
    msgs = [_m(1, 10, "ok"), _m(2, 11, "", media="photos/x.jpg")]
    mark_notable(msgs)
    assert msgs[1].is_notable


def test_messages_near_a_climb_event_are_notable():
    msgs = [_m(1, 10, "short"), _m(2, 14, "short")]
    mark_notable(msgs, climb_events=[t(10) + timedelta(minutes=5)], window_s=1200)
    assert msgs[0].is_notable and not msgs[1].is_notable


def test_unusually_long_messages_are_notable():
    msgs = [_m(i, 10 + i, "ok") for i in range(6)]
    msgs.append(_m(99, 20, "x" * 500))
    mark_notable(msgs)
    assert msgs[-1].is_notable
    assert not msgs[0].is_notable


# -- cards -------------------------------------------------------------

@pytest.mark.parametrize("text,ok", [
    ("Пермиты сделаем в Катманду, не парься", True),
    ("День 4. Высота 4410. Ноги отваливаются.", True),
    ("https://example.com/route.gpx", False),      # a link is not a line
    ("👍", False),                                   # bare emoji
    ("ok", False),                                  # too short
    ("Да", False),                                  # single word
    ("2023", False),                                # bare number
    ("x" * 400, False),                             # too long to burn in
])
def test_card_candidates(text, ok):
    assert is_card_candidate(text) is ok


def test_mark_cards_counts_and_sets():
    msgs = [_m(1, 10, "Пермиты сделаем в Катманду"), _m(2, 11, "👍")]
    assert mark_cards(msgs) == 1
    assert msgs[0].usable_as_card and not msgs[1].usable_as_card


# -- vocabulary --------------------------------------------------------

def test_vocabulary_picks_up_recurring_proper_nouns():
    msgs = [_m(i, 10, f"сегодня были в Намче и видели Куликова") for i in range(5)]
    vocab = extract_vocabulary(msgs, min_count=3)
    assert "Намче" in vocab
    assert "Куликова" in vocab


def test_vocabulary_excludes_stopwords_and_rare_terms():
    msgs = [_m(i, 10, "это просто как обычно") for i in range(5)]
    msgs.append(_m(99, 12, "Уникальный Случай"))
    vocab = extract_vocabulary(msgs, min_count=3)
    assert "это" not in vocab and "просто" not in vocab
    assert "Уникальный" not in vocab, "appears once, not a recurring term"


def test_vocabulary_on_empty_input():
    assert extract_vocabulary([]) == []


# -- media resolution --------------------------------------------------

def test_resolve_media_path(tmp_path):
    (tmp_path / "photos").mkdir()
    f = tmp_path / "photos" / "photo_1.jpg"
    f.write_bytes(b"x")
    assert resolve_media_path("photos/photo_1.jpg", tmp_path) == f


def test_resolve_media_path_by_basename_fallback(tmp_path):
    (tmp_path / "round_video_messages").mkdir()
    f = tmp_path / "round_video_messages" / "v.mp4"
    f.write_bytes(b"x")
    assert resolve_media_path("v.mp4", tmp_path) == f


def test_resolve_media_path_missing(tmp_path):
    assert resolve_media_path("photos/absent.jpg", tmp_path) is None
