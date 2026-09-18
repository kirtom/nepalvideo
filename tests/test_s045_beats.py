"""S04.5 end to end against recorded answers: a wrong sheet, then a right one."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db
from nepal.cloud import claude as cl, spend
from nepal.config import Config
from nepal.stages import s045_beats

NAMES = ("Kirill Keller", "Sasha Kulikov", "Keller", "Kulikov")


def _cfg(tmp_path, **beats):
    return Config({
        "project": {"data_root": str(tmp_path / "data"), "work_root": str(tmp_path / "work"),
                    "db_path": str(tmp_path / "work" / "db" / "n.sqlite")},
        "spine": {"card_max_chars": 180},
        "cloud": {"spend_ceiling_usd": 25},
        "api": {"prices_usd_per_mtok": {"claude-opus-5": {"input": 5, "output": 25}}},
        "beats": {"model": "claude-opus-5", "effort": "high", "max_output_tokens": 4000,
                  "min_speech_beats": 2, "max_speech_beats": 3, "max_speech_s": 25,
                  "min_act1_quotes": 1, "max_act1_quotes": 2, "max_act4_beats": 1,
                  "max_pairs": 2, "cut_tolerance_s": 0.05, "snap_s": 0.5, "retries": 1,
                  "max_input_tokens": 200000, "estimate_usd_cap": 3.0,
                  "estimate_chars_per_token": 3.0, "chat_max_chars": 500, **beats}})


def _seed(cfg):
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) VALUES ('r', 'camera', 0)")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc, alt_dem_m) "
                 "VALUES ('a1', 'k1', 'camera', 'video360', '2024-05-03T04:00:00+00:00', 3500)")
    docs = {
        "r#0001": {"start_s": 0.0, "end_s": 10.0, "act": 2, "utc": "2024-04-28T04:00:00+00:00",
                   "text": "Мы приехали. Тут жарко.",
                   "words": [{"start_s": 0.0, "end_s": 4.9, "word": "приехали."},
                             {"start_s": 5.0, "end_s": 10.0, "word": "жарко."}]},
        "r#0002": {"start_s": 40.0, "end_s": 52.8, "act": 3, "utc": "2024-05-02T04:00:00+00:00",
                   "text": "Я вот на этом курумнике прям сдох. Серьёзно.",
                   "words": [{"start_s": 40.0, "end_s": 48.0, "word": "сдох."},
                             {"start_s": 48.4, "end_s": 52.8, "word": "Серьёзно."}]},
    }
    for sid, d in docs.items():
        doc = {"shot_id": sid, "text": d["text"],
               "segments": [{"start_s": d["start_s"], "end_s": d["end_s"], "text": d["text"],
                             "words": d["words"]}]}
        conn.execute(
            "INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, start_utc, act, "
            "status, transcript, transcript_json, hallucinated) VALUES (?,?,?,?,?,?,?,?,?,?,0)",
            (sid, "r", "video", d["start_s"], d["end_s"], d["utc"], d["act"], "candidate",
             d["text"], json.dumps(doc)))
    conn.executemany(
        "INSERT INTO messages(msg_id, ts_utc, author, text, phase) VALUES (?,?,?,?,?)",
        [("m1", "2024-02-11T19:02:00+00:00", "Kirill Keller", "20 км в день не проблема", "planning"),
         ("m4", "2024-05-20T10:00:00+00:00", "Sasha Kulikov", "Скучаю по горам", "after")])
    conn.commit()
    conn.close()


def _sheet(text2="Я вот на этом курумнике прям сдох"):
    return json.dumps({
        "title": "Перевал", "beats": [
            {"beat_id": "b01", "kind": "speech", "act": 2, "shot_id": "r#0001", "msg_id": None,
             "src_in": 0.0, "src_out": 4.9, "text": "Мы приехали", "levity": True,
             "effect": "none", "rank": 2, "rationale": "arrival"},
            {"beat_id": "b02", "kind": "speech", "act": 3, "shot_id": "r#0002", "msg_id": None,
             "src_in": 40.0, "src_out": 48.0, "text": text2, "levity": True,
             "effect": "freeze", "rank": 1, "rationale": "the cost"},
            {"beat_id": "q01", "kind": "quote", "act": 1, "shot_id": None, "msg_id": "m1",
             "src_in": None, "src_out": None, "text": "20 км в день не проблема",
             "levity": True, "effect": "none", "rank": 3, "rationale": "confidence"}],
        "pairs": [{"planning_msg_id": "m1", "trek_beat_id": "b02", "why": "20 km a day"}],
        "closing": {"msg_id": "m4", "text": "Скучаю по горам"},
        "act_notes": [{"act": 3, "note": "slower"}], "stat_card_ideas": ["gain"],
        "trailer": {"hook_beat_id": "b02", "cliffhanger_beat_id": "b01", "cards": ["5,100 m"]}},
        ensure_ascii=False)


def test_a_wrong_sheet_is_sent_back_once_and_the_right_one_is_written(tmp_path):
    cfg = _cfg(tmp_path)
    _seed(cfg)
    fake = cl.FakeClaude([_sheet(text2="я тут умер"), _sheet()], n_tokens=50_000)
    rep = s045_beats.run(cfg, claude=fake)
    assert rep["attempts"] == 2 and rep["errors"] == []
    assert rep["n_speech"] == 2 and rep["n_quotes"] == 1 and rep["n_pairs"] == 1
    assert rep["per_act"] == {"2": 1, "3": 1} and rep["title"] == "Перевал"
    # the second call carried the answer and the complaint
    second = fake.calls[-1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "not a verbatim part of r#0002" in second[-1]["content"]
    assert fake.calls[-1]["schema"]["required"][:2] == ["title", "beats"]
    # both calls are on the ledger
    led = spend.ledger(cfg)
    assert len(led.entries) == 2 and all(e.what == "beats" for e in led.entries)
    assert rep["usd"] == round(sum(e.usd for e in led.entries), 4)
    # the rows, every kind
    conn = db.init(cfg.db_path)
    rows = {r["beat_id"]: dict(r) for r in conn.execute("SELECT * FROM story_beats")}
    assert set(rows) == {"b01", "b02", "q01", "closing", "title"}
    assert rows["b02"]["src_out"] == 48.0 and rows["b02"]["levity"] == 1
    assert rows["closing"]["msg_id"] == "m4" and rows["title"]["text"] == "Перевал"
    # the files, with nobody named
    sheet = json.loads((cfg.work_root / "beats" / "beats.json").read_text())
    assert sheet["meta"]["attempts"] == 2 and sheet["sheet"]["title"] == "Перевал"
    page = (cfg.work_root / "gates" / "gate2" / "index.html").read_text()
    prompt = (cfg.work_root / "beats" / "prompt.txt").read_text()
    for n in NAMES:
        assert n not in page and n not in prompt
    assert "проксик" not in page and "r_eq.mp4#t=40.0" in page
    assert (cfg.work_root / "beats" / "response_2.json").exists()
    # a second run without --force does not call again
    rep2 = s045_beats.run(cfg, claude=cl.FakeClaude([]))
    assert rep2["skipped"].startswith("5 beats exist")


def test_a_dry_run_writes_the_prompt_and_calls_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    _seed(cfg)
    fake = cl.FakeClaude([], n_tokens=70_000)
    rep = s045_beats.run(cfg, dry_run=True, claude=fake)
    assert rep["dry_run"] and rep["input_tokens"] == 70_000 and rep["tokens_how"] == "counted"
    assert rep["estimate_usd"] == cl.estimate_usd(70_000, 4000, cl.Price(5, 25))
    assert [c["op"] for c in fake.calls] == ["count_tokens"]
    assert (cfg.work_root / "beats" / "prompt.txt").exists()
    assert not (cfg.work_root / "beats" / "beats.json").exists()
    assert not spend.ledger(cfg).entries


def test_the_two_refusals_happen_before_any_call(tmp_path):
    cfg = _cfg(tmp_path, estimate_usd_cap=0.01)
    _seed(cfg)
    fake = cl.FakeClaude([_sheet()], n_tokens=50_000)
    with pytest.raises(s045_beats.BeatsRefused):
        s045_beats.run(cfg, claude=fake)
    assert len(fake.calls) == 1                       # counted, never completed
    cfg = _cfg(tmp_path, max_input_tokens=100)
    with pytest.raises(s045_beats.BeatsRefused):
        s045_beats.run(cfg, claude=cl.FakeClaude([_sheet()], n_tokens=50_000))
    # the ledger ceiling is the third guard
    cfg = _cfg(tmp_path)
    spend.ledger(cfg).record("remote", 24.99)
    with pytest.raises(spend.SpendCeiling):
        s045_beats.run(cfg, claude=cl.FakeClaude([_sheet()], n_tokens=50_000))


def test_a_sheet_still_wrong_after_the_retry_writes_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    _seed(cfg)
    fake = cl.FakeClaude([_sheet(text2="нет"), _sheet(text2="тоже нет")])
    with pytest.raises(s045_beats.BeatsInvalid) as exc:
        s045_beats.run(cfg, claude=fake)
    assert exc.value.errors and exc.value.path.name == "response_2.json"
    conn = db.init(cfg.db_path)
    assert conn.execute("SELECT COUNT(*) FROM story_beats").fetchone()[0] == 0
    assert len(spend.ledger(cfg).entries) == 2         # paid for, and recorded


def test_tokens_are_estimated_when_they_cannot_be_counted(tmp_path):
    cfg = _cfg(tmp_path)
    _seed(cfg)

    class NoKey(cl.FakeClaude):
        def count_tokens(self, system, messages):
            raise RuntimeError("no key")
    rep = s045_beats.run(cfg, dry_run=True, claude=NoKey([]))
    assert rep["tokens_how"] == "estimated" and rep["input_tokens"] > 100
