"""What the beat-sheet call is given: anonymised, hallucination-free, cut-pointed."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal import db
from nepal.config import Config
from nepal.story import beats_input as bi, beats_schema

NAMES = ("Kirill Keller", "Sasha Kulikov", "Лёша Карташкин")


def test_cast_tags_by_first_appearance_and_matches_faces_to_chat_voices():
    cast = bi.Cast.build(["Sasha Kulikov", "Kirill Keller", "Sasha Kulikov", None,
                          "Лёша Карташкин"], ["keller", "kulikov", None, "dharma", "stranger"])
    assert cast.author("Sasha Kulikov") == "A" and cast.author("Kirill Keller") == "B"
    assert cast.author("Лёша Карташкин") == "C" and cast.author(None) == "?"
    assert cast.cluster("kulikov") == "A" and cast.cluster("keller") == "B"
    assert cast.cluster("dharma") == "guide" and cast.cluster(None) is None
    assert cast.cluster("stranger") == "other"            # a label never leaks
    assert cast.source("phone_keller") == "phone of B" and cast.source("camera") == "camera"
    assert cast.source("telegram") == "telegram" and cast.source(None) == "?"
    for n in NAMES:
        assert n not in json.dumps(cast.tags.values().__repr__())


def test_cut_points_are_segment_edges_and_clause_ends():
    segs = [{"start_s": 10.0, "end_s": 20.0, "text": "Йоу! Мы готовы. Да",
             "words": [{"start_s": 10.0, "end_s": 10.5, "word": "Йоу!"},
                       {"start_s": 11.0, "end_s": 11.3, "word": "Мы"},
                       {"start_s": 11.3, "end_s": 12.0, "word": "готовы."},
                       {"start_s": 12.02, "end_s": 12.4, "word": "Да"}]},
            {"start_s": 20.0, "end_s": 25.0, "text": "и всё"}]
    assert bi.cut_points(segs) == [10.0, 10.5, 11.0, 12.0, 12.02, 20.0, 25.0]
    marked = bi._marked_text(segs[0])
    assert marked == "Йоу! ⟨10.5|11⟩ Мы готовы. ⟨12⟩ Да"      # a pause shows both edges
    assert bi._marked_text(segs[1]) == "и всё"                 # no words: plain text


def _seed(tmp_path):
    cfg = Config({"project": {"data_root": str(tmp_path / "data"),
                              "work_root": str(tmp_path / "work"),
                              "db_path": str(tmp_path / "work" / "db" / "n.sqlite")},
                  "beats": {"chat_max_chars": 40}, "spine": {"card_max_chars": 180}})
    cfg.db_path.parent.mkdir(parents=True)
    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) "
                 "VALUES ('phone_keller_IMG_1', 'phone_keller', 0)")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc, alt_dem_m, "
                 "place_name) VALUES ('a1', 'k1', 'camera', 'video360', "
                 "'2024-05-03T04:00:00+00:00', 3500, 'Samagaon')")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc, alt_dem_m) "
                 "VALUES ('a2', 'k2', 'phone_keller', 'photo', '2024-05-03T05:00:00+00:00', 3600)")
    conn.execute("INSERT INTO activities(activity_id, name, start_utc, distance_m, gain_m, "
                 "hr_max, moving_s) VALUES ('act1', 'Lho to Samagaon', "
                 "'2024-05-03T02:00:00+00:00', 12100, 640, 171, 5400)")
    good = {"shot_id": "r#0001", "text": "Я вот на этом курумнике прям сдох. Серьёзно.",
            "segments": [{"start_s": 40.0, "end_s": 52.8,
                          "text": "Я вот на этом курумнике прям сдох. Серьёзно.",
                          "words": [{"start_s": 40.0, "end_s": 41.0, "word": "Я"},
                                    {"start_s": 41.0, "end_s": 48.0, "word": "сдох."},
                                    {"start_s": 48.4, "end_s": 52.8, "word": "Серьёзно."}]}]}
    bad = {"shot_id": "r#0002", "text": "Субтитры сделал DimaTorzok",
           "segments": [{"start_s": 60.0, "end_s": 62.0, "text": "Субтитры сделал DimaTorzok",
                         "hallucinated_reasons": ["phrase:Субтитры сделал"]}]}
    conn.executemany(
        "INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, start_utc, act, "
        "day_index, place_name, alt_dem_m, face_cluster, face_score, status, transcript, "
        "transcript_json, hallucinated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [("phone_keller_IMG_1#0001", "phone_keller_IMG_1", "video", 40.0, 52.8,
          "2024-05-03T04:10:00+00:00", 3, 5, "Samagaon", 3520, "keller", 0.91, "candidate",
          good["text"], json.dumps(good), 0),
         ("phone_keller_IMG_1#0002", "phone_keller_IMG_1", "video", 60.0, 62.0,
          "2024-05-03T04:11:00+00:00", 3, 5, "Samagaon", 3520, None, None, "rejected",
          bad["text"], json.dumps(bad), 1)])
    conn.executemany(
        "INSERT INTO messages(msg_id, ts_utc, author, text, phase) VALUES (?,?,?,?,?)",
        [("m1", "2024-02-11T19:02:00+00:00", "Kirill Keller",
          "20 км в день это вообще не проблема, я считаю", "planning"),
         ("m2", "2024-02-11T19:03:00+00:00", "Sasha Kulikov", "ну посмотрим " * 10, "planning"),
         ("m3", "2024-05-03T04:30:00+00:00", "Лёша Карташкин", "как вы там", "planning"),
         ("m4", "2024-05-20T10:00:00+00:00", "Sasha Kulikov", "Скучаю по горам", "after"),
         ("m5", "2024-05-20T10:01:00+00:00", "Kirill Keller", "", "after")])
    conn.commit()
    (cfg.work_root / "vocab").mkdir(parents=True)
    (cfg.work_root / "vocab" / "telegram_vocab.json").write_text(json.dumps(["Manaslu", "Larke"]))
    return cfg, conn


def test_chat_bodies_lose_contacts_mentions_and_the_authors_names():
    """The first live prompt carried an @mention, a visa email addressed by
    full name, and a contact card with a phone number and an email."""
    cast = bi.Cast.build(["Kirill Keller", "Sasha Kulikov"], [])
    s = bi.scrub_text
    assert s("@SashaKulikov шо думаешь?", cast) == "@someone шо думаешь?"
    assert s("Mariia Kulikova, +79151301705, manycan@gmail.com", cast) \
        == "Mariia B, [phone], [email]"
    assert s("Dear ALEKSANDR KULIKOV , thank you", cast) == "Dear ALEKSANDR B , thank you"
    assert s("keller и Kirill идут", cast) == "A и A идут"
    # what must survive: a price, a date, a sum, a time
    for keep in ("995+92+55 = 1142 USD", "вылет 20.04.2024 в 06:35", "20 км в день"):
        assert s(keep, cast) == keep
    # the operator's extra words
    assert s("Саша и Кирюха ушли", cast, ["Саша", "Кирюх"]) == "[name] и [name] ушли"


def test_the_prompt_names_nobody_hides_hallucinations_and_carries_the_facts_back(tmp_path):
    cfg, conn = _seed(tmp_path)
    rules = beats_schema.Rules.from_cfg(cfg).as_prompt_dict()
    p = bi.build_prompt(cfg, conn, rules=rules)
    text = p.system + p.user
    for n in NAMES + ("Keller", "Kulikov", "Карташкин", "keller", "kulikov"):
        assert n not in text, n
    assert "DimaTorzok" not in p.user                       # the hallucination is not offered
    assert "## shot S0001 |" in p.user and "S0002" not in p.user
    assert "IMG_1" not in p.user                             # the file name never travels
    assert "| phone of A |" in p.user
    assert "on screen: A (face fills the frame)" in p.user   # the first chat voice is A
    assert "[m1 2024-02-11 19:02 planning] A:" in p.user
    assert "B: ну посмотрим" in p.user and "…" in p.user      # clipped at chat_max_chars
    assert "m5" not in p.user                               # an empty message is not a line
    assert "| 1 | 2024-05-03 |" in p.user and "Lho to Samagaon | 12.1 | 640 | 1.5 | 171 |" in p.user
    assert "Manaslu, Larke" in p.user
    assert "⟨48|48.4⟩" in p.user                            # the pause after "сдох."
    assert p.meta["n_shots"] == 1 and p.meta["n_messages"] == 4 and p.meta["n_days"] == 1
    assert p.meta["shot_info"]["S0001"]["cuts"] == [40.0, 48.0, 48.4, 52.8]
    assert p.meta["shot_info"]["S0001"]["act"] == 3
    assert p.meta["shot_info"]["S0001"]["shot_id"] == "phone_keller_IMG_1#0001"
    assert p.meta["msg_info"]["m4"] == {"phase": "after", "text": "Скучаю по горам"}
    assert p.meta["acts_present"] == [3]
    assert p.messages() == [{"role": "user", "content": p.user}]
    assert "10 to 16 speech beats" in p.user               # the rules are stated once
