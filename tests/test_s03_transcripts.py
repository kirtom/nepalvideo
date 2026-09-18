"""S03.5's bookkeeping around the transcripts: segments in the row, and the
hallucination filter beside the gate. No model, no media."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal import db
from nepal.config import Config
from nepal.stages import s03_process


def _cfg(tmp_path):
    return Config({"project": {"data_root": str(tmp_path / "data"),
                               "work_root": str(tmp_path / "work"),
                               "db_path": str(tmp_path / "work" / "db" / "n.sqlite")}})


def _db_with_shots(cfg, rows):
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) VALUES ('r', 'camera', 0)")
    for r in rows:
        conn.execute("INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, "
                     "transcript, speech_s, has_speech, status) VALUES (?,?,?,?,?,?,?,?,?)",
                     (r["shot_id"], "r", "video", r["start_s"], r["end_s"],
                      r.get("transcript"), r.get("speech_s", 5.0), 1, "candidate"))
    conn.commit()
    return conn


def test_backfill_reads_the_file_when_there_is_one_and_invents_a_segment_otherwise(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _db_with_shots(cfg, [
        {"shot_id": "r#0001", "start_s": 0.0, "end_s": 10.0, "transcript": "раз два"},
        {"shot_id": "r#0002", "start_s": 10.0, "end_s": 20.0, "transcript": "три"},
        {"shot_id": "r#0003", "start_s": 20.0, "end_s": 30.0, "transcript": None}])
    tdir = cfg.workdir("transcripts", "shots")
    (tdir / "r_0001.json").write_text(json.dumps(
        {"shot_id": "r#0001", "text": "раз два",
         "segments": [{"start_s": 1.0, "end_s": 4.0, "text": "раз"},
                      {"start_s": 4.0, "end_s": 8.0, "text": "два"}]}))
    rep = s03_process.backfill_transcript_json(cfg, conn)
    assert rep == {"n_rows": 2, "from_files": 1, "synthetic": 1}
    got = {r["shot_id"]: r["transcript_json"] for r in
           conn.execute("SELECT shot_id, transcript_json FROM shots")}
    assert len(json.loads(got["r#0001"])["segments"]) == 2
    synth = json.loads(got["r#0002"])
    assert synth["synthetic"] and synth["segments"] == [
        {"start_s": 10.0, "end_s": 20.0, "text": "три"}]
    assert got["r#0003"] is None                       # nothing to fill
    # idempotent: a second pass finds nothing to do
    assert s03_process.backfill_transcript_json(cfg, conn) == {"n_rows": 0}
