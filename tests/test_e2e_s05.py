"""End-to-end S05 on the seeded database with real media: score, timeline,
cues and the draft with sound, through ``s05_cut.run(cfg)`` exactly as
``nepal cut`` runs it. Slow: builds lavfi proxies, wavs and music, then
renders twice (the measurement pass and the draft). Run with: pytest -m slow

Every recording gets the same tiny proxy and the same wav, every still the
same frame: nothing here reads the picture, and what is under test is the
wiring -- that every slot and every cue finds its file, that the two ffmpeg
passes run on what Python asked for, and that the draft comes back as long
as the timeline says.
"""
import json
import shutil
import subprocess
import sys, pathlib
from datetime import datetime, timedelta

import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_s05_timeline import _cfg, _seed
from nepal import db
from nepal.stages import s05_cut

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")]


def _lavfi(src, path, *extra):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", src, *extra, str(path)], check=True)


def _probe(path, stream, entry):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
                           f"stream={entry}", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_s05_runs_end_to_end_and_the_draft_hears_itself(tmp_path):
    cfg = _cfg(tmp_path)
    # A small frame: the arithmetic is the same at any size, the encode is not.
    cfg._data["render"] = {**cfg.get("render"), "draft": {"width": 320, "height": 180, "crf": 30}}
    conn = _seed(cfg)
    # Keyed under raw/ as the ingest writes them, so the draft resolves the
    # local file the way it will on the real corpus.
    conn.execute("UPDATE music_tracks SET s3_key = 'raw/music/' || track_id || '.wav'")
    conn.commit()
    recs = [(r[0], float(r[1])) for r in conn.execute("SELECT recording_id, duration_s FROM recordings")]
    photos = [r[0] for r in conn.execute("SELECT shot_id FROM shots WHERE media_kind = 'photo'")]
    tracks = [(r[0], float(r[1])) for r in conn.execute("SELECT s3_key, duration_s FROM music_tracks")]
    conn.close()

    fx = tmp_path / "fixtures"
    fx.mkdir()
    longest = max(d for _, d in recs)
    # At the draft's own rate, so a leg of N seconds is exactly N*fps frames
    # and the concat's length is the timeline's, not the timeline's plus a
    # rounding per cut.
    _lavfi(f"testsrc=size=64x36:rate={int(cfg.get('render.fps'))}:duration={longest:.0f}", fx / "proxy.mp4",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")
    # Two seconds of tone, two of silence, over and over: every cue has
    # something to measure and something not to. Mono, as S03 writes them.
    _lavfi(f"aevalsrc=0.2*sin(2*PI*440*t)*(1+sgn(sin(PI*t/2)))/2:s=48000:d={longest:.0f}",
           fx / "tone.wav", "-ac", "1")
    _lavfi("testsrc=size=64x36:rate=1", fx / "still.jpg", "-frames:v", "1")
    _lavfi(f"sine=frequency=220:sample_rate=48000:duration={max(d for _, d in tracks):.0f}",
           fx / "music.wav", "-ac", "2")
    for rid, _ in recs:
        shutil.copyfile(fx / "proxy.mp4", cfg.work("proxies", f"{rid}_eq.mp4"))
        shutil.copyfile(fx / "tone.wav", cfg.work("audio", f"{rid}.wav"))
    for sid in photos:
        shutil.copyfile(fx / "still.jpg", cfg.work("stills", f"{sid}.jpg"))
    for key, _ in tracks:
        dst = cfg.data_root / key.removeprefix("raw/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fx / "music.wav", dst)

    rep = s05_cut.run(cfg)
    draft = rep["draft"]
    assert "error" not in draft and "skipped" not in draft, draft
    out = pathlib.Path(draft["path"])
    assert out.exists() and draft["n_skipped"] == 0
    assert draft["audio_tracks"] == 3 and draft["n_dropped_cues"] == 0 and draft["two_pass"]
    assert all(draft["n_cues"][t] >= 1 for t in ("speech", "location", "music")), draft["n_cues"]
    assert _probe(out, "a:0", "codec_type") == "audio", "the draft has a sound track"

    conn = db.init(cfg.db_path)
    end = float(conn.execute("SELECT MAX(t_out) FROM timeline").fetchone()[0])
    first = conn.execute(
        "SELECT t.src_in, s.start_utc, s.start_s FROM timeline t JOIN shots s ON s.shot_id = t.shot_id "
        "WHERE t.kind = 'video' ORDER BY t.slot_index LIMIT 1").fetchone()
    n_cues = conn.execute("SELECT COUNT(*) FROM audio_cues").fetchone()[0]
    conn.close()
    dur = float(_probe(out, "v:0", "duration"))
    assert abs(dur - end) <= 1.0, f"the draft runs {dur:.2f} s against a timeline of {end:.2f} s"
    assert abs(float(draft["draft_s"]) - dur) < 0.1
    assert n_cues == sum(draft["n_cues"].values())

    page = (cfg.work_root / "gates" / "gate3" / "index.html").read_text()
    when = datetime.fromisoformat(first["start_utc"]) + timedelta(
        seconds=float(first["src_in"]) - float(first["start_s"] or 0.0), hours=5, minutes=45)
    assert when.strftime("%Y-%m-%d %H:%M:%S") in page, "the first slot's wall clock, on Nepal's clock"
    assert "Kulikov" not in page, "the chat author is a tag on the page, never a name"

    otio = json.loads((cfg.work_root / "timeline.otio").read_text())
    names = [t["name"] for t in otio["tracks"]["children"]]
    assert names == ["V1", "speech", "location", "music", "overlays"], names
    assert (cfg.work_root / "gates" / "gate3" / "draft.filters").exists()
