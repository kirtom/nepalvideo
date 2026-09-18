# Film v2 — Step 4: schema v2, assembly v2 and the audio graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The draft is cut around the beat sheet instead of around scores: every speech beat is an audio cue with its own utterance, picture is filled between the beats by MMR with a deterministic similarity fallback, the two phones are balanced and paired, one bridge crossing runs unbroken, slot lengths follow the music's energy, the music is chosen per scene, and the draft has sound — speech, location and music on three tracks with the envelopes the spec prescribes. The film aims at 25 minutes and may grow to 40. Everything before this step's draft is watchable at Gate 3 with a shot list and a per-act source ratio.

**Architecture:** Pure modules do the deciding — `story/anchors.py` (beats → anchors), `process/assemble.py` (fill, constraints, fallback similarity), `process/pairs.py`, `process/longtake.py`, `process/rhythm.py`, `spine/scenes.py` (scenes and their targets), `spine/music.py` (Viterbi over scenes), `process/cues.py` (audio cues and overlay rows), `process/mix.py` (envelopes), `process/render.py` (the ffmpeg graph as data). `stages/s05_cut.py` stays the one stage (`S05`, sub-steps `score`, `timeline`, `cues`, `draft`) and writes the v2 tables, `music_map.json` in its existing shape, the OTIO/FCPXML, the draft and the Gate 3 page. Runs on the box: `nepal remote run cut`.

**Tech Stack:** Python 3.10+, numpy, scipy (existing), SQLite, ffmpeg (`amix`, `loudnorm`, `volume` expressions, `acrossfade`, `hstack`, `-filter_complex_script`), pytest with a seeded database and `ffmpeg -f lavfi` fixtures for the slow tests.

**Spec:** `docs/superpowers/specs/2026-09-16-film-v2-voice-spine-design.md` §1.1–1.4, §3.3, §5 (all of it), §6.1 (the `split` motion only), §7, §8.9, §9 (Gate 3), §10, §11 step 4, §13.5, §13.11 items 4 and 5.

## Global Constraints

- **Nothing computes on the local machine.** Pure single-file tests may be run through the main checkout's interpreter with `PYTHONPATH=src /data/projects/nepalvideo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/<one file>` (seconds); `-m slow` tests, the whole suite and every `nepal cut` run on the box (`nepal remote up`, `nepal remote run cut`, `nepal remote pull`, `nepal remote down`).
- **Every tunable lives in `config/pipeline.yaml`.** No new top-level blocks are needed: `film:`, `acts:`, `assemble:`, `render:`, `music:`, `beats:` all exist. Duplicate top-level keys raise, so never add a second block with an existing name. Twelve keys already declared and read by nothing (`film.photo_share`, `film.photo_share_by_act`, `film.cold_open_s`, `assemble.subject_shot_every_s`, `assemble.levity_min_per_act`, `assemble.act4_held_shot_s`, `assemble.natural_sound_windows`, `assemble.natural_sound_window_s`, `assemble.max_reassembly_loops`, `render.music_lufs`, `render.duck_lufs`, `render.duck_lufs_speech`, `music.preferred_tracks`, `music.preferred_bonus`) are wired by this step rather than duplicated.
- **The voice-spine table is `story_beats`** (step 3's amendment of spec §3.2); `timeline.beat_id` and `audio_cues.beat_id` reference it. The music grid stays `beats(track_id, t_s, is_downbeat)`.
- **No author names in anything that leaves the machine.** Chat cards carry `A`/`B`/`C`; message bodies are not masked (operator's decision, 2026-09-18). Gate pages are local.
- **A slot never claims more footage than its shot has** (`assemble.shot_available_s`); **a beat's audio is never cut by the rhythm pass.**
- **Six acts land in step 7.** This step keeps the five-row `acts` table and the spine's five acts; Act 0 (the cold open) is built by assembly from Act 3/4 material and is not an `acts` row. The length decision of §13.11.5 (`film.target_duration_s` 1500, `film.growth_bias` 0.6) is made here because S05 now allocates the acts from the material; the sixth row scales the same table when the return boundary exists.
- **Per-act source share** (§13.11.4): each phone gets at least `assemble.source_share_min` (0.25) of an act's phone slots where it has material; the status page and Gate 3 show slots versus available per source per act.
- **Draft-only placeholders are allowed to use `drawtext`** (a card slot's text, the slot label), because S06.5 (step 6) renders the real overlays. `overlays` rows are written now with their payloads so step 6 has nothing to invent.
- Commit after every task, message ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M`. Commits are made through a script file (the worktree guard refuses inline multi-command git).

**Facts fixed on 2026-09-18 (from the explorer's read of HEAD 5bd8f26):** none of step 4 exists. `timeline` is v1 (10 columns, `msg_id`/`transition` never written); there are no `audio_cues`/`overlays` tables; `story_beats` is read by nothing; `assemble.mmr_select` uses CLIP cosine and returns 0.0 similarity when an embedding is missing (no embeddings exist yet — CLIP is step 5, so the fallback is the only similarity this step will ever see); `place_count_ok` is the only constraint wired; `needs_subject`/`missing_levity`/`select.plan_photo_slots` exist and are unused; S06/S07 are functions inside `s05_cut.py` recorded under stage `S05`; the draft is silent (`concat … a=0`, `music_path` never passed); `process/mix.py` is orphaned; `timeline_io.FPS` is 25 while `render.fps` is 30; `music_map.json` is written once by S02.7 with per-act Hungarian assignment and the beat grids **on the film timeline** while S05 reads the `beats` table **on the track's clock**. The beat sheet on the box: 15 speech beats (acts 2/3/4/5: 3/7/1/4), 7 Act 1 quotes, 5 pairs, a closing line, effects `freeze`/`burst`/`ramp` on four beats, trailer hook b17 and cliffhanger b16. Shots: 1,550 surviving, 82/77/884/151/347 per act 1–5. Sources on `assets.source`: `camera | phone_keller | phone_kulikov | telegram`; video shots reach their source through `recordings.source`, photos through `assets.source`. Extracted audio: `work/audio/<recording_id>.wav`, mono PCM (draft quality). Proxies: `work/proxies/<recording_id>_eq.mp4`; stills: `work/stills/<shot_id>.jpg`. `effort.hardest_windows(prof, count, window_s, min_gap_s)` and `effort.at(prof, ts)` exist; `gps.interpolate_at` exists.

## File structure

| File | Responsibility |
|---|---|
| `config/pipeline.yaml` | `film.target_duration_s` 1500, `growth_bias` 0.6, `cold_open_s: [15, 25]`, `cold_open_card`; `beats.face_hold_s/pre_roll_s/broll_window_s`; `assemble.*` fill, pair, long-take, rhythm, share keys; `music.*` scene keys; `render.*` audio keys |
| `src/nepal/db.py` | `timeline` v2 (rebuilt: it is derived data), `audio_cues`, `overlays` |
| `src/nepal/process/assemble.py` | fallback similarity, `mmr_select(similarity=…)`, recording-run and source-alternation constraints, source-share repair, `lay_out` with `kind` |
| `src/nepal/story/anchors.py` (new) | beats → `Anchor`s: where they sit in the act, own picture vs B-roll, chat-card quotes, the cold open pick |
| `src/nepal/process/pairs.py` (new) | two phones, one moment |
| `src/nepal/process/longtake.py` (new) | the bridge crossing |
| `src/nepal/process/rhythm.py` (new) | section energy → slot lengths, bursts, the Act 4 held shot |
| `src/nepal/spine/scenes.py` (new) | slots → scenes with attributes and targets |
| `src/nepal/spine/music.py` | `assign_scenes` (Viterbi), `music_map_from_scenes` (existing shape), `preferred_tracks` bonus |
| `src/nepal/process/cues.py` (new) | `audio_cues` rows (speech, location, music) and `overlays` rows (chat cards, closing card, cold-open card) |
| `src/nepal/process/mix.py` (rewritten) | per-track envelopes from cues and windows; `volume` expressions |
| `src/nepal/process/render.py` | the v2 graph: per-slot video incl. `split` and `card`, three audio tracks, `amix`, `loudnorm`; `-filter_complex_script` |
| `src/nepal/process/timeline_io.py` | `fps` from config; three audio tracks and an overlays track in OTIO |
| `src/nepal/stages/s05_cut.py` | `timeline` v2 (durations, cold open, anchors, pairs, long take, fill, scenes → music map, rhythm), `cues`, `draft` with sound, Gate 3 page |
| `src/nepal/story/gate3.py` (new) | the Gate 3 page: slot list by wall clock, cues, overlays, per-act source ratio |
| `src/nepal/cloud/status.py` | per-act source ratio table |
| `tests/test_db.py`, `test_assemble.py`, `test_anchors.py`, `test_pairs.py`, `test_longtake.py`, `test_rhythm.py`, `test_scenes.py`, `test_music.py`, `test_cues.py`, `test_mix.py`, `test_render.py`, `test_s05_timeline.py`, `test_gate3.py`, `test_cloud_status.py`, `test_e2e_s05.py` (slow) | as named per task |

**Slot dict — the one shape every module passes around** (keys exact; `None` where empty, never omitted, because `db.upsert` takes its columns from the first row):

```python
SLOT_KEYS = ("slot_index", "act", "t_in", "t_out", "kind", "shot_id", "src_in", "src_out",
             "secondary_shot_id", "secondary_src_in", "motion", "speed", "transition",
             "beat_id", "scene_id", "msg_id", "yaw", "locked")
# kind: video | photo | card | black ; motion: JSON text or None ; speed: 1.0 ;
# transition: cut | dissolve | dip_black ; locked: 1 when the rhythm pass may not move
# the slot's bounds (a beat's own picture, the long take, the held shot) -- not a column.
```

`Anchor` (dataclass, `story/anchors.py`):

```python
@dataclass(frozen=True)
class Anchor:
    beat_id: str
    act: int
    kind: str            # speech | quote | closing
    utc: str | None      # the beat's moment (shot start_utc + src_in) for chronology
    recording_id: str | None
    shot_id: str | None
    src_in: float        # on the recording's clock, pre-roll already applied
    src_out: float
    duration_s: float    # src_out - src_in
    own_picture: bool    # True: the picture stays on the beat's recording (walking shot)
    face_hold_s: float   # seconds of the speaker's own shot before B-roll (0 when own_picture)
    text: str
    effect: str          # none | freeze | burst | ramp
    t_in: float = -1.0   # on the film timeline, set by place_anchors
```

---

## Part A — picture

### Task 1: Schema v2 and the config keys

**Files:**
- Modify: `src/nepal/db.py` (SCHEMA after `story_beats`; a rebuild helper beside `_rebuild_shots_for_photo_slots`)
- Modify: `config/pipeline.yaml` (`film:`, `beats:`, `assemble:`, `music:`, `render:`)
- Test: `tests/test_db.py`, `tests/test_config_paths.py` (or a new `tests/test_config_step4.py`)

**Interfaces:**
- Produces: tables `timeline` (v2 columns below), `audio_cues`, `overlays`; `db.TIMELINE_V2_COLUMNS: tuple[str, ...]`; config keys listed below.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py (append)
def test_a_fresh_db_has_the_picture_and_audio_tables(tmp_path):
    conn = db.init(tmp_path / "n.sqlite")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(timeline)")}
    assert {"kind", "secondary_shot_id", "secondary_src_in", "motion", "speed",
            "transition", "beat_id", "scene_id"} <= cols
    assert {r[1] for r in conn.execute("PRAGMA table_info(audio_cues)")} >= {
        "cue_id", "track", "t_in", "t_out", "source", "src_in", "src_out",
        "gain_lufs", "fade_in_s", "fade_out_s", "beat_id"}
    assert {r[1] for r in conn.execute("PRAGMA table_info(overlays)")} >= {
        "overlay_id", "kind", "t_in", "t_out", "payload", "asset_path"}


def test_an_older_timeline_is_rebuilt_because_it_is_derived(tmp_path):
    """timeline is DELETEd and rewritten by every S05 run; a v1 table is
    dropped and recreated rather than migrated column by column."""
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE timeline (slot_index INTEGER PRIMARY KEY, act INTEGER, "
              "shot_id TEXT, msg_id TEXT, t_in REAL, t_out REAL, src_in REAL, src_out REAL, "
              "yaw REAL, transition TEXT)")
    c.execute("INSERT INTO timeline(slot_index, act) VALUES (0, 1)")
    c.commit(); c.close()
    conn = db.init(p)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(timeline)")}
    assert "kind" in cols and "beat_id" in cols
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 0
```

```python
# tests/test_config_step4.py (new)
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `PYTHONPATH=src /data/projects/nepalvideo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_db.py tests/test_config_step4.py`
Expected: FAIL — missing columns/tables, missing keys.

- [ ] **Step 3: The schema**

In `src/nepal/db.py` replace the `timeline` DDL with the v2 shape (spec §3.3 plus `scene_id`, `msg_id` and `yaw`, which the existing renderer and the card slots use) and add the two tables after `story_beats`:

```sql
-- Film v2 section 3.3: the picture track. Rebuilt, not migrated: S05 deletes
-- and rewrites it on every run, so an older shape carries nothing worth keeping.
CREATE TABLE IF NOT EXISTS timeline (
  slot_index  INTEGER PRIMARY KEY,
  act         INTEGER,
  t_in REAL, t_out REAL,
  kind        TEXT NOT NULL,                 -- video | photo | card | black
  shot_id     TEXT REFERENCES shots(shot_id),
  src_in REAL, src_out REAL,
  secondary_shot_id TEXT,                    -- split screen: the other phone's clip
  secondary_src_in  REAL,
  motion      TEXT,                          -- JSON: {"type":"split"} now; step 6 adds the rest
  speed       REAL DEFAULT 1.0,
  transition  TEXT,                          -- cut | dissolve | dip_black
  beat_id     TEXT REFERENCES story_beats(beat_id),
  scene_id    INTEGER,                       -- the music scene the slot belongs to
  msg_id      TEXT REFERENCES messages(msg_id),   -- a card slot's message
  yaw         REAL
);

-- the audio tracks
CREATE TABLE IF NOT EXISTS audio_cues (
  cue_id      TEXT PRIMARY KEY,
  track       TEXT NOT NULL,                 -- speech | location | music
  t_in REAL, t_out REAL,
  source      TEXT,                          -- recording_id or track_id
  src_in REAL, src_out REAL,
  gain_lufs   REAL,
  fade_in_s REAL DEFAULT 0, fade_out_s REAL DEFAULT 0,
  beat_id     TEXT REFERENCES story_beats(beat_id)
);

-- generated pictures laid over the picture track (rendered by S06.5, step 6)
CREATE TABLE IF NOT EXISTS overlays (
  overlay_id  TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,                 -- hud_map | hud_day | place_card | profile | chat_card | stat_card | subtitle
  t_in REAL, t_out REAL,
  payload     TEXT,                          -- JSON the renderer needs
  asset_path  TEXT
);
```

```python
TIMELINE_V2_COLUMNS = ("slot_index", "act", "t_in", "t_out", "kind", "shot_id", "src_in",
                       "src_out", "secondary_shot_id", "secondary_src_in", "motion", "speed",
                       "transition", "beat_id", "scene_id", "msg_id", "yaw")


def _rebuild_timeline_v2(conn: sqlite3.Connection) -> bool:
    """A v1 timeline is dropped: it is derived data that the next S05 run
    rewrites in full. Returns True when it did something."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(timeline)")}
    if not existing or "beat_id" in existing:
        return False
    conn.execute("DROP TABLE timeline")
    conn.commit()
    return True
```

Call `_rebuild_timeline_v2(conn)` in `init` **before** `executescript(SCHEMA)` (so `CREATE TABLE IF NOT EXISTS` recreates it) — the same place `_rebuild_shots_for_photo_slots` is called.

- [ ] **Step 4: The config keys** (append inside the existing blocks; values from the spec)

```yaml
film:
  target_duration_s: 1500        # 25 min (section 13.11.5); max_duration_s stays 2400
  growth_bias: 0.6
  cold_open_s: [15, 25]          # replaces the dead scalar
  cold_open_card: "тремя месяцами ранее"
beats:
  face_hold_s: 2.5               # the speaker's own shot before B-roll takes over
  pre_roll_s: 0.4                # picture cuts this much before the utterance starts
  broll_window_s: 7200           # B-roll candidates within +-2 h of the beat, same act
  own_picture_face_score_below: 0.35   # below this the beat is a walking shot: picture stays
assemble:
  pair_window_s: 60
  pairs_per_act: 2
  max_consecutive_recording: 2   # outside a beat
  source_alternation_after: 3    # soft: prefer another source after this many in a row
  source_share_min: 0.25         # of an act's phone slots, to each phone that has material
  long_take_s: [60, 90]
  long_take_keywords: ["мост", "мосту", "моста", "bridge", "suspension"]
  similarity_fallback: {same_recording_60s: 1.0, same_recording: 0.6, same_place_hour: 0.3}
  rhythm:                        # section energy percentile -> slot length (s)
    low:  [5.0, 8.0]             # < 33
    mid:  [3.0, 5.0]             # 33-66
    high: [1.5, 2.5]             # > 66, cut on downbeats
    burst_slots: [8, 12]         # one beat each, on the act's highest swell
  chat_card_s: 4.0               # an Act 1 quote on screen
  closing_card_s: 6.0
  card_s: 3.0                    # the cold-open card
music:
  assignment: scene              # scene | act
  min_scene_s: 45
  reuse_gap_s: 300
  switch_cost: 0.6
  continuity_bonus: 0.3
  repeat_penalty: 0.5
  scene_targets:                 # weights of the distance terms
    energy: 1.0
    tempo: 0.6
    dynamics: 0.5
    brightness: 0.4
render:
  speech_lufs: -16.0
  location_full_lufs: -18.0
  location_under_speech_lufs: -24.0
  music_xfade_s: 2.0
  window_fade_s: 1.0             # music out into a natural-sound window or the silence
  cue_fade_s: 0.15
  final_lufs: -14.0
  true_peak_db: -1.5
  audio_bitrate_k: 192
```

(`render.music_lufs` −14, `render.duck_lufs` −28, `render.duck_lufs_speech` −22, `assemble.act4_held_shot_s`, `natural_sound_windows` 4, `natural_sound_window_s` [10, 20], `subject_shot_every_s` 40, `levity_min_per_act` 1, `max_reassembly_loops` 3, `film.photo_share` 0.10, `music.preferred_tracks`/`preferred_bonus` already exist; leave them.)

- [ ] **Step 5: Run the two test files; expect PASS. Then `tests/test_config_paths.py` and `tests/test_e2e_s01.py -k config` are not needed — run `tests/test_db.py tests/test_config_step4.py tests/test_s045_beats.py` (s045 seeds the DB and must still init).**

- [ ] **Step 6: Commit** — "Schema v2: the picture track, the audio cues and the overlays, and step 4's keys".

---

### Task 2: Deterministic similarity, the constraint ladder and source share (pure)

**Files:**
- Modify: `src/nepal/process/assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Produces:
  - `fallback_similarity(a: Mapping, b: Mapping, *, weights: Mapping[str, float]) -> float`
  - `make_similarity(embeddings: Mapping[str, np.ndarray], *, fallback_weights: Mapping[str, float]) -> Callable[[Mapping, Mapping], float]`
  - `mmr_select(candidates, *, budget, similarity: Callable[[Mapping, Mapping], float], lam=MMR_LAMBDA, admissible=None, prefer=None)` — **signature change**: `embeddings` is replaced by `similarity`; `prefer(candidate, chosen) -> float` is a soft bonus added to the MMR value (used for source alternation).
  - `recording_run_ok(candidate, chosen, *, limit: int) -> bool` — no more than `limit` consecutive chosen shots from one recording (looks at the tail of `chosen`).
  - `source_alternation_bonus(candidate, chosen, *, after: int, bonus: float = 0.05) -> float` — `+bonus` when the last `after` chosen share a source different from the candidate's, else 0.
  - `source_share_repair(chosen: list, pool: Sequence, *, min_share: float, phones=("phone_keller", "phone_kulikov")) -> list` — for each phone whose share of the phone slots in `chosen` is below `min_share` while `pool` still holds its shots, swap the lowest-scored slot of the over-represented phone for the best remaining shot of the starved phone, until the share holds or the pool is dry.
  - `lay_out(shots, *, start_s, duration_range, beats, downbeats=None) -> list[dict]` now emits the full slot dict (`SLOT_KEYS` minus `slot_index`, with `kind` from `media_kind`, `locked` 0, `speed` 1.0, `transition` "cut", others `None`).

- [ ] **Step 1: Failing tests**

```python
def test_fallback_similarity_is_deterministic_and_graded():
    w = {"same_recording_60s": 1.0, "same_recording": 0.6, "same_place_hour": 0.3}
    a = {"recording_id": "r1", "start_utc": "2024-05-01T04:00:00+00:00", "place_name": "Deng"}
    b = {"recording_id": "r1", "start_utc": "2024-05-01T04:00:30+00:00", "place_name": "Deng"}
    c = {"recording_id": "r1", "start_utc": "2024-05-01T05:30:00+00:00", "place_name": "Deng"}
    d = {"recording_id": "r2", "start_utc": "2024-05-01T04:20:00+00:00", "place_name": "Deng"}
    e = {"recording_id": "r3", "start_utc": "2024-05-02T04:20:00+00:00", "place_name": "Bihi"}
    assert asm.fallback_similarity(a, b, weights=w) == 1.0
    assert asm.fallback_similarity(a, c, weights=w) == 0.6
    assert asm.fallback_similarity(a, d, weights=w) == 0.3
    assert asm.fallback_similarity(a, e, weights=w) == 0.0


def test_the_fill_breaks_up_a_run_of_one_recording_without_embeddings():
    """The first draft's runs of one recording: with no CLIP, similarity was
    0.0 and MMR was score order. The fallback alone prevents it."""
    cands = [{"shot_id": f"r1#{i}", "recording_id": "r1", "score_total": 0.9 - i * 0.01,
              "start_utc": f"2024-05-01T04:0{i}:00+00:00", "place_name": "Deng"} for i in range(4)]
    cands += [{"shot_id": "r2#0", "recording_id": "r2", "score_total": 0.5,
               "start_utc": "2024-05-01T06:00:00+00:00", "place_name": "Bihi"}]
    sim = asm.make_similarity({}, fallback_weights={"same_recording_60s": 1.0,
                                                     "same_recording": 0.6, "same_place_hour": 0.3})
    picked = asm.mmr_select(cands, budget=3, similarity=sim, lam=0.5)
    assert "r2#0" in {p["shot_id"] for p in picked}


def test_recording_run_limit_and_source_alternation():
    chosen = [{"recording_id": "r1", "source": "phone_keller"}] * 2
    assert not asm.recording_run_ok({"recording_id": "r1"}, chosen, limit=2)
    assert asm.recording_run_ok({"recording_id": "r9"}, chosen, limit=2)
    chosen3 = [{"source": "camera"}] * 3
    assert asm.source_alternation_bonus({"source": "phone_keller"}, chosen3, after=3) > 0
    assert asm.source_alternation_bonus({"source": "camera"}, chosen3, after=3) == 0


def test_source_share_repair_gives_the_starved_phone_its_quarter():
    chosen = [{"shot_id": f"k{i}", "source": "phone_kulikov", "score_total": 0.9 - i * 0.05}
              for i in range(8)]
    pool = [{"shot_id": f"e{i}", "source": "phone_keller", "score_total": 0.4} for i in range(3)]
    out = asm.source_share_repair(chosen, pool, min_share=0.25)
    keller = [s for s in out if s["source"] == "phone_keller"]
    assert len(out) == 8 and len(keller) == 2          # 2 of 8 = 0.25
    assert {s["shot_id"] for s in out} >= {"k0", "k1", "k2"}   # the best of kulikov stay
```

- [ ] **Step 2: Run `tests/test_assemble.py`; expect the new tests to FAIL (and the two existing `mmr_select` callers in the file to break on the signature: update them to pass `similarity=asm.make_similarity(emb, fallback_weights=W)`).**

- [ ] **Step 3: Implement**

```python
def _utc_seconds(row: Mapping[str, Any]) -> float | None:
    ts = row.get("start_utc")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except ValueError:
        return None


def fallback_similarity(a, b, *, weights) -> float:
    """What two shots share when nothing has looked at their pictures. The
    first draft's runs of one recording came from a similarity of 0.0 for
    every pair; this alone breaks them up."""
    if a.get("recording_id") and a.get("recording_id") == b.get("recording_id"):
        ta, tb = _utc_seconds(a), _utc_seconds(b)
        if ta is not None and tb is not None and abs(ta - tb) <= 60.0:
            return float(weights.get("same_recording_60s", 1.0))
        return float(weights.get("same_recording", 0.6))
    if a.get("place_name") and a.get("place_name") == b.get("place_name"):
        ta, tb = _utc_seconds(a), _utc_seconds(b)
        if ta is not None and tb is not None and abs(ta - tb) <= 3600.0:
            return float(weights.get("same_place_hour", 0.3))
    return 0.0


def make_similarity(embeddings, *, fallback_weights):
    def sim(a, b) -> float:
        ea, eb = embeddings.get(a.get("shot_id")), embeddings.get(b.get("shot_id"))
        if ea is not None and eb is not None:
            return float(np.dot(ea, eb))
        return fallback_similarity(a, b, weights=fallback_weights)
    return sim
```

`mmr_select`: replace `_similarity(c, ch, embeddings)` with `similarity(c, ch)`, add `+ (prefer(c, chosen) if prefer else 0.0)` to the value. `recording_run_ok`: walk `chosen` from the end while `recording_id` equals the candidate's; `False` when the run reaches `limit`. `source_alternation_bonus`: `bonus` if `len(chosen) >= after` and all of the last `after` share one `source` that differs from the candidate's. `source_share_repair`: count phone slots; for each phone below `min_share` with pool material, repeatedly replace the lowest-scored slot of the most represented phone by the highest-scored pool shot of the starved phone (keeping chronology to the caller — it re-sorts); stop when the share holds or the pool is dry; never drop below one slot for the other phone.

`lay_out`: emit every `SLOT_KEYS` field except `slot_index`; `kind = "photo" if s.get("media_kind") == "photo" else "video"`.

- [ ] **Step 4: Run `tests/test_assemble.py`; expect PASS (existing tests updated for the signature).**

- [ ] **Step 5: Commit** — "Similarity without embeddings, and the constraints the spec named".

---

### Task 3: Anchors from the beat sheet (pure)

**Files:**
- Create: `src/nepal/story/anchors.py`
- Test: `tests/test_anchors.py`

**Interfaces:**
- Consumes: `story_beats` rows (dicts with the table's columns), `shots` rows by `shot_id` (needs `recording_id`, `start_utc`, `start_s`, `end_s`, `face_score`, `act`, `media_kind`), config values.
- Produces:
  - `Anchor` (above).
  - `speech_anchors(beats, shots_by_id, *, face_hold_s, pre_roll_s, own_picture_below) -> list[Anchor]` — `kind == "speech"` beats only; `src_in = max(shot.start_s, beat.src_in - pre_roll_s)`; `utc = shot.start_utc + (beat.src_in - shot.start_s)`; `own_picture = (shot.face_score or 0) < own_picture_below`; `face_hold_s = 0 if own_picture else min(face_hold_s, duration_s)`.
  - `quote_anchors(beats) -> list[Anchor]` — `kind in ("quote", "closing")`, `recording_id/shot_id None`, `duration_s` 0 (the card length is config), `utc` from the message's `ts_utc` if present on the row else `None`.
  - `place_anchors(anchors, *, act_t0: float, act_len_s: float, act_utc0: float | None, act_utc1: float | None, min_gap_s: float = 4.0) -> list[Anchor]` — position each speech anchor at `act_t0 + act_len_s * frac` where `frac` is its `utc`'s fraction of `[act_utc0, act_utc1]` (anchors without `utc` are spread evenly in rank order); then sweep forward so `t_in >= previous.t_in + previous.duration_s + min_gap_s`; then sweep backward from `act_t0 + act_len_s` so the last anchor ends inside the act. Returns copies with `t_in` set, sorted by `t_in`.
  - `broll_candidates(anchor, shots, *, window_s) -> list[dict]` — shots in the same act, `|utc - anchor.utc| <= window_s`, `recording_id != anchor.recording_id`, `media_kind == "video"`, sorted by score.
  - `cold_open_pick(beats, shots_by_id, *, length_range: Sequence[float]) -> dict | None` — the top-ranked speech beat in act 3 or 4 (lowest `rank`), extended around its utterance to `length_range[0]..[1]` within the shot's bounds; returns a slot dict (`kind video`, `beat_id`, `locked 1`, `transition "dip_black"`); `None` when no such beat.

- [ ] **Step 1: Failing tests** (build 3 speech beats in act 3 with a shot each, one with `face_score` 0.2 → `own_picture`; one quote; assert `src_in` pre-rolled and clamped at `start_s`; `place_anchors` keeps order, respects the gap, and ends inside the act; `broll_candidates` excludes the beat's own recording and the other act; `cold_open_pick` picks rank 1 in act 3/4 and returns a slot of 15–25 s inside the shot's bounds; a shot shorter than 15 s gives what it has.)

- [ ] **Step 2: Run to fail. Step 3: Implement as specified. Step 4: Run to pass.**

- [ ] **Step 5: Commit** — "Anchors: where the beats sit and what picture they carry".

---

### Task 4: Two phones, one moment (pure)

**Files:**
- Create: `src/nepal/process/pairs.py`
- Test: `tests/test_pairs.py`

**Interfaces:**
- Consumes: shot rows with `shot_id`, `recording_id`, `source`, `start_utc`, `start_s`, `end_s`, `score_total`, `face_cluster`, `width`, `height` (width/height joined from the recording's first asset in S05).
- Produces:
  - `is_portrait(row) -> bool` (`height > width`).
  - `find_pairs(rows, *, window_s: float, per_act: int) -> list[dict]` — pairs of shots from different sources (`phone_keller`/`phone_kulikov`, or a phone and `camera`) whose UTC spans overlap within `window_s`, at least one portrait; ranked by `score_total` sum plus 0.2 when `face_cluster` differs and both are set; at most `per_act` per act, no shot in two pairs. Each pair is a slot dict: `kind "video"`, `shot_id` = the higher-scored, `secondary_shot_id` = the other, `src_in`/`secondary_src_in` aligned to the same instant (the later start), `src_out = src_in + min(remaining of both)`, `motion '{"type":"split"}'`, `locked 0`.

- [ ] **Step 1: Failing tests** (two phones 20 s apart → one pair aligned to the later start; same source → none; three pairs in one act → two; a shot cannot appear twice; landscape+landscape → none.)
- [ ] **Steps 2–4: fail, implement, pass.**
- [ ] **Step 5: Commit** — "Pairs: two phones, one moment".

---

### Task 5: The long take (pure)

**Files:**
- Create: `src/nepal/process/longtake.py`
- Test: `tests/test_longtake.py`

**Interfaces:**
- Consumes: shot rows (with `transcript`, `caption`, `recording_id`, `act`, `start_s`, `end_s`, `alt_dem_m`, `has_face`) and recording rows (`recording_id`, `duration_s`, `is_360`).
- Produces:
  - `bridge_candidates(shots, recordings, *, keywords) -> list[dict]` — recordings with at least one shot whose `transcript` or `caption` contains a keyword (case-insensitive), ranked by (`duration_s` of the recording capped at 120, number of keyword shots, `-has_face` share); one dict per recording with `recording_id`, `first_shot_id`, `act`, `score`.
  - `long_take_slot(candidate, recordings_by_id, shots_by_recording, *, length_range) -> dict | None` — one slot of `length_range[1]` seconds (or the recording's whole length if shorter, `None` below `length_range[0]`) starting at the first keyword shot's `start_s`, `kind video`, `locked 1`, `motion None`, `speed 1.0`, `transition "cut"`.

DEM drop below the bridge (spec §13.5) is not ranked here: it needs the crossing point, which the peak geometry of §13.3 (step 5/6) provides; the plan records that in the module docstring.

- [ ] **Steps 1–5** as above; tests: a recording with "мост" in a transcript wins over a longer silent one; a 40 s recording yields `None` under `[60, 90]`; a 300 s one yields 90 s from the keyword shot.
- [ ] **Commit** — "The long take: the bridge, unbroken".

---

### Task 6: Scenes and their targets (pure)

**Files:**
- Create: `src/nepal/spine/scenes.py`
- Test: `tests/test_scenes.py`

**Interfaces:**
- Consumes: slots (with `act`, `t_in`, `t_out`, `shot_id`, `beat_id`), a per-slot attribute lookup provided by S05: `attrs(slot) -> dict(speed_ms, hr_bpm, gain_m_per_h, alt_m, hour, place_name, act, levity, motion_mag, under_speech)`.
- Produces:
  - `activity_class(a: Mapping) -> str` — `summit` (act 4), `city` (act 1/5 with place in the city list or alt < 1500), `transport` (speed > 6 m/s), `resting` (speed < 0.3), `crossing` (place contains "bridge"/"мост"), `village` (place_name set and speed < 1.0), `climbing` (gain > 150 m/h), `descending` (gain < −150 m/h), else `walking`.
  - `Scene` dataclass: `scene_id: int`, `act: int`, `t_in`, `t_out`, `slot_indices: tuple[int, ...]`, `activity: str`, `speed_ms`, `hr_bpm`, `alt_m`, `hour`, `voice_share`, `levity: bool`, `picture_energy`.
  - `group_scenes(slots, attrs, *, min_scene_s, max_scene_s, beats_spans: Sequence[tuple[float, float]]) -> list[Scene]` — consecutive slots merge while act and activity match and speed/HR stay within one band (`speed` bands 0/0.3/1.0/2.0/6 m/s; HR bands of 20 bpm); a scene shorter than `min_scene_s` is merged with its neighbour in the same act; longer than `max_scene_s` is split at the slot boundary nearest the middle that is not inside any beat span.
  - `scene_target(scene, *, weights) -> dict[str, float]` — `energy` in [0,1]: 0.3 base + 0.4·effort (hr-or-gain normalised) + 0.2·(act in 3,4) − 0.2·(activity in village/resting/city); `tempo_bpm`: 60 + 60·min(speed_ms/1.4, 1.5) (accepting half/double time downstream); `dynamics` in [0,1]: 0.8 for climbing/summit, 0.3 under speech (voice_share > 0.5), else 0.5; `brightness` in [0,1]: 0.2 pre-dawn or alt > 4,500, 0.9 village/city daytime, else 0.5.

- [ ] **Steps 1–5**; tests: 10 slots alternating walking/village produce ≥ 2 scenes; a 30 s scene merges; a 400 s scene splits outside a beat span; `scene_target` energy is higher on the climb than in a village; summit brightness is low.
- [ ] **Commit** — "Scenes: what a stretch of the film is, before music is chosen for it".

---

### Task 7: Music per scene (Viterbi) and the map in the existing shape

**Files:**
- Modify: `src/nepal/spine/music.py`
- Test: `tests/test_music.py` (append)

**Interfaces:**
- Consumes: `Track` (existing; sections with `start_s/end_s/energy/is_swell`), `Scene`, `scene_target`.
- Produces:
  - `section_features(track: Track, section: Mapping) -> dict` — `energy_pct` (percentile of the section's energy within the **library**, 0–1), `tempo_bpm` (track's), `dynamics` (track `dyn_range` normalised over the library), `brightness` (centroid normalised), `role` (`intro|build|swell|outro` from `is_swell` and the position in the piece), `length_s`.
  - `pair_cost(target: Mapping, feat: Mapping, *, weights) -> float` — weighted distance; tempo term is `min(|log2(t/f)|, |log2(t/(2f))|, |log2(2t/f)|)` so half and double time count as a match.
  - `SceneAssignment` dataclass: `by_scene: dict[int, tuple[str, str]]` (scene_id → (track_id, section_id)), `cost: float`, `note: str`.
  - `assign_scenes(scenes, tracks, *, weights, switch_cost, continuity_bonus, repeat_penalty, reuse_gap_s, preferred: Sequence[str], preferred_bonus, exclude: Sequence[str], act4_swell: bool = True) -> SceneAssignment` — Viterbi over scenes with (track, section) states: cost = `pair_cost` + `switch_cost` when the piece changes − `continuity_bonus` when the section is the next one in the same piece + `repeat_penalty` when the piece was heard within `reuse_gap_s` (approximated by the previous scenes' spans) − `preferred_bonus` when the track's file name is in `preferred`; `exclude` (the credits track) is removed from the states; the last scene of act 4 is restricted to `is_swell` sections when any exist; Act 1 and the last act: a `callback_affinity`-style bonus for sharing a piece with Act 1 (reuse the existing `callback_affinity`).
  - `music_map_from_scenes(scenes, assignment, tracks, *, act_spans: Mapping[int, tuple[float, float]], silence_s, act0_span: tuple[float, float] | None) -> dict` — the **existing** `music_map.json` shape (`total_duration_s`, `acts[…].segments[{track_id,title,artist,t_in,t_end,src_in,src_out}]`, `t_start`, `t_end`, `swells`, `beat_grid`, `downbeats` on the film timeline, `silence_window`, `assignment_note`): each scene becomes a segment of its act; consecutive scenes on the same section merge; a segment's `src_in` is the section's `start_s` (not 0.0 any more) and `src_out = src_in + length`, wrapping into the next section of the piece when the scene is longer; Act 0 takes the Act 4 swell segment from its swell.

- [ ] **Step 1: Failing tests** — a two-track library (one calm, one driving) and four scenes (village, climb, summit, village): the climb and summit get the driving track, the villages the calm one; continuity keeps the summit on the section after the climb's; the excluded credits track never appears; `preferred_bonus` tilts a near tie; the map's segments cover each act's span within 0.5 s and `beat_grid` values are on the film timeline (≥ `t_start`).
- [ ] **Steps 2–4.**
- [ ] **Step 5: Commit** — "Music chosen by the scene: a Viterbi pass over sections".

---

### Task 8: Rhythm from the music (pure)

**Files:**
- Create: `src/nepal/process/rhythm.py`
- Test: `tests/test_rhythm.py`

**Interfaces:**
- Consumes: an act's slots (with `locked`), the act's music sections **on the film timeline** (`t_in`, `t_out`, `energy`), the beat grid and downbeats on the film timeline, config `assemble.rhythm`, `assemble.act4_held_shot_s`.
- Produces:
  - `energy_percentiles(sections) -> list[float]` — within the act's cue.
  - `target_length(pct: float, table: Mapping) -> tuple[float, float]`.
  - `retime(slots, *, sections, beats, downbeats, table, burst_slots: Sequence[int], is_act4: bool, held_shot_s: Sequence[float], silence_t: float | None) -> list[dict]` — walks the act's slots in order: a `locked` slot keeps its bounds; an unlocked slot's length becomes `target_length` of the section under its `t_in` (clamped to `shot_available_s`), its `t_out` snapped to the nearest beat (downbeat above the 66th percentile); the run of unlocked slots under the act's highest swell becomes a **burst**: `burst_slots` slots of one beat each, taken from the burst's own shots first and then the act's next unlocked slots (each contributes one beat); in act 4 the last unlocked slot before `silence_t` is held for `held_shot_s` (mid) and its `t_out` = `silence_t`. Slots that fall off the end of the act are dropped; the function returns the re-timed list and the caller re-fills a remaining gap by calling the fill again (S05 does this in `assemble.max_reassembly_loops` rounds).

- [ ] **Step 1: Failing tests** — three sections (low/mid/high): slots under each get lengths inside the table's ranges and ends on beats; a locked slot keeps its bounds; the swell section yields a run of one-beat slots; act 4 ends on a held shot at `silence_t`; a shot with 2 s of material never gets 5.
- [ ] **Steps 2–4.**
- [ ] **Step 5: Commit** — "Rhythm: slot lengths from the music's energy, bursts on the swell".

---

### Task 9: The timeline v2 in S05

**Files:**
- Modify: `src/nepal/stages/s05_cut.py` (`build_timeline` rewritten; helpers `_shot_rows`, `_act_spans`, `_attrs_for`)
- Modify: `src/nepal/process/timeline_io.py` (`fps` parameter passed from `render.fps`; three empty audio tracks placeholder is Task 12's job — here only the fps)
- Modify: `src/nepal/cloud/status.py` (`per_act_sources`)
- Test: `tests/test_s05_timeline.py` (new; seeded DB, no media), `tests/test_cloud_status.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8; `select.plan_photo_slots`; `music_mod.choose_total_duration`, `allocate_act_durations`; `effort.profile`/`at`/`hardest_windows`; `db.get_decision(conn, "act_boundaries")`.
- Produces:
  - `_shot_rows(conn) -> list[dict]` — shortlisted shots joined with `source` (recording's for video, asset's for photo), `is_360`, `width`, `height` (from the recording's first asset by `chapter_index`), `face_score`, `transcript`, `caption`.
  - `plan_act(cfg, act, act_rows, beats, *, t0, act_len_s, act_span_utc, similarity, photo_budget, long_take, pairs) -> tuple[list[dict], list[Anchor]]` — the per-act algorithm below.
  - `build_timeline(cfg, conn) -> dict` — the report keys: `n_slots`, `duration_s`, `per_act` (slots), `per_act_sources` (`{act: {source: {"slots": n, "available": n}}}`), `n_anchors`, `n_pairs`, `long_take` (recording id or `None`), `n_scenes`, `music_assignment` (`scene|act`), `cold_open_beat`.
  - `status.per_act_sources(conn) -> dict[str, dict[str, dict[str, int]]]` and a `<h2>sources per act</h2>` table in `render_html` (slots / available / share).

**The algorithm of `build_timeline`:**

1. `rows = _shot_rows(conn)`; `beats = SELECT * FROM story_beats`; `bounds = act_boundaries` decision → `act_span_utc[act] = (utc0, utc1)`; `prof = effort.profile(gps points from gps_points ordered by ts)`.
2. `material_s = sum(min(shot_available_s(r), 20.0) for r in rows)`; `total_s = choose_total_duration(film.target, film.max, min_s=film.min, material_s=material_s, growth_bias=film.growth_bias, selectivity=film.material_selectivity)`; `act_len = allocate_act_durations(cfg.act_targets(), total_s)`.
3. `similarity = make_similarity(_load_embeddings(cfg), fallback_weights=assemble.similarity_fallback)`.
4. **Act 0:** `cold = cold_open_pick(beats, shots_by_id, length_range=film.cold_open_s)`; slots: `[cold, card(film.cold_open_card, assemble.card_s)]` at `t = 0`; both `locked`; if `cold` is `None`, the highest-scoring act-4 shot for `cold_open_s[1]` seconds.
5. **Per act 1–5, in order** (`plan_act`): `photo_budget = plan_photo_slots({act: photo rows}, {act: act_len[act]}, share=film.photo_share, share_by_act=film.photo_share_by_act)[act]` → the chosen photo ids are admissible, other photos are not (never two adjacent unless in a burst: `admissible` refuses a photo when the previous chosen is a photo); `anchors = place_anchors(speech_anchors(...), act_t0, act_len, *act_span_utc)`; the long take (act of the best candidate, Task 5) and pairs (Task 4, `pairs_per_act`) are inserted as locked/unlocked slots at their chronological positions among the anchors; for each gap between consecutive anchors: candidates = act rows with utc inside the gap's utc window (or all act rows when utc is missing), excluding the anchors' recordings; `budget = slot_budget(gap_len, assemble.shot_duration_s[act])`; `mmr_select(cands, budget, similarity, lam, admissible=place_count_ok ∧ recording_run_ok ∧ photo rule, prefer=source_alternation_bonus)`; `chosen = source_share_repair(chosen, remaining act rows, min_share=assemble.source_share_min)`; `chronological`; `lay_out(...)` into the gap; then the anchor itself: `own_picture` → one slot on the beat's recording for its whole utterance (`locked`); else `face_hold_s` on the beat's shot (`locked`) then B-roll from `broll_candidates` laid to the utterance's end (`beat_id` set on every slot under the beat, `locked` on the first). After the act: `needs_subject`/`missing_levity` are reported (not enforced — the beat sheet's levity beats satisfy most acts; the report says which act misses).
6. **Scenes → music:** `attrs = _attrs_for(prof, rows)`; `scenes = group_scenes(all slots, attrs, min_scene_s, max_segment_s, beat spans)`; `assignment = assign_scenes(...)` (or the S02.7 per-act map when `music.assignment == "act"`); `mmap = music_map_from_scenes(...)`; `check_music_map` problems logged; written to `work/music/music_map.json` (the S02.7 promise, kept) and `scene_id` set on every slot.
7. **Rhythm:** per act, `retime(...)` with the act's sections from `mmap` (segments → sections on the film timeline with their energies from `music_sections`), the act's `beat_grid`/`downbeats`; the natural-sound windows (`hardest_windows(prof, count=assemble.natural_sound_windows, window_s=mid of natural_sound_window_s)`) are mapped to film time through the slot whose utc contains them and stored in the report as `natural_windows` (Task 11 turns them into cues); a gap left by shortened slots is refilled with the next-best candidates up to `assemble.max_reassembly_loops` times, then the act's end moves.
8. `slot_index` assigned in order; `DELETE FROM timeline`; `db.upsert(conn, "timeline", ["slot_index"], rows)` with every column of `TIMELINE_V2_COLUMNS`; `timeline_io.write(rows, …, fps=render.fps)`.

- [ ] **Step 1: Failing test** `tests/test_s05_timeline.py` — seed a DB the way `tests/test_s045_beats.py::_seed` does but with ~30 shortlisted shots over acts 1–5 from three sources (both phones and camera), portrait phone pairs at the same minute, one recording with "мост" in its transcript of 120 s, three `story_beats` (two speech incl. one walking, one quote), two `music_tracks` with sections and a `beats` grid, `act_boundaries` decision, a few `gps_points`; run `s05_cut.build_timeline(cfg, conn)`; assert: the first slot is the cold open and the second a `card`; every speech beat has slots with its `beat_id` and the walking beat's slots are all on its recording; a `split` slot exists with two different sources; the long take slot is 90 s and locked; every act's phone slots give each phone ≥ 25 % where it had material (`per_act_sources`); `music_map.json` exists with `segments` whose `src_in` is not always 0; `scene_id` is set on every slot; no two adjacent photo slots; `t_in` non-decreasing and `t_out > t_in` everywhere; no slot exceeds its shot's material.
- [ ] **Step 2: Run to fail. Step 3: Implement. Step 4: Run to pass; also `tests/test_cloud_status.py` for the new table.**
- [ ] **Step 5: Commit** — "The timeline v2: anchors, pairs, the long take, the fill, scenes, rhythm".

---

### Task 10: Part A on the box — a silent v2 draft the operator can watch

- [ ] `nepal remote up`; `nepal remote run cut --redo score,timeline,draft` (the existing `render_draft` still works: it reads `timeline` rows by `slot_index` — make sure it selects `kind`, ignores `card` slots by rendering black with `drawtext` of the payload, and renders `split` as the primary only for now); read `reports/remote_jobs.log`; `nepal remote pull`; check `work/reports/s05_cut.json` (`per_act_sources`, `n_anchors`, `long_take`, `n_scenes`); watch `work/gates/gate3/draft.mp4` for the cut order; `nepal remote down`.
- [ ] Record in `docs/STATE.md` (a "step 4, part A" paragraph): total duration chosen, slots per act, source ratio per act, anchors placed, the long take's recording, the pairs, scenes and the music assignment, and what the picture looks like.
- [ ] Commit — "STATE: the first v2 picture cut, measured".

---

## Part B — sound

### Task 11: Audio cues and overlay rows (pure + stage sub-step)

**Files:**
- Create: `src/nepal/process/cues.py`
- Modify: `src/nepal/stages/s05_cut.py` (sub-step `cues` between `timeline` and `draft`; CLI default `--redo score,timeline,cues,draft`)
- Modify: `src/nepal/cli.py` (the `cut` sub-command's `--redo` help)
- Test: `tests/test_cues.py`, `tests/test_s05_timeline.py` (append)

**Interfaces:**
- Produces:
  - `speech_cues(anchors: Sequence[Anchor], *, lufs: float, fade_s: float) -> list[dict]` — one `speech` cue per speech anchor: `cue_id f"sp_{beat_id}"`, `t_in = anchor.t_in`, `t_out = t_in + duration_s`, `source = recording_id`, `src_in/src_out`, `gain_lufs = lufs`, fades, `beat_id`.
  - `location_cues(slots, *, lufs_under_music, lufs_full, lufs_under_speech, speech_spans, windows, silence) -> list[dict]` — one `location` cue per `video` slot from its own recording at `src_in..src_out`; a `photo`/`card` slot gets the previous video slot's recording continuing from its `src_out` (held ambience); `gain_lufs` = full inside a natural-sound window or the silence window, under-speech inside a speech span, else under-music.
  - `music_cues(mmap, *, lufs, xfade_s) -> list[dict]` — one `music` cue per segment of every act (plus act 0), `source = track_id`, `src_in/src_out`, `fade_in_s = fade_out_s = xfade_s` (the renderer turns adjacent cues into crossfades).
  - `overlay_rows(beats, anchors, slots, *, chat_card_s, closing_card_s, cast_tags: Mapping[str, str]) -> list[dict]` — `chat_card` overlays for quote beats spread over Act 1's span in rank order (`payload` JSON: `{"text", "author_tag", "side": "left"|"right" alternating, "msg_id"}`), one `chat_card` for the closing beat over the last `closing_card_s` before credits, and `stat_card` rows are **not** written here (step 6, from the beat sheet's ideas). `author_tag` comes from `Cast.build` over the messages table order (reuse `story.beats_input.Cast`).
  - `S05.cues` sub-step: `DELETE FROM audio_cues`/`overlays`; upsert both; report `n_cues` per track, `n_overlays`, `natural_windows`.

- [ ] **Step 1: Failing tests** — a walking anchor gives a speech cue with its `src_in/out`; a photo slot's location cue continues the previous recording; a slot inside a natural window has `gain_lufs == lufs_full`; a slot under a speech span has the under-speech level; music cues cover each act's span end to end; quote overlays alternate sides and carry tags, never names (assert `"Kulikov" not in json.dumps(rows)` — the author, not the body).
- [ ] **Steps 2–4.**
- [ ] **Step 5: Commit** — "Audio cues and overlay rows: what the mix and step 6 read".

---

### Task 12: Envelopes (pure; `mix.py` rewritten)

**Files:**
- Rewrite: `src/nepal/process/mix.py`
- Test: `tests/test_mix.py` (rewritten)

**Interfaces:**
- Produces:
  - `Envelope = list[tuple[float, float]]` — `(t_s, gain_db)` breakpoints, piecewise linear, sorted, starting at `t = 0`.
  - `music_envelope(*, total_s, speech_spans, windows, silence, base_db=0.0, under_speech_db=-8.0, window_fade_s=1.0, cue_fade_s=0.15) -> Envelope` — `base_db` everywhere; `under_speech_db` inside speech spans with `cue_fade_s` ramps; ramp to −70 dB over `window_fade_s` into each natural-sound window and the silence window, back over the same after.
  - `location_envelope(*, total_s, cues) -> Envelope` — from the location cues' `gain_lufs` relative to `render.location_full_lufs` (0 dB) with `cue_fade_s` ramps at cue edges.
  - `volume_expr(env: Envelope) -> str` — an ffmpeg `volume` expression in dB: nested `if(lt(t,T1), A+(B-A)*(t-T0)/(T1-T0), …)` with the last value held; `eval=frame` is added by the renderer. Gains are emitted as `pow(10, dB/20)`.
  - `levels_for_shot` and `MixLevels` are removed; the numbers come from config through the callers.

- [ ] **Step 1: Failing tests** — a music envelope with one speech span and one window: 0 dB outside, −8 dB inside the span with 0.15 s ramps, −70 dB inside the window with 1 s ramps; breakpoints sorted and start at 0; `volume_expr` of a two-point envelope evaluates (by a tiny Python evaluator in the test) to the midpoint gain at the midpoint time; the expression has no newline and no spaces.
- [ ] **Steps 2–4.**
- [ ] **Step 5: Commit** — "Envelopes: the mix as breakpoints, from the cues".

---

### Task 13: The audio graph and the v2 picture chain in `render.py`

**Files:**
- Modify: `src/nepal/process/render.py`
- Test: `tests/test_render.py` (command-shape tests + one `@pytest.mark.slow` two-track render checked with `ebur128`/`volumedetect`)

**Interfaces:**
- Produces:
  - `segment_filters(row, index, *, width, height, overlay, fps, secondary_index: int | None = None, card_text: str | None = None) -> str` — `split`: primary scaled/cropped to `width//2 × height`, secondary likewise, `hstack`; `card`: `color=c=black:s=WxH:r=fps:d=D` with `drawtext` of `card_text` centred (draft placeholder); `photo`/`video` as today.
  - `build_command(rows, *, sources, out_path, cues: Sequence[Mapping] = (), audio_sources: Mapping[str, Path] = {}, music_sources: Mapping[str, Path] = {}, envelopes: Mapping[str, str] = {}, width, height, crf, fps, overlay, levels: Mapping[str, float], script_path: Path | None = None) -> list[str]` — video inputs as today (+ one input per `secondary_shot_id`); one input per speech cue (`-ss src_in -t dur -i wav`), one per location cue, one per music cue (`-ss src_in -t dur -i track`); filters: speech cues → `loudnorm=I=speech_lufs:TP=-1.5:LRA=11,afade=in:d=cue_fade,afade=out:st=…,adelay=t_in*1000|t_in*1000` → `amix` into `[speech]`; location cues → `volume=<gain>dB,adelay` → `amix` into `[loc]` then `volume='<location_expr>':eval=frame`; music cues → `acrossfade=d=xfade` chained in time order → `[mus]` then `volume='<music_expr>':eval=frame`; `[speech][loc][mus]amix=inputs=3:normalize=0,loudnorm=I=final_lufs:TP=true_peak_db:LRA=11[aout]`; `-map [vout] -map [aout] -c:a aac -b:a 192k`. When `script_path` is given the graph is written there and `-filter_complex_script` is used (the graph is tens of kilobytes). No cues → the current silent command.
  - `probe_loudness(path, *, t_in, t_out) -> float` — `ffmpeg -ss -t -i -af volumedetect` mean volume, for the slow test.

- [ ] **Step 1: Failing tests** — shape: a `split` row yields an `hstack` and two inputs; a `card` row yields `color=` and the text; with cues the command has `amix=inputs=3`, `loudnorm=I=-14`, `-filter_complex_script`, one `adelay` per cue, and `acrossfade` between two music cues. Slow: two lavfi video clips, a 440 Hz "speech" wav cue at 2–5 s, a music track (`aevalsrc` 220 Hz) over 0–12 s, a silence window at 8–10 s: render, then `probe_loudness` shows the 2–5 s window ≥ 6 dB louder in the speech band than 5–8 s… simpler and robust: `volumedetect` mean at 0–2 s (music alone) is within 3 dB of 5–8 s, at 8–10 s is below −50 dB, and the file has an audio stream of ≥ 11.5 s.
- [ ] **Steps 2–4.**
- [ ] **Step 5: Commit** — "The audio graph as data: three tracks, envelopes, one mix".

---

### Task 14: The draft with sound and the Gate 3 page

**Files:**
- Modify: `src/nepal/stages/s05_cut.py` (`render_draft`: cues, sources, envelopes, the Gate 3 page)
- Create: `src/nepal/story/gate3.py`
- Modify: `src/nepal/process/timeline_io.py` (three audio tracks from `audio_cues` and an overlays track in OTIO; FCPXML unchanged but at `render.fps`)
- Test: `tests/test_gate3.py`, `tests/test_timeline_io.py` (append), `tests/test_e2e_s05.py` (slow, new)

**Interfaces:**
- Produces:
  - `gate3.render_gate3(slots, cues, overlays, *, per_act_sources, report, cast_tags) -> str` — a static page: the shot list keyed by wall-clock time and slot index (source, kind, beat id, scene, seconds), the per-act source ratio table (slots / available / share), the cues per track, the overlays with payload text (tags only), the run report's numbers; no author name anywhere (test asserts).
  - `render_draft(cfg, conn)` — rows from `timeline` (v2 columns) with sources; cues from `audio_cues`; `audio_sources = {recording_id: work/audio/<rid>.wav}`; `music_sources = {track_id: data_root/<s3_key minus "raw/">}` from `music_tracks`; envelopes from Task 12 (speech spans from the speech cues, windows from the report, silence from the map); `build_command(..., script_path=work/gates/gate3/draft.filters)`; the draft at `work/gates/gate3/draft.mp4`; the page at `work/gates/gate3/index.html`; report `audio_tracks: 3`, `n_cues`, `draft_s`.
  - `timeline_io.write(rows, out_dir, *, media_dir, fps, cues=(), overlays=())`.

- [ ] **Step 1: Failing tests** — `test_gate3`: the page lists a slot with its wall clock and the source ratio, contains the cue count and no names; `test_timeline_io`: OTIO has one video track plus three audio tracks named `speech`/`location`/`music` when cues are given and the fps is 30; `test_e2e_s05` (slow): a fixture DB (as in Task 9) plus lavfi proxies/wavs/music under a tmp work root, `s05_cut.run(cfg)` end to end: the draft exists, has an audio stream, its duration is within 1 s of the timeline's, and `gates/gate3/index.html` exists.
- [ ] **Steps 2–4.**
- [ ] **Step 5: Commit** — "The draft hears itself: speech, location and music, and Gate 3".

---

### Task 15: Part B on the box — Gate 3

- [ ] `nepal remote up`; `nepal remote run cut --redo cues,draft`; the suite: `nepal remote exec -- '.venv/bin/python -m pytest -q -p no:cacheprovider -m "slow or not slow"'` (detached, log under `reports/remote_jobs/`); `nepal remote pull`; open `work/gates/gate3/index.html`, watch `draft.mp4`; `nepal remote down`.
- [ ] `docs/STATE.md`: the step 4 section with the numbers (duration, slots, anchors, pairs, long take, scenes, the music assignment note, cues per track, loudness of the final mix from `loudnorm`'s summary), what the operator should listen for at Gate 3, and the open items (DEM drop for the bridge, six acts in step 7, overlays rendered in step 6). `README.md`: `nepal cut` sub-steps and Gate 3. `CLAUDE.md`: any trap met.
- [ ] Commit — "STATE: step 4 on the box; Gate 3 is the operator's".

---

## Self-review against the spec

- §1.1 six acts: Act 0 built here, Act 6 deferred to step 7 (stated in Global Constraints). §1.2 moments: anchors with `src_in/src_out` (Task 3), picture slots (Task 9). §1.3: the beat sheet drives the cut (Tasks 3, 9). §1.4: rhythm (Task 8).
- §3.3: Task 1 (`timeline`, `audio_cues`, `overlays`; OTIO tracks in Task 14).
- §5.1 anchors, face hold, B-roll window, walking shots, chat cards, closing card: Tasks 3, 9, 11. §5.2 fill, fallback similarity, the seven constraints: Task 2 (stills rule, subject, levity reported, place, recording run, source alternation, chronology) and Task 9 (photo budget via `plan_photo_slots`). §5.3 pairs: Task 4. §5.4 rhythm, bursts, Act 4 held shot and silence, natural-sound windows: Tasks 8, 9, 11. §5.5 Act 0 and the card: Tasks 3, 9; credits stay as they are (generated at render; unchanged). §5.6 scenes, targets, Viterbi, preferred bonus, credits track withheld, Act 4 ends on a swell, Act 0 takes Act 4's swell, the map in the existing shape, `music.assignment: act` kept: Tasks 6, 7.
- §6.1: only `split` (Task 4/13); the rest is step 6. §7: Tasks 11–14. §8.9 photo budget and the constraints invoked: Tasks 2, 9. §9 Gate 3: Task 14. §10 tests: pure per module, slow render with a loudness check, e2e S05. §13.5: Task 5 (DEM drop deferred, stated). §13.11.4 source share and the ratio on the status page and Gate 3: Tasks 2, 9, 14. §13.11.5 length: Task 1 (config) and Task 9 (allocation from material).
- Type consistency: `Anchor` fields used in Tasks 9 and 11 match Task 3; slot dict keys are `SLOT_KEYS` everywhere; `assign_scenes` → `music_map_from_scenes` → `retime` consume the shapes named in Tasks 7–8; `build_command`'s `cues`/`envelopes` come from Tasks 11–12.
