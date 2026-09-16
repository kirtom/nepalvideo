# Film v2 — Step 1: data hygiene and the Strava spine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the data defects that make the first draft wrong (ghost recordings, faces never flagged, video shots without position, stills-then-videos, 120 fps) and bring the Strava track, heart rate and the phone heading tags into the database, so every later step of Film v2 builds on correct data.

**Architecture:** Every fix follows the repository's rule — decisions in pure functions, shelling and SQL in thin stage code. New modules: `nepal/prune.py` (what the pipeline no longer produces), `nepal/spine/strava.py` (FIT and CSV in, `GpsPoint` out), `nepal/spine/place.py` (position, altitude, place and day for a moment). Existing stages gain `--redo` so a cheap sub-step can be re-run without paying for the expensive ones. Nothing here needs the cloud; the two real-corpus runs that exceed five minutes locally (the manifest re-probe and the draft render) are deferred to the remote box that step 2 builds.

**Tech Stack:** Python 3.10+, SQLite, numpy, `fitdecode` (pure-Python FIT decoder, new core dependency), exiftool, ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-film-v2-voice-spine-design.md` — this plan implements §8 items 1–5, 8, 9 (constraints wiring is step 4), 11, 13, 14, 15, and the data half of §13.1.

## Global Constraints

- Every tunable lives in `config/pipeline.yaml`, reachable by dotted path; no magic numbers in code. **Duplicate top-level keys raise**, so new keys go *inside* the existing `probe:`, `spine:`, `render:` blocks, never as a second block.
- Everything runs through the `nepal` console script: `/data/projects/nepalvideo/.venv/bin/nepal`. Tests run with `/data/projects/nepalvideo/.venv/bin/python -m pytest`.
- Nothing that runs longer than about five minutes runs on this machine. Cheap local re-runs only: `nepal prune`, `nepal s02 --redo ...`, `nepal s03 --redo place`, `nepal cut --redo score,timeline`.
- `db.upsert` never removes. Anything a rebuild no longer produces must be deleted explicitly.
- Modules that shell out keep the shelling in a thin wrapper; decision logic stays pure and unit-testable with no media.
- Comments say why, not what.
- Commit after every task with a message that explains the reasoning, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Media (`/data/projects/nepal_data`) and `work_root` (`/data/projects/nepal_work`) stay where they are; never copy or move them.

## File structure

| File | Responsibility |
|---|---|
| `src/nepal/db.py` | schema, migrations, `upsert` (Task 1: refuse heterogeneous rows; Tasks 3, 5: new columns and the `activities` table) |
| `src/nepal/prune.py` (new) | `plan_prune` (pure) + `execute` + `run` — remove recordings, shots, timeline rows and work files the pipeline no longer produces |
| `src/nepal/probe/manifest.py` | `HEADING_TAGS`, `parse_heading`, `walk_media(ignore_globs=…)` |
| `src/nepal/util/proc.py` | `EXIF_TAGS` derived from `CAPTURE_TAGS` **and** `HEADING_TAGS` |
| `src/nepal/stages/s01_probe.py` | reads the ignore list, stores heading columns, `--redo`, capture-time overrides |
| `src/nepal/stages/s02_spine.py` | Strava first in the track, `activities` rows, `--redo` |
| `src/nepal/spine/gps.py` | `GpsPoint.hr`, `GpsPoint.activity_id`, `PREFERENCE`, `interpolate_point` |
| `src/nepal/spine/strava.py` (new) | `Activity`, `read_activities_csv`, `iter_fit_records` (thin), `points_from_records` (pure), `load_strava` |
| `src/nepal/spine/effort.py` | heart-rate term in `exertion` |
| `src/nepal/spine/place.py` (new) | `Placement`, `load_track`, `trek_span`, `describe`, `place_at`, `day_of` |
| `src/nepal/stages/s03_process.py` | the `place` sub-step; `detect_shots` and `build_photo_shots` write uniform rows; `has_face` derived at gate time |
| `src/nepal/process/gate.py` | `has_face` (pure) |
| `src/nepal/process/render.py` | `fps` parameter, `-r` on the output |
| `src/nepal/stages/s05_cut.py` | passes `render.fps` |
| `src/nepal/diagnose.py` | positioned-shots section |
| `src/nepal/cli.py` | `prune`, `--redo` on `s01` and `s02` |
| `config/pipeline.yaml` | new keys (Task 10 lists every one) |
| `pyproject.toml` | `fitdecode` core dependency |
| `docs/STATE.md`, `README.md`, `CLAUDE.md` | what changed, the new commands, the new trap |

---

### Task 1: `db.upsert` refuses rows that do not share their keys

The generic upsert takes its column list from the first row. `detect_shots` only added `lat`/`lon` when interpolation succeeded, so when the first recording was a February Telegram clip outside the track, every video shot lost its position silently. Make that loud, then make `detect_shots` uniform.

**Files:**
- Modify: `src/nepal/db.py:319-333`
- Modify: `src/nepal/stages/s03_process.py:248-256` (the `for row in rows:` loop inside `detect_shots`)
- Test: `tests/test_db.py` (new)

**Interfaces:**
- Produces: `db.upsert(conn, table, key_cols, rows)` raises `ValueError` naming the missing keys when rows disagree; unchanged otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
"""The SQLite layer: schema, migrations, and the generic upsert."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db


def test_upsert_refuses_rows_with_different_keys(tmp_path):
    """Columns come from the first row. A row that lacks a key would have
    that column silently dropped for every row -- which is how 930 video
    shots lost their position: the first row had no `lat`."""
    conn = db.init(tmp_path / "t.sqlite")
    rows = [{"key": "a", "value": "1", "confidence": 1.0, "method": "m"},
            {"key": "b", "value": "2", "method": "m"}]          # no confidence
    with pytest.raises(ValueError) as exc:
        db.upsert(conn, "decisions", ["key"], rows)
    assert "confidence" in str(exc.value)


def test_upsert_writes_every_column_of_uniform_rows(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    rows = [{"key": "a", "value": "1", "confidence": 1.0, "method": "m"},
            {"key": "b", "value": "2", "confidence": 0.5, "method": "n"}]
    assert db.upsert(conn, "decisions", ["key"], rows) == 2
    got = {r["key"]: r["confidence"] for r in conn.execute("SELECT key, confidence FROM decisions")}
    assert got == {"a": 1.0, "b": 0.5}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /data/projects/nepalvideo/.claude/worktrees/film-v2-design && /data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_db.py -v`
Expected: `test_upsert_refuses_rows_with_different_keys` FAILS with `KeyError: 'confidence'` (not a ValueError); the second test passes.

- [ ] **Step 3: Make `upsert` check the keys**

In `src/nepal/db.py` replace the body of `upsert`:

```python
def upsert(conn: sqlite3.Connection, table: str, key_cols: Sequence[str],
           rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    # The column list comes from the first row, so a row missing a key would
    # have that column dropped for *every* row -- silently. That is how 930
    # video shots lost their position: the first recording was a Telegram
    # clip outside the GPS track, so it had no `lat`, so nobody did.
    want = set(cols)
    for i, r in enumerate(rows):
        have = set(r.keys())
        if have != want:
            missing = sorted(want - have)
            extra = sorted(have - want)
            raise ValueError(
                f"upsert into {table}: row {i} does not share the first row's "
                f"keys (missing {missing}, extra {extra}). Every row must carry "
                f"every column, with None where there is no value.")
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in key_cols)
    sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT({','.join(key_cols)}) DO UPDATE SET {updates}"
           if updates else
           f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})")
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return len(rows)
```

- [ ] **Step 4: Make `detect_shots` rows uniform**

In `src/nepal/stages/s03_process.py`, inside `detect_shots`, replace

```python
        for row in rows:
            ts = rec_start + timedelta(seconds=row["start_s"]) if rec_start else None
            row["start_utc"] = ts.isoformat() if ts else None
            row["act"] = acts_mod.act_for(ts, bounds) if (ts and bounds) else None
            pos = gps_mod.interpolate_at(track, ts, max_gap_s=max_gap) \
                if (ts and track) else None
            if pos:
                row["lat"], row["lon"] = pos
```

with

```python
        for row in rows:
            ts = rec_start + timedelta(seconds=row["start_s"]) if rec_start else None
            row["start_utc"] = ts.isoformat() if ts else None
            row["act"] = acts_mod.act_for(ts, bounds) if (ts and bounds) else None
            pos = gps_mod.interpolate_at(track, ts, max_gap_s=max_gap) \
                if (ts and track) else None
            # Every row carries every column: db.upsert takes its column list
            # from the first row and refuses rows that differ.
            row["lat"], row["lon"] = pos if pos else (None, None)
```

- [ ] **Step 5: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_db.py tests/test_shots.py tests/test_s03_photos.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nepal/db.py src/nepal/stages/s03_process.py tests/test_db.py
git commit -m "An upsert row missing a key drops that column for every row

db.upsert took its columns from the first row and read the rest by key. A
row without a key raised KeyError -- but detect_shots only set lat/lon when
interpolation succeeded, and the first recording is a Telegram clip from
February outside the track, so the whole batch went in with no lat column
and 930 video shots were never positioned. Now the upsert refuses rows
that do not share keys, and detect_shots writes None where there is no
position.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `nepal prune` — remove what the pipeline no longer produces

`phone_kulikov_IMG` (302 clips merged, 61 surviving shots, 9 in the draft) and `phone_keller_IMG` survive because the chapter fix landed after S01.2 had run and `db.upsert` never removes. The prune step re-derives the grouping from the assets already in the database (no probing) and removes stale recordings, their shots, timeline rows, stage units and work files.

**Files:**
- Create: `src/nepal/prune.py`
- Modify: `src/nepal/cli.py` (add the `prune` sub-command)
- Test: `tests/test_prune.py` (new)

**Interfaces:**
- Consumes: `nepal.probe.chapters.group_recordings(assets)` (existing; assets need `asset_id, s3_key, source, kind, container, duration_s, created_at, created_at_utc, filename`).
- Produces:
  - `prune.PrunePlan` dataclass: `recordings: list[str]`, `shots: list[str]`, `files: list[Path]`, `reasons: dict[str, str]`, property `is_empty`.
  - `prune.plan_prune(*, recordings, assets, shots, work_root) -> PrunePlan` (pure).
  - `prune.execute(conn, plan, *, dry_run=False) -> dict`.
  - `prune.run(cfg, *, dry_run=False) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prune.py
"""nepal prune: db.upsert never removes, so something must."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal import db, prune


def _seed(tmp_path):
    conn = db.init(tmp_path / "db" / "nepal.sqlite")
    work = tmp_path / "work"
    for d in ("proxies", "audio", "transcripts/shots"):
        (work / d).mkdir(parents=True)
    conn.executemany(
        "INSERT INTO assets(asset_id, s3_key, source, kind, container, duration_s, "
        "created_at, created_at_utc, recording_id) VALUES (?,?,?,?,?,?,?,?,?)",
        [("a1", "raw/media_from_phones/kulikov/IMG_0001.MOV", "phone_kulikov",
          "video_flat", "mov", 4.0, "2024-05-01T01:00:00+00:00",
          "2024-05-01T01:00:00+00:00", "phone_kulikov_IMG_0001"),
         ("a2", "raw/media_from_phones/kulikov/IMG_0002.MOV", "phone_kulikov",
          "video_flat", "mov", 3.0, "2024-05-01T02:00:00+00:00",
          "2024-05-01T02:00:00+00:00", "phone_kulikov_IMG_0002")])
    conn.executemany(
        "INSERT INTO recordings(recording_id, source, is_360, start_utc, duration_s, "
        "asset_count) VALUES (?,?,?,?,?,?)",
        [("phone_kulikov_IMG_0001", "phone_kulikov", 0, "2024-05-01T01:00:00+00:00", 4.0, 1),
         ("phone_kulikov_IMG_0002", "phone_kulikov", 0, "2024-05-01T02:00:00+00:00", 3.0, 1),
         # the ghost: produced by the old grouping rule, owns no asset any more
         ("phone_kulikov_IMG", "phone_kulikov", 0, "2024-04-27T15:12:57+00:00", 1554.0, 302)])
    conn.executemany(
        "INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, status) "
        "VALUES (?,?,?,?,?,?)",
        [("phone_kulikov_IMG#0000", "phone_kulikov_IMG", "video", 0.0, 20.0, "shortlisted"),
         ("phone_kulikov_IMG_0001#0000", "phone_kulikov_IMG_0001", "video", 0.0, 4.0, "candidate")])
    conn.execute("INSERT INTO timeline(slot_index, act, shot_id, t_in, t_out) "
                 "VALUES (0, 1, 'phone_kulikov_IMG#0000', 0.0, 4.0)")
    conn.execute("INSERT INTO stage_units(stage, unit_id, status) VALUES ('S03', 'proxy:phone_kulikov_IMG', 'done')")
    conn.commit()
    for name in ("proxies/phone_kulikov_IMG_eq.mp4", "audio/phone_kulikov_IMG.wav",
                 "transcripts/shots/phone_kulikov_IMG_0005.json",
                 "proxies/phone_kulikov_IMG_0001_eq.mp4",
                 "proxies/camera_20240101_000000_eq.mp4"):      # a proxy the DB never heard of
        (work / name).write_bytes(b"x")
    return conn, work


def _plan(conn, work):
    recordings = [dict(r) for r in conn.execute("SELECT * FROM recordings")]
    assets = [dict(r) for r in conn.execute("SELECT * FROM assets")]
    shots = [dict(r) for r in conn.execute("SELECT shot_id, recording_id, asset_id FROM shots")]
    return prune.plan_prune(recordings=recordings, assets=assets, shots=shots, work_root=work)


def test_plan_finds_the_ghost_and_its_artefacts(tmp_path):
    conn, work = _seed(tmp_path)
    plan = _plan(conn, work)
    assert plan.recordings == ["phone_kulikov_IMG"]
    assert plan.shots == ["phone_kulikov_IMG#0000"]
    names = {p.name for p in plan.files}
    assert names == {"phone_kulikov_IMG_eq.mp4", "phone_kulikov_IMG.wav",
                     "phone_kulikov_IMG_0005.json", "camera_20240101_000000_eq.mp4"}
    assert "phone_kulikov_IMG_0001_eq.mp4" not in names
    assert "phone_kulikov_IMG" in plan.reasons


def test_execute_removes_rows_and_files_and_keeps_the_rest(tmp_path):
    conn, work = _seed(tmp_path)
    report = prune.execute(conn, _plan(conn, work))
    assert report["n_recordings"] == 1 and report["n_shots"] == 1 and report["n_files"] == 4
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM stage_units").fetchone()[0] == 0
    assert not (work / "proxies" / "phone_kulikov_IMG_eq.mp4").exists()
    assert (work / "proxies" / "phone_kulikov_IMG_0001_eq.mp4").exists()


def test_dry_run_changes_nothing(tmp_path):
    conn, work = _seed(tmp_path)
    report = prune.execute(conn, _plan(conn, work), dry_run=True)
    assert report["dry_run"] is True
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 3
    assert (work / "proxies" / "phone_kulikov_IMG_eq.mp4").exists()


def test_nothing_to_prune_is_an_empty_plan(tmp_path):
    conn, work = _seed(tmp_path)
    prune.execute(conn, _plan(conn, work))
    again = _plan(conn, work)
    assert again.is_empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_prune.py -v`
Expected: FAIL with `ImportError: cannot import name 'prune'`.

- [ ] **Step 3: Write the module**

```python
# src/nepal/prune.py
"""Remove what the pipeline no longer produces.

``db.upsert`` never removes. A recording produced by an older grouping rule,
a proxy for a file that has since been deleted, a shot of either: all of it
survives every re-run and stays eligible for selection. On this corpus the
first draft cut carried nine slots from ``phone_kulikov_IMG`` -- 302 phone
clips the old rule had merged into one 26-minute "recording", timestamped
from the first clip, so Larke Pass footage landed in the Planning act.

The plan is pure: given the rows and the files, decide what goes. Execution
is a handful of DELETEs and unlinks. The grouping is re-derived from the
assets already in the database, so this never re-probes anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from nepal import db
from nepal.probe import chapters

log = logging.getLogger(__name__)


@dataclass
class PrunePlan:
    recordings: list[str] = field(default_factory=list)
    shots: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.recordings or self.shots or self.files)


def plan_prune(*, recordings: Sequence[Mapping[str, Any]],
               assets: Sequence[Mapping[str, Any]],
               shots: Sequence[Mapping[str, Any]],
               work_root: Path) -> PrunePlan:
    """What the current grouping and the current files do not account for."""
    rows = []
    for a in assets:
        a = dict(a)
        a["filename"] = Path(a["s3_key"]).name
        rows.append(a)
    current = {r.recording_id for r in chapters.group_recordings(rows)}
    owned = {a.get("recording_id") for a in assets if a.get("recording_id")}

    plan = PrunePlan()
    for r in recordings:
        rid = str(r["recording_id"])
        if rid not in current:
            plan.reasons[rid] = "not produced by the current grouping rule"
        elif rid not in owned:
            plan.reasons[rid] = "owns no asset"
        else:
            continue
        plan.recordings.append(rid)
    stale = set(plan.recordings)
    plan.shots = [str(s["shot_id"]) for s in shots if s.get("recording_id") in stale]

    transcripts = work_root / "transcripts" / "shots"
    for rid in plan.recordings:
        for p in (work_root / "proxies" / f"{rid}_eq.mp4",
                  work_root / "audio" / f"{rid}.wav"):
            if p.exists():
                plan.files.append(p)
        if transcripts.is_dir():
            plan.files.extend(sorted(transcripts.glob(f"{rid}_*.json")))

    # A proxy for a recording the database does not know at all is the same
    # kind of leftover: the file was deleted from data_root, the manifest
    # dropped the asset, and the 230 MB proxy stayed.
    known = {str(r["recording_id"]) for r in recordings}
    proxies = work_root / "proxies"
    if proxies.is_dir():
        for p in sorted(proxies.glob("*_eq.mp4")):
            rid = p.name[:-len("_eq.mp4")]
            if rid not in known and rid not in current:
                plan.files.append(p)
                plan.reasons[rid] = "proxy for a recording the database does not hold"
    return plan


def execute(conn, plan: PrunePlan, *, dry_run: bool = False) -> dict[str, Any]:
    """Apply the plan. Rows first (children before parents), then files."""
    if not dry_run and not plan.is_empty:
        conn.executemany("DELETE FROM timeline WHERE shot_id=?", [(s,) for s in plan.shots])
        conn.executemany("DELETE FROM shots WHERE shot_id=?", [(s,) for s in plan.shots])
        conn.executemany("DELETE FROM recordings WHERE recording_id=?",
                         [(r,) for r in plan.recordings])
        conn.executemany("DELETE FROM stage_units WHERE unit_id=? OR unit_id=?",
                         [(r, f"proxy:{r}") for r in plan.recordings])
        conn.commit()
        for p in plan.files:
            try:
                p.unlink()
            except OSError as exc:
                log.warning("prune: could not remove %s: %s", p, exc)
    for rid in plan.recordings:
        log.info("prune: recording %s -- %s", rid, plan.reasons.get(rid, ""))
    return {"dry_run": dry_run, "n_recordings": len(plan.recordings),
            "n_shots": len(plan.shots), "n_files": len(plan.files),
            "bytes": sum(p.stat().st_size for p in plan.files if p.exists()),
            "recordings": plan.recordings, "reasons": plan.reasons,
            "files": [str(p) for p in plan.files]}


def run(cfg, *, dry_run: bool = False) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    recordings = [dict(r) for r in conn.execute("SELECT * FROM recordings")]
    assets = [dict(r) for r in conn.execute(
        "SELECT asset_id, s3_key, source, kind, container, duration_s, "
        "created_at, created_at_utc, recording_id FROM assets")]
    shots = [dict(r) for r in conn.execute(
        "SELECT shot_id, recording_id, asset_id FROM shots")]
    plan = plan_prune(recordings=recordings, assets=assets, shots=shots,
                      work_root=cfg.work_root)
    report = execute(conn, plan, dry_run=dry_run)
    conn.close()
    log.info("prune%s: %d recording(s), %d shot(s), %d file(s), %.0f MB",
             " (dry run)" if dry_run else "", report["n_recordings"],
             report["n_shots"], report["n_files"], report["bytes"] / 1e6)
    return report
```

- [ ] **Step 4: Add the sub-command**

In `src/nepal/cli.py`, after the `decisions`/`report`/`doctor` parsers (line ~98) add:

```python
    pprune = sub.add_parser("prune", help="remove recordings, shots and work files "
                                          "the pipeline no longer produces")
    pprune.add_argument("--dry-run", action="store_true",
                        help="report what would go without touching anything")
```

and in the dispatch, before `if args.cmd == "decisions":`:

```python
    if args.cmd == "prune":
        from nepal import prune
        rep = prune.run(cfg, dry_run=args.dry_run)
        tag = "would remove" if rep["dry_run"] else "removed"
        print(f"prune: {tag} {rep['n_recordings']} recording(s), {rep['n_shots']} shot(s), "
              f"{rep['n_files']} file(s), {rep['bytes'] / 1e6:.0f} MB")
        for rid in rep["recordings"]:
            print(f"  {rid}: {rep['reasons'].get(rid, '')}")
        return 0
```

Also add `nepal prune [--dry-run]` to the module docstring list at the top of `cli.py`.

- [ ] **Step 5: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_prune.py tests/test_chapters.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nepal/prune.py src/nepal/cli.py tests/test_prune.py
git commit -m "nepal prune: remove what the pipeline no longer produces

The chapter fix that stopped merging every phone clip into one recording
landed after S01.2 had run, and db.upsert never removes, so the merged
ghosts survived with their 230 MB proxies and 61 surviving shots -- nine
of which the first draft placed in the Planning act with Larke Pass in
them. The plan is pure; execution deletes children before parents.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Manifest ignore list and the heading tags

`awscliv2.zip` was ingested; the new `strava/` folder would be. iPhone stills carry a true-north heading and a 35 mm focal length that §13.3 needs. The tag list must stay derived: a tag that is not requested does not exist.

**Files:**
- Modify: `src/nepal/probe/manifest.py` (`HEADING_TAGS`, `parse_heading`, `walk_media(ignore_globs=…)`)
- Modify: `src/nepal/util/proc.py:160-171` (`EXIF_TAGS`)
- Modify: `src/nepal/db.py:200-216` (`MIGRATIONS`)
- Modify: `src/nepal/stages/s01_probe.py:55` (walk) and `:144-167` (row)
- Test: `tests/test_manifest.py` (extend), `tests/test_db.py` (extend)

**Interfaces:**
- Produces: `manifest.HEADING_TAGS = ("GPSImgDirection", "GPSImgDirectionRef", "GPSHPositioningError", "FocalLengthIn35mmFormat")`; `manifest.parse_heading(row) -> dict` with keys `heading_deg`, `heading_ref`, `pos_error_m`, `focal_35mm`; `manifest.walk_media(root, *, skip_hidden=True, exclude_dirs=None, ignore_globs=())`; `assets` columns `heading_deg REAL`, `heading_ref TEXT`, `pos_error_m REAL`, `focal_35mm REAL`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_manifest.py`:

```python
from nepal.probe.manifest import HEADING_TAGS, parse_heading


def test_heading_tags_are_requested_from_exiftool():
    """A tag that is ranked but never requested does not exist."""
    from nepal.util.proc import EXIF_TAGS
    for tag in HEADING_TAGS:
        assert f"-{tag}" in EXIF_TAGS


def test_parse_heading_reads_iphone_fields():
    row = {"GPS:GPSImgDirection": 34.23838045, "GPS:GPSImgDirectionRef": "T",
           "GPS:GPSHPositioningError": 6.460409194,
           "ExifIFD:FocalLengthIn35mmFormat": 26}
    h = parse_heading(row)
    assert h["heading_deg"] == pytest.approx(34.238, abs=1e-3)
    assert h["heading_ref"] == "T"
    assert h["pos_error_m"] == pytest.approx(6.46, abs=1e-2)
    assert h["focal_35mm"] == 26.0


def test_parse_heading_is_all_none_without_the_tags():
    assert parse_heading({}) == {"heading_deg": None, "heading_ref": None,
                                 "pos_error_m": None, "focal_35mm": None}


def test_walk_media_skips_ignored_globs_and_dirs(tmp_path):
    (tmp_path / "media_from_phones" / "keller").mkdir(parents=True)
    (tmp_path / "media_from_phones" / "keller" / "IMG_1.HEIC").write_bytes(b"x")
    (tmp_path / "aws").mkdir()
    (tmp_path / "awscliv2.zip").write_bytes(b"x")
    (tmp_path / "strava").mkdir()
    (tmp_path / "strava" / "activities.csv").write_text("a")
    (tmp_path / "chat_export" / "files").mkdir(parents=True)
    (tmp_path / "chat_export" / "files" / "KTM_Hotel.pdf_thumb.jpg").write_bytes(b"x")
    (tmp_path / "chat_export" / "files" / "KTM_Hotel.pdf").write_bytes(b"x")
    got = {p.relative_to(tmp_path).as_posix() for p in manifest_walk(
        tmp_path, exclude_dirs={"aws", "strava"},
        ignore_globs=("*.zip", "*_thumb.jpg"))}
    assert got == {"media_from_phones/keller/IMG_1.HEIC", "chat_export/files/KTM_Hotel.pdf"}
```

and at the top of the file, next to the existing import, add `from nepal.probe.manifest import walk_media as manifest_walk`.

Append to `tests/test_db.py`:

```python
def test_assets_carry_the_heading_columns(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(assets)")}
    assert {"heading_deg", "heading_ref", "pos_error_m", "focal_35mm"} <= cols
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_manifest.py tests/test_db.py -v`
Expected: the four new manifest tests and the db test FAIL (ImportError on `HEADING_TAGS`, then missing columns).

- [ ] **Step 3: Implement**

In `src/nepal/probe/manifest.py`, after `CAPTURE_TAGS` (line 191) add:

```python
# The camera's heading and lens, where the device recorded them. An iPhone
# still carries GPSImgDirection against true north and a 35 mm-equivalent
# focal length, which together say where the frame was pointing and how wide
# it is -- enough to compute where a named summit falls in the picture
# (spec section 13.3). Requested from exiftool by the same derivation as
# CAPTURE_TAGS: a tag that is not asked for does not exist.
HEADING_TAGS = ("GPSImgDirection", "GPSImgDirectionRef",
                "GPSHPositioningError", "FocalLengthIn35mmFormat")


def parse_heading(row: dict[str, Any]) -> dict[str, Any]:
    """Heading in degrees, its reference ('T' true / 'M' magnetic), the
    horizontal positioning error in metres, and the 35 mm-equivalent focal
    length. None wherever the device wrote nothing."""
    ref = exif_get(row, "GPSImgDirectionRef")
    return {
        "heading_deg": _num(exif_get(row, "GPSImgDirection")),
        "heading_ref": str(ref).strip()[:1].upper() if ref is not None else None,
        "pos_error_m": _num(exif_get(row, "GPSHPositioningError")),
        "focal_35mm": _num(exif_get(row, "FocalLengthIn35mmFormat")),
    }
```

Replace `walk_media` with:

```python
def walk_media(root: Path, *, skip_hidden: bool = True,
               exclude_dirs: frozenset[str] | set[str] | None = None,
               ignore_globs: Sequence[str] = ()) -> list[Path]:
    """Every regular media-bearing file under root, sorted for deterministic
    asset ordering. Excluded directories are pruned rather than filtered, so a
    large tree of irrelevant files costs nothing to skip. ``ignore_globs``
    match the file name or the relative path: an installer zip, a PDF
    thumbnail Telegram wrote next to the PDF, a spreadsheet lock file."""
    import fnmatch
    excluded = {d.lower() for d in
                (DEFAULT_EXCLUDE_DIRS if exclude_dirs is None else exclude_dirs)}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if skip_hidden and any(part.startswith(".") for part in parts):
            continue
        if any(part.lower() in excluded for part in parts[:-1]):
            continue
        if any(fnmatch.fnmatch(p.name, g) or fnmatch.fnmatch(rel.as_posix(), g)
               for g in ignore_globs):
            continue
        out.append(p)
    return out
```

In `src/nepal/util/proc.py`, change `_capture_tag_args` so both lists feed `EXIF_TAGS`:

```python
    from nepal.probe.manifest import CAPTURE_TAGS, HEADING_TAGS
    return [f"-{tag}" for tag in (*CAPTURE_TAGS, *HEADING_TAGS)]
```

In `src/nepal/db.py` extend `MIGRATIONS`:

```python
    # Where the phone was pointing and how wide the lens was, for placing a
    # named summit in the frame (spec section 13.3).
    ("assets", "heading_deg", "REAL"),
    ("assets", "heading_ref", "TEXT"),
    ("assets", "pos_error_m", "REAL"),
    ("assets", "focal_35mm", "REAL"),
```

In `src/nepal/stages/s01_probe.py::build_manifest`:

```python
    files = manifest.walk_media(
        root,
        exclude_dirs=set(manifest.DEFAULT_EXCLUDE_DIRS)
        | {str(d) for d in (cfg.get("probe.exclude_dirs", []) or [])},
        ignore_globs=tuple(cfg.get("probe.ignore_globs", []) or []))
```

and in the `rows.append({...})` dict, after `"probe_json": probe_json,` add `**manifest.parse_heading(exif),`.

- [ ] **Step 4: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_manifest.py tests/test_db.py tests/test_s01_replay.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nepal/probe/manifest.py src/nepal/util/proc.py src/nepal/db.py src/nepal/stages/s01_probe.py tests/test_manifest.py tests/test_db.py
git commit -m "Ask exiftool for the heading and the lens, and ignore what is not media

iPhone stills record GPSImgDirection against true north and a 35 mm-
equivalent focal length: together they say where the frame pointed and
how wide it was, which is what placing Manaslu in the picture needs.
Requested by the same derivation as the capture tags, so they cannot
drift from the reader. The manifest also gains an ignore list: an AWS
installer zip was an asset, and the Strava export would have become one.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `--redo` for S01 and S02, and capture-time overrides

`nepal s01 --force` re-solves FOV and clocks (an hour). The spec asks for `nepal s01 --redo fov` and the manifest re-probe needs `--redo manifest,chapters`. S02 has the same need. The three undated Kulikov clips get a config override.

**Files:**
- Modify: `src/nepal/stages/s01_probe.py:770-905` (`apply_offsets`, `run`)
- Modify: `src/nepal/stages/s02_spine.py:909-941` (`run`)
- Modify: `src/nepal/cli.py` (arguments and the two calls)
- Test: `tests/test_stage_redo.py` (new)

**Interfaces:**
- Produces: `s01_probe.run(cfg, *, force=False, skip_fov=False, skip_clock=False, redo=None)`; `s02_spine.run(cfg, *, force=False, skip_asr=False, redo=None)`; `s01_probe.apply_offsets(conn, offsets, overrides=None) -> int`; config `probe.capture_time_overrides: {filename: iso-8601}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage_redo.py
"""--redo re-runs a named sub-step without paying for the others."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db
from nepal.config import Config
from nepal.stages import s01_probe


def _cfg(tmp_path):
    data = tmp_path / "data"
    (data / "media_from_phones" / "kulikov").mkdir(parents=True)
    (data / "media_from_phones" / "kulikov" / "note.txt").write_text("not media")
    return Config({
        "project": {"data_root": str(data), "work_root": str(tmp_path / "work"),
                    "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
        "probe": {"max_chapter_gap_s": 1.0, "exclude_dirs": [], "ignore_globs": [],
                  "capture_time_overrides": {}},
    })


def test_s01_redo_reruns_only_the_named_unit(tmp_path):
    cfg = _cfg(tmp_path)
    first = s01_probe.run(cfg, skip_fov=True, skip_clock=True)
    assert "n_assets" in first["manifest"] and "n_recordings" in first["chapters"]
    second = s01_probe.run(cfg, skip_fov=True, skip_clock=True)
    assert second["manifest"] == {"skipped": "already done"}
    third = s01_probe.run(cfg, skip_fov=True, skip_clock=True, redo={"manifest"})
    assert "n_assets" in third["manifest"]
    assert third["chapters"] == {"skipped": "already done"}


def test_s01_redo_rejects_an_unknown_unit(tmp_path):
    with pytest.raises(SystemExit):
        s01_probe.run(_cfg(tmp_path), skip_fov=True, skip_clock=True, redo={"nonsense"})


def test_capture_time_override_sets_created_at_utc_by_filename(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at) "
                 "VALUES ('x', 'raw/media_from_phones/kulikov/video_1.mp4', "
                 "'phone_kulikov', 'video_flat', '2025-11-22T23:46:56+00:00')")
    conn.commit()
    n = s01_probe.apply_offsets(conn, {"kulikov": 0.0},
                                overrides={"video_1.mp4": "2024-05-12T09:30:00+05:45"})
    assert n == 1
    got = conn.execute("SELECT created_at_utc FROM assets WHERE asset_id='x'").fetchone()[0]
    assert got == "2024-05-12T03:45:00+00:00"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_stage_redo.py -v`
Expected: FAIL — `run() got an unexpected keyword argument 'redo'`, `apply_offsets() got an unexpected keyword argument 'overrides'`.

- [ ] **Step 3: Implement in S01**

In `src/nepal/stages/s01_probe.py`, change `apply_offsets`:

```python
def apply_offsets(conn, offsets: dict[str, float],
                  overrides: dict[str, str] | None = None) -> int:
    """Write ``created_at_utc`` = created_at + offset for every asset.

    The reference device takes offset 0; every other device takes the offset
    S01.5 measured for it. Telegram timestamps come from the server rather than
    a device, so they are already correct and are never shifted.

    ``overrides`` maps a file name to a capture time the operator knows and
    the file does not: three phone clips on this corpus carry only an export
    date, and no offset can recover a time that was never recorded.
    """
```

then after the `n += 1` loop and before the recordings UPDATE, add:

```python
    for name, when in (overrides or {}).items():
        dt = manifest.parse_exif_datetime(when)
        if dt is None:
            log.warning("S01 capture_time_overrides: cannot parse %r for %s", when, name)
            continue
        cur = conn.execute("UPDATE assets SET created_at_utc=? WHERE s3_key LIKE ?",
                           (dt.astimezone(timezone.utc).isoformat(), f"%/{name}"))
        if cur.rowcount:
            log.info("S01 %s: capture time set to %s by operator override", name, when)
        else:
            log.warning("S01 capture_time_overrides names %s, which is not an asset", name)
    conn.commit()
```

Change `run`:

```python
S01_UNITS = ("manifest", "chapters", "fov", "clock")


def run(cfg: Config, *, force: bool = False, skip_fov: bool = False,
        skip_clock: bool = False, redo: set[str] | None = None) -> dict[str, Any]:
    redo = set(redo or ())
    unknown = redo - set(S01_UNITS)
    if unknown:
        raise SystemExit(f"unknown --redo unit(s): {', '.join(sorted(unknown))}; "
                         f"valid: {', '.join(S01_UNITS)}")
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    done = db.done_units(conn, STAGE)
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force,
        rerun_hint="nepal s01 --force --skip-fov --skip-clock")
    overrides = dict(cfg.get("probe.capture_time_overrides", {}) or {})

    def wanted(unit: str) -> bool:
        return force or unit in redo or unit not in done
```

and replace each `if force or "<unit>" not in done:` with `if wanted("<unit>"):` (manifest, chapters, fov, clock), set `clock_will_run = not skip_clock and wanted("clock")`, and pass `overrides` to both `apply_offsets` calls: `apply_offsets(conn, offsets, overrides)`.

- [ ] **Step 4: Implement in S02**

In `src/nepal/stages/s02_spine.py::run`:

```python
S02_UNITS = ("gps_track", "telegram", "geotag", "asr", "music", "acts")


def run(cfg: Config, *, force: bool = False, skip_asr: bool = False,
        redo: set[str] | None = None) -> dict[str, Any]:
    redo = set(redo or ())
    unknown = redo - set(S02_UNITS)
    if unknown:
        raise SystemExit(f"unknown --redo unit(s): {', '.join(sorted(unknown))}; "
                         f"valid: {', '.join(S02_UNITS)}")
```

and in the loop: `if not force and name in done and name not in redo:`.

- [ ] **Step 5: Wire the CLI**

In `src/nepal/cli.py`: add to `p1` and `p2`

```python
    p1.add_argument("--redo", metavar="UNITS", default="",
                    help="comma-separated sub-steps to recompute: manifest,chapters,fov,clock")
    p2.add_argument("--redo", metavar="UNITS", default="",
                    help="comma-separated sub-steps to recompute: gps_track,telegram,geotag,asr,music,acts")
```

and pass them: `s01_probe.run(cfg, force=args.force, skip_fov=args.skip_fov, skip_clock=args.skip_clock, redo={x.strip() for x in args.redo.split(",") if x.strip()})` and likewise for `s02_spine.run(..., redo=...)`. Update the docstring lines at the top of `cli.py`.

- [ ] **Step 6: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_stage_redo.py tests/test_s01_replay.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/nepal/stages/s01_probe.py src/nepal/stages/s02_spine.py src/nepal/cli.py tests/test_stage_redo.py
git commit -m "--redo for S01 and S02, and a capture-time override for undated clips

Re-solving the FOV or re-probing the manifest cost --force, which also
re-runs an hour of clock correlation. Named sub-steps are now reachable
the way S03's already were. Three phone clips carry only an export date;
an operator who knows when they were shot can now say so in the config.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Strava is the spine

Eleven FIT files, one per trekking day, with a fix every second, barometric altitude and heart rate. They outrank photo EXIF wherever they exist.

**Files:**
- Create: `src/nepal/spine/strava.py`
- Modify: `src/nepal/spine/gps.py:26-36` (`GpsPoint`), `:108` (`PREFERENCE`)
- Modify: `src/nepal/db.py` (`SCHEMA`: `activities` table; `MIGRATIONS`: `gps_points` columns)
- Modify: `src/nepal/stages/s02_spine.py:50-108` (`build_gps_track`)
- Modify: `pyproject.toml:13-24`
- Test: `tests/test_strava.py` (new), `tests/test_gps.py` (extend), `tests/test_db.py` (extend)

**Interfaces:**
- Produces:
  - `gps.GpsPoint(ts, lat, lon, ele=None, source="unknown", accuracy_m=None, hr=None, activity_id=None)`; `gps.PREFERENCE = ("strava", "gpx", "phone_keller", "phone_kulikov")`.
  - `strava.Activity` dataclass: `activity_id, name, kind, start_utc: datetime, elapsed_s, moving_s, distance_m, gain_m, hr_max, hr_avg, filename`; property `end_utc`.
  - `strava.read_activities_csv(path) -> list[Activity]`.
  - `strava.iter_fit_records(path) -> Iterator[dict]` (thin wrapper over `fitdecode`, gzip-aware).
  - `strava.points_from_records(records, *, activity_id, source="strava", sample_s=0.0) -> list[GpsPoint]` (pure).
  - `strava.load_strava(strava_dir, *, sample_s) -> tuple[list[Activity], list[GpsPoint], dict]`.
  - Table `activities(activity_id PK, name, kind, start_utc, end_utc, elapsed_s, moving_s, distance_m, gain_m, hr_max, hr_avg, filename, n_points)`; `gps_points` columns `hr_bpm REAL`, `alt_baro_m REAL`, `activity_id TEXT`.

- [ ] **Step 1: Install the decoder and declare it**

In `pyproject.toml` `dependencies`, after `"python-dateutil>=2.8",` add:

```python
  # Strava exports Apple Watch activities as FIT. fitdecode is pure Python
  # and decodes the record messages (position, altitude, heart rate) with
  # no native build step, which keeps the remote box's bootstrap trivial.
  "fitdecode>=0.10",
```

Run: `/data/projects/nepalvideo/.venv/bin/pip install -e /data/projects/nepalvideo/.claude/worktrees/film-v2-design 2>&1 | tail -1`
Expected: `Successfully installed ... fitdecode-...`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_strava.py
"""Strava: the trek at one fix per second, with heart rate."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal.spine import strava
from nepal.spine.gps import GpsPoint, merge_points

UTC = timezone.utc
T0 = datetime(2024, 5, 6, 4, 2, 34, tzinfo=UTC)

CSV = (
    "Activity ID,Activity Date,Activity Name,Activity Type,Activity Description,"
    "Elapsed Time,Distance,Max Heart Rate,Relative Effort,Commute,Activity Private Note,"
    "Activity Gear,Filename,Athlete Weight,Bike Weight,Elapsed Time,Moving Time,Distance,"
    "Max Speed,Average Speed,Elevation Gain,Elevation Loss,Elevation Low,Elevation High,"
    "Max Grade,Average Grade,Average Positive Grade,Average Negative Grade,Max Cadence,"
    "Average Cadence,Max Heart Rate,Average Heart Rate\n"
    '11343979900,"May 6, 2024, 4:02:34 AM",Larkya Pass to Bimtang,Hike,,18485,12.36,128,'
    ',false,,,activities/12116271787.fit.gz,,,18485,15099,12360.4,1.781,0.819,718.8,'
    ',3705,5154.2,,,,,,,128,111\n'
)


def test_activities_csv_takes_the_metric_columns_not_the_display_ones(tmp_path):
    """Strava writes 'Distance' twice: 12.36 (km, for people) and 12360.4 (m).
    The last occurrence is the one in base units."""
    p = tmp_path / "activities.csv"
    p.write_text(CSV, encoding="utf-8")
    acts = strava.read_activities_csv(p)
    assert len(acts) == 1
    a = acts[0]
    assert a.activity_id == "11343979900"
    assert a.name == "Larkya Pass to Bimtang" and a.kind == "Hike"
    assert a.start_utc == T0
    assert a.elapsed_s == 18485 and a.moving_s == 15099
    assert a.distance_m == pytest.approx(12360.4)
    assert a.gain_m == pytest.approx(718.8)
    assert a.hr_max == 128 and a.hr_avg == 111
    assert a.filename == "activities/12116271787.fit.gz"
    assert a.end_utc == T0 + timedelta(seconds=18485)


def _rec(seconds, lat_deg, lon_deg, ele=4000.0, hr=120):
    semi = 2 ** 31 / 180.0
    return {"timestamp": T0 + timedelta(seconds=seconds),
            "position_lat": int(lat_deg * semi), "position_long": int(lon_deg * semi),
            "enhanced_altitude": ele, "heart_rate": hr}


def test_points_convert_semicircles_and_carry_hr_and_baro():
    pts = strava.points_from_records([_rec(0, 28.65, 84.62, 5106.0, 131)], activity_id="A")
    assert len(pts) == 1
    p = pts[0]
    assert p.lat == pytest.approx(28.65, abs=1e-6) and p.lon == pytest.approx(84.62, abs=1e-6)
    assert p.ele == 5106.0 and p.hr == 131.0
    assert p.source == "strava" and p.activity_id == "A" and p.ts == T0


def test_points_skip_records_without_a_fix_and_downsample():
    recs = [_rec(i, 28.65 + i * 1e-5, 84.62) for i in range(10)]
    recs.insert(3, {"timestamp": T0 + timedelta(seconds=3), "heart_rate": 100})   # indoor gap
    pts = strava.points_from_records(recs, activity_id="A", sample_s=5.0)
    assert [int((p.ts - T0).total_seconds()) for p in pts] == [0, 5]


def test_points_accept_degrees_if_a_decoder_already_converted():
    pts = strava.points_from_records([{"timestamp": T0, "position_lat": 28.65,
                                       "position_long": 84.62}], activity_id="A")
    assert pts[0].lat == pytest.approx(28.65)


def test_naive_timestamps_are_utc():
    pts = strava.points_from_records([{"timestamp": T0.replace(tzinfo=None),
                                       "position_lat": 28.65, "position_long": 84.62}],
                                     activity_id="A")
    assert pts[0].ts.tzinfo is not None and pts[0].ts == T0


def test_strava_outranks_a_photo_fix_at_the_same_second():
    photo = GpsPoint(T0, 28.0, 84.0, None, "phone_keller")
    watch = GpsPoint(T0, 28.65, 84.62, 5106.0, "strava", None, 130.0, "A")
    merged = merge_points([[photo], [watch]])
    assert merged[0].source == "strava"


REAL = os.environ.get("NEPAL_STRAVA_DIR")


@pytest.mark.skipif(not REAL, reason="set NEPAL_STRAVA_DIR to the strava/ export to run")
def test_real_export_decodes_every_activity():
    """Verify at the real boundary: fitdecode against the actual files."""
    acts, pts, rep = strava.load_strava(pathlib.Path(REAL), sample_s=5.0)
    assert len(acts) >= 10
    assert len(pts) > 10_000
    assert all(a.activity_id in rep["points_per_activity"] for a in acts)
    hrs = [p.hr for p in pts if p.hr is not None]
    assert hrs and 40 < min(hrs) and max(hrs) < 220
    assert max(p.ele for p in pts if p.ele is not None) > 5000
```

Append to `tests/test_db.py`:

```python
def test_gps_points_and_activities_carry_the_strava_columns(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    gcols = {r[1] for r in conn.execute("PRAGMA table_info(gps_points)")}
    assert {"hr_bpm", "alt_baro_m", "activity_id"} <= gcols
    acols = {r[1] for r in conn.execute("PRAGMA table_info(activities)")}
    assert {"activity_id", "name", "start_utc", "end_utc", "hr_max", "n_points"} <= acols
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `NEPAL_STRAVA_DIR=/data/projects/nepal_data/strava /data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_strava.py tests/test_db.py -v`
Expected: FAIL with `ImportError: cannot import name 'strava'`.

- [ ] **Step 4: Extend `GpsPoint` and the preference**

In `src/nepal/spine/gps.py`:

```python
@dataclass(frozen=True)
class GpsPoint:
    ts: datetime
    lat: float
    lon: float
    ele: float | None = None
    source: str = "unknown"
    accuracy_m: float | None = None
    # From a watch: heart rate, and which activity the fix belongs to. Both
    # None for a photo fix.
    hr: float | None = None
    activity_id: str | None = None
```

and `PREFERENCE = ("strava", "gpx", "phone_keller", "phone_kulikov")`. Update the module docstring's second paragraph: "A Strava track from a watch outranks everything: a fix a second with barometric altitude and heart rate. A shared ``.gpx`` comes next, then photo EXIF."

- [ ] **Step 5: Write the module**

```python
# src/nepal/spine/strava.py
"""S02.1 -- the Strava export as the track spine.

An Apple Watch on the operator's wrist recorded every trekking day: a fix a
second, barometric altitude and heart rate, exported by Strava as one FIT
file per activity plus ``activities.csv``. Photo EXIF gives a fix every few
minutes; this gives one every second, and the heart rate is the only direct
measure of effort the corpus holds.

The FIT decoding is a thin wrapper around ``fitdecode``; everything that
decides -- unit conversion, sampling, which fields count -- is pure and
tested on dicts.
"""
from __future__ import annotations

import csv
import gzip
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from nepal.spine.gps import GpsPoint

log = logging.getLogger(__name__)

# FIT stores position as signed 32-bit semicircles.
SEMICIRCLE_DEG = 180.0 / 2 ** 31
CSV_DATE = "%b %d, %Y, %I:%M:%S %p"          # 'May 6, 2024, 4:02:34 AM', UTC


@dataclass(frozen=True)
class Activity:
    activity_id: str
    name: str
    kind: str
    start_utc: datetime
    elapsed_s: float
    moving_s: float | None
    distance_m: float | None
    gain_m: float | None
    hr_max: float | None
    hr_avg: float | None
    filename: str

    @property
    def end_utc(self) -> datetime:
        return self.start_utc + timedelta(seconds=self.elapsed_s)


def _num(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_activities_csv(path: str | Path) -> list[Activity]:
    """The activity table. Strava repeats some column names -- 'Distance'
    appears as kilometres for people and again as metres -- and the last
    occurrence is the one in base units, so columns are resolved by their
    last index rather than through DictReader."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return []
        last: dict[str, int] = {}
        for i, name in enumerate(header):
            last[name.strip()] = i

        def col(row: list[str], name: str) -> str | None:
            i = last.get(name)
            return row[i] if i is not None and i < len(row) else None

        out: list[Activity] = []
        for row in reader:
            if not row or not col(row, "Activity ID"):
                continue
            try:
                start = datetime.strptime(str(col(row, "Activity Date")).strip(),
                                          CSV_DATE).replace(tzinfo=timezone.utc)
            except ValueError:
                log.warning("strava: cannot parse date %r", col(row, "Activity Date"))
                continue
            out.append(Activity(
                activity_id=str(col(row, "Activity ID")).strip(),
                name=str(col(row, "Activity Name") or "").strip(),
                kind=str(col(row, "Activity Type") or "").strip(),
                start_utc=start,
                elapsed_s=_num(col(row, "Elapsed Time")) or 0.0,
                moving_s=_num(col(row, "Moving Time")),
                distance_m=_num(col(row, "Distance")),
                gain_m=_num(col(row, "Elevation Gain")),
                hr_max=_num(col(row, "Max Heart Rate")),
                hr_avg=_num(col(row, "Average Heart Rate")),
                filename=str(col(row, "Filename") or "").strip(),
            ))
    out.sort(key=lambda a: a.start_utc)
    return out


def iter_fit_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Every ``record`` message of a FIT file as a plain dict. Thin: the
    only thing here that touches the decoder."""
    import fitdecode
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rb") as fh, fitdecode.FitReader(fh) as reader:
        for frame in reader:
            if getattr(frame, "frame_type", None) != fitdecode.FIT_FRAME_DATA:
                continue
            if frame.name != "record":
                continue
            yield {f.name: f.value for f in frame.fields}


def _degrees(v: Any) -> float | None:
    if v is None:
        return None
    x = float(v)
    # A decoder that already converted returns degrees; raw FIT is semicircles.
    return x if abs(x) <= 180.0 else x * SEMICIRCLE_DEG


def points_from_records(records: Iterable[dict[str, Any]], *, activity_id: str,
                        source: str = "strava", sample_s: float = 0.0
                        ) -> list[GpsPoint]:
    """Records to points. Skips records without a fix, keeps at most one
    point per ``sample_s`` seconds (0 keeps all), reads barometric altitude
    from ``enhanced_altitude`` first, and tags naive timestamps as UTC,
    which is what FIT stores."""
    out: list[GpsPoint] = []
    last_ts: datetime | None = None
    for rec in records:
        lat = _degrees(rec.get("position_lat"))
        lon = _degrees(rec.get("position_long"))
        ts = rec.get("timestamp")
        if lat is None or lon is None or not isinstance(ts, datetime):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if last_ts is not None and (ts - last_ts).total_seconds() < sample_s:
            continue
        ele = rec.get("enhanced_altitude")
        if ele is None:
            ele = rec.get("altitude")
        hr = rec.get("heart_rate")
        out.append(GpsPoint(ts, lat, lon, float(ele) if ele is not None else None,
                            source, None, float(hr) if hr is not None else None,
                            activity_id))
        last_ts = ts
    return out


def load_strava(strava_dir: str | Path, *, sample_s: float = 5.0
                ) -> tuple[list[Activity], list[GpsPoint], dict[str, Any]]:
    """Everything the export holds: activities, points, and a report."""
    root = Path(strava_dir)
    csv_path = root / "activities.csv"
    if not csv_path.exists():
        return [], [], {"skipped": f"no activities.csv in {root}"}
    activities = read_activities_csv(csv_path)
    points: list[GpsPoint] = []
    per: dict[str, int] = {}
    missing: list[str] = []
    for a in activities:
        f = root / a.filename
        if not a.filename or not f.exists():
            missing.append(a.activity_id)
            continue
        try:
            pts = points_from_records(iter_fit_records(f), activity_id=a.activity_id,
                                      sample_s=sample_s)
        except Exception as exc:                      # a bad file, not a bug
            log.warning("strava: %s unreadable (%s)", f.name, exc)
            missing.append(a.activity_id)
            continue
        per[a.activity_id] = len(pts)
        points.extend(pts)
    points.sort(key=lambda p: p.ts)
    log.info("strava: %d activit%s, %d points at >= %.0f s spacing, %d file(s) missing",
             len(activities), "y" if len(activities) == 1 else "ies", len(points),
             sample_s, len(missing))
    return activities, points, {"n_activities": len(activities), "n_points": len(points),
                                "points_per_activity": per, "missing_files": missing}
```

- [ ] **Step 6: Schema**

In `src/nepal/db.py` `SCHEMA`, after the `gps_points` table add:

```sql
-- ADDITION (Film v2 section 13.1): one row per Strava activity, which is one
-- row per trekking day. The name is the day's title and the start is the
-- alpine start nobody wrote down.
CREATE TABLE IF NOT EXISTS activities (
  activity_id TEXT PRIMARY KEY,
  name        TEXT,
  kind        TEXT,
  start_utc   TEXT,
  end_utc     TEXT,
  elapsed_s   REAL,
  moving_s    REAL,
  distance_m  REAL,
  gain_m      REAL,
  hr_max      REAL,
  hr_avg      REAL,
  filename    TEXT,
  n_points    INTEGER
);
```

and to `MIGRATIONS`:

```python
    # From the watch: heart rate and barometric altitude per fix, and the
    # activity the fix belongs to. alt_dem_m keeps holding the *resolved*
    # altitude (barometric where it exists, DEM otherwise), as it always did
    # for GPX elevations; alt_baro_m keeps the raw reading beside it.
    ("gps_points", "hr_bpm", "REAL"),
    ("gps_points", "alt_baro_m", "REAL"),
    ("gps_points", "activity_id", "TEXT"),
```

- [ ] **Step 7: Read Strava in `build_gps_track`**

In `src/nepal/stages/s02_spine.py`, add `from nepal.spine import strava as strava_mod` to the imports, and in `build_gps_track` replace the block from `merged = gps_mod.merge_points([gpx_points, photo_points])` through the `db.upsert(conn, "gps_points", ...)` with:

```python
    # The watch outranks everything: a fix a second, barometric altitude,
    # heart rate. Photo EXIF fills the days it did not run (jeep days, the
    # cities).
    strava_dir = cfg.data_root / str(cfg.get("spine.strava_dir", "strava"))
    activities, strava_points, strava_rep = strava_mod.load_strava(
        strava_dir, sample_s=float(cfg.get("spine.strava_sample_s", 5.0))) \
        if strava_dir.exists() else ([], [], {"skipped": f"no {strava_dir}"})

    merged = gps_mod.merge_points([strava_points, gpx_points, photo_points])
    merged = gps_mod.drop_outliers(merged)

    # S02.3 for the track itself. A barometric (Strava) or GPX elevation
    # outranks the DEM; photo points have none, so they take the DEM value.
    srtm = dem_mod.Srtm(cfg.srtm_dir)
    rows = []
    n_alt = 0
    for p in merged:
        alt, _src = dem_mod.resolve_altitude(srtm.elevation(p.lat, p.lon), p.ele, None)
        if alt is not None:
            n_alt += 1
        rows.append({"ts_utc": p.key(), "lat": p.lat, "lon": p.lon,
                     "alt_dem_m": alt, "source": p.source,
                     "hr_bpm": p.hr,
                     "alt_baro_m": p.ele if p.source == "strava" else None,
                     "activity_id": p.activity_id})
    # A re-run with a different sample spacing must not leave the old
    # spacing's points behind: the track is rebuilt, not accumulated.
    conn.execute("DELETE FROM gps_points")
    db.upsert(conn, "gps_points", ["ts_utc"], rows)
    per = strava_rep.get("points_per_activity", {})
    db.upsert(conn, "activities", ["activity_id"], [{
        "activity_id": a.activity_id, "name": a.name, "kind": a.kind,
        "start_utc": a.start_utc.isoformat(), "end_utc": a.end_utc.isoformat(),
        "elapsed_s": a.elapsed_s, "moving_s": a.moving_s, "distance_m": a.distance_m,
        "gain_m": a.gain_m, "hr_max": a.hr_max, "hr_avg": a.hr_avg,
        "filename": a.filename, "n_points": per.get(a.activity_id, 0),
    } for a in activities])
```

Update the log line and the return value: log `"S02.1 %d GPS points (%d from Strava, %d from photos, %d from GPX), %s .. %s"` with `len(strava_points)` inserted, and add `"n_from_strava": len(strava_points), "strava": strava_rep, "n_activities": len(activities)` to the returned dict.

- [ ] **Step 8: Run the tests**

Run: `NEPAL_STRAVA_DIR=/data/projects/nepal_data/strava /data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_strava.py tests/test_gps.py tests/test_db.py tests/test_effort.py -v`
Expected: all PASS, including the real-export test (it decodes eleven files in a few seconds).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/nepal/spine/strava.py src/nepal/spine/gps.py src/nepal/db.py src/nepal/stages/s02_spine.py tests/test_strava.py tests/test_db.py
git commit -m "Strava is the spine: a fix a second, barometric altitude, heart rate

An Apple Watch recorded every trekking day and Strava exported it as one
FIT file per activity. That is a track sampled every second instead of
every few minutes, altitude from a barometer instead of a DEM, and the
only direct measure of effort in the corpus. Strava points outrank photo
EXIF at the same second; photos still cover the jeep days and the cities.
The activity table carries each day's name and its real start time.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Heart rate in the effort model

**Files:**
- Modify: `src/nepal/spine/effort.py:27-95`
- Test: `tests/test_effort.py` (extend)

**Interfaces:**
- Produces: `Effort.hr_bpm: float | None = None`; `effort.HR_REST_BPM = 60.0`, `effort.HR_MAX_BPM = 170.0`; `exertion(e, *, hr_rest=HR_REST_BPM, hr_max=HR_MAX_BPM) -> float`; `profile()` carries `GpsPoint.hr` through.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_effort.py`:

```python
def test_profile_carries_heart_rate_from_the_track():
    pts = [GpsPoint(T0 + timedelta(minutes=i), 28.5 + i * 90 / 111_000.0, 84.6, 4000.0,
                    "strava", None, 100.0 + i, "A") for i in range(4)]
    prof = profile(pts)
    assert [e.hr_bpm for e in prof] == [101.0, 102.0, 103.0]


def test_heart_rate_raises_exertion_on_the_same_climb():
    gps_only = Effort(T0, 0.5, 250.0, 4500.0)
    hard = Effort(T0, 0.5, 250.0, 4500.0, hr_bpm=160.0)
    easy = Effort(T0, 0.5, 250.0, 4500.0, hr_bpm=80.0)
    assert exertion(hard) > exertion(gps_only) > exertion(easy)


def test_heart_rate_is_clamped_to_the_configured_band():
    e = Effort(T0, 0.5, 0.0, 4500.0, hr_bpm=250.0)
    assert exertion(e, hr_rest=60.0, hr_max=170.0) == pytest.approx(0.5, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_effort.py -v`
Expected: the three new tests FAIL (`unexpected keyword argument 'hr_bpm'`).

- [ ] **Step 3: Implement**

In `src/nepal/spine/effort.py`, after `STEEP_GAIN_M_PER_H` add:

```python
# The heart-rate band that maps to 0..1 effort. Rest is a fit adult at
# breakfast; the top is what the watch actually recorded on day one of this
# trek (160). A calibration point, not a law: the scene targets in S06 read
# the same two config keys.
HR_REST_BPM = 60.0
HR_MAX_BPM = 170.0
```

extend the dataclass with `hr_bpm: float | None = None` (after `stopped_s`), make `profile` append `Effort(cur.ts, speed, gain, cur.ele, stopped_run, cur.hr)`, and change `exertion`:

```python
def exertion(e: Effort, *, hr_rest: float = HR_REST_BPM,
             hr_max: float = HR_MAX_BPM) -> float:
    """How hard this moment was, 0..1.

    Two things count from the track and they are not the same. Climbing
    steeply is effort even at a reasonable pace. Moving slowly *while*
    climbing is effort at its limit -- which at altitude is what the last
    hour to a pass looks like, and is the footage the film most wants.

    A long stop scores too. Nobody stands still for twenty minutes on a cold
    trail because things are going well.

    Where the watch recorded a heart rate, it is half the answer: the track
    can only infer effort, the pulse measures it.
    """
    gain = max(0.0, e.gain_m_per_h) / STEEP_GAIN_M_PER_H
    climb = min(1.0, gain)
    # slowness only counts while climbing; ambling downhill is not effort
    slow = max(0.0, 1.0 - e.speed_ms / MAX_WALK_MS) if gain > 0.15 else 0.0
    stop = min(1.0, e.stopped_s / 1200.0)          # twenty minutes
    from_track = float(min(1.0, 0.55 * climb + 0.30 * slow + 0.15 * stop))
    if e.hr_bpm is None or hr_max <= hr_rest:
        return from_track
    pulse = min(1.0, max(0.0, (float(e.hr_bpm) - hr_rest) / (hr_max - hr_rest)))
    return float(min(1.0, 0.5 * pulse + 0.5 * from_track))
```

- [ ] **Step 4: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_effort.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nepal/spine/effort.py tests/test_effort.py
git commit -m "Effort reads the pulse where the watch recorded one

The track can only infer effort from gain and slowness; the heart rate
measures it. Where a fix carries one it is half the answer, clamped to a
rest-to-maximum band that the scene targets will read from the same keys.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Every shot gets a position, an altitude, a place and a day

A new pure module answers "where and when was this moment", a new cheap S03 sub-step `place` applies it to every shot and always re-runs (the track and the boundaries move with S02), and `nepal diagnose` reports the result so a regression is visible.

**Files:**
- Create: `src/nepal/spine/place.py`
- Modify: `src/nepal/spine/gps.py` (add `interpolate_point`)
- Modify: `src/nepal/stages/s03_process.py` (`detect_shots` rows, `build_photo_shots` rows, new `place_shots`, `run` steps)
- Modify: `src/nepal/diagnose.py:126-175` (new section)
- Modify: `src/nepal/cli.py:73-75` (the `--redo` help text)
- Test: `tests/test_place.py` (new), `tests/test_gps.py` (extend)

**Interfaces:**
- Produces:
  - `gps.interpolate_point(points, ts, *, max_gap_s=14400.0) -> GpsPoint | None` — lat/lon linear; `ele` and `hr` interpolated only when both bracketing points carry an `activity_id` (a barometric track), else None; `source="interp"`.
  - `place.Placement(lat, lon, alt_m, alt_source, place_name, day_index)` (frozen dataclass).
  - `place.load_track(conn) -> list[GpsPoint]`.
  - `place.trek_span(bounds) -> tuple[datetime, datetime] | None` — from act 2's start to act 5's end (act 6's when present).
  - `place.day_of(ts, span) -> int | None`.
  - `place.describe(lat, lon, *, srtm, gazetteer, ele=None) -> tuple[float | None, str, str | None]`.
  - `place.place_at(track, ts, *, srtm, gazetteer, max_gap_s, span) -> Placement`.
  - S03 sub-step `place` (`s03_process.place_shots(cfg, conn) -> dict`), valid in `--redo`, always re-run.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gps.py`:

```python
from nepal.spine.gps import interpolate_point


def test_interpolate_point_carries_altitude_only_between_watch_fixes():
    a = GpsPoint(t(6), 28.0, 84.0, 4000.0, "strava", None, 100.0, "A")
    b = GpsPoint(t(8), 28.2, 84.2, 4400.0, "strava", None, 120.0, "A")
    mid = interpolate_point([a, b], t(7))
    assert mid.lat == pytest.approx(28.1) and mid.ele == pytest.approx(4200.0)
    assert mid.hr == pytest.approx(110.0) and mid.source == "interp"
    photo = GpsPoint(t(8), 28.2, 84.2, 4400.0, "phone_keller")
    assert interpolate_point([a, photo], t(7)).ele is None
    assert interpolate_point([a, b], t(9)) is None
```

```python
# tests/test_place.py
"""Where and when a moment was: position, altitude, place, trek day."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal.spine import place
from nepal.spine.acts import ActBoundary
from nepal.spine.dem import Srtm
from nepal.spine.geocode import Gazetteer, Place
from nepal.spine.gps import GpsPoint

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 2, 0, tzinfo=UTC)          # trek day 1, morning


def _track():
    return [GpsPoint(T0 + timedelta(hours=h), 28.5 + 0.01 * h, 84.6, 2000.0 + 100 * h,
                     "strava", None, 100.0, "A") for h in range(4)]


def _bounds():
    return [ActBoundary(1, T0 - timedelta(days=60), T0 - timedelta(hours=1)),
            ActBoundary(2, T0 - timedelta(hours=1), T0 + timedelta(days=3)),
            ActBoundary(3, T0 + timedelta(days=3), T0 + timedelta(days=6)),
            ActBoundary(4, T0 + timedelta(days=6), T0 + timedelta(days=7)),
            ActBoundary(5, T0 + timedelta(days=7), T0 + timedelta(days=12))]


GAZ = Gazetteer([Place("Namrung", 28.515, 84.6, "P", "PPL", 400)])


def test_trek_span_runs_from_act_2_to_the_last_act():
    span = place.trek_span(_bounds())
    assert span == (T0 - timedelta(hours=1), T0 + timedelta(days=12))


def test_day_of_is_null_outside_the_trek():
    span = place.trek_span(_bounds())
    assert place.day_of(T0 + timedelta(hours=2), span) == 1
    assert place.day_of(T0 + timedelta(days=1, hours=2), span) == 2
    assert place.day_of(T0 - timedelta(days=30), span) is None
    assert place.day_of(T0 + timedelta(days=40), span) is None


def test_place_at_interpolates_and_names_and_dates(tmp_path):
    got = place.place_at(_track(), T0 + timedelta(hours=1, minutes=30),
                         srtm=Srtm(tmp_path), gazetteer=GAZ, max_gap_s=14400.0,
                         span=place.trek_span(_bounds()))
    assert got.lat == pytest.approx(28.515, abs=1e-6)
    assert got.alt_m == pytest.approx(2150.0) and got.alt_source == "gpx"
    assert got.place_name == "Namrung"
    assert got.day_index == 1


def test_place_at_outside_the_track_has_no_position_but_still_a_day(tmp_path):
    got = place.place_at(_track(), T0 + timedelta(days=2),
                         srtm=Srtm(tmp_path), gazetteer=GAZ, max_gap_s=14400.0,
                         span=place.trek_span(_bounds()))
    assert got.lat is None and got.place_name is None and got.alt_m is None
    assert got.day_index == 3


def test_describe_a_photos_own_fix(tmp_path):
    alt, src, name = place.describe(28.515, 84.6, srtm=Srtm(tmp_path), gazetteer=GAZ)
    assert alt is None and src == "none" and name == "Namrung"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_place.py tests/test_gps.py -v`
Expected: FAIL with ImportError on `interpolate_point` and `place`.

- [ ] **Step 3: `interpolate_point`**

In `src/nepal/spine/gps.py`, after `interpolate_at` add:

```python
def interpolate_point(points: Sequence[GpsPoint], ts: datetime, *,
                      max_gap_s: float = 14400.0) -> GpsPoint | None:
    """Like ``interpolate_at`` but a whole point: altitude and heart rate
    come along **only between two watch fixes**. A photo fix's ``ele`` is a
    DEM lookup at the photo, and interpolating two of those is worse than
    looking the DEM up at the interpolated position, so between anything
    else they are left None for the caller to resolve."""
    if not points:
        return None
    lo, hi = _bracket(points, ts)
    if lo is None or hi is None:
        return None
    if lo is hi:
        return GpsPoint(ts, lo.lat, lo.lon, lo.ele if lo.activity_id else None,
                        "interp", None, lo.hr if lo.activity_id else None,
                        lo.activity_id)
    span = (hi.ts - lo.ts).total_seconds()
    if span > max_gap_s:
        return None
    f = 0.0 if span <= 0 else (ts - lo.ts).total_seconds() / span
    both_watch = bool(lo.activity_id and hi.activity_id)

    def mix(a: float | None, b: float | None) -> float | None:
        if not both_watch or a is None or b is None:
            return None
        return a + (b - a) * f

    return GpsPoint(ts, lo.lat + (hi.lat - lo.lat) * f, lo.lon + (hi.lon - lo.lon) * f,
                    mix(lo.ele, hi.ele), "interp", None, mix(lo.hr, hi.hr),
                    lo.activity_id if both_watch else None)
```

- [ ] **Step 4: The place module**

```python
# src/nepal/spine/place.py
"""Where and when a moment was.

Every shot needs a position, an altitude, a place name and a trek day: the
place cards, the altitude axis, the context score's "first at a place" and
"new altitude record" terms, and the six-act boundaries all read them. S03.0
filled them for photographs from the photo's own fix; video shots never had
them, so on the first draft the context score favoured stills and Act 2 was
a slideshow.

Pure except for the two lookups it is handed (the DEM and the gazetteer),
which are cheap, local and already tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from nepal.spine import dem as dem_mod
from nepal.spine import gps as gps_mod
from nepal.spine.gps import GpsPoint


@dataclass(frozen=True)
class Placement:
    lat: float | None
    lon: float | None
    alt_m: float | None
    alt_source: str
    place_name: str | None
    day_index: int | None


def load_track(conn) -> list[GpsPoint]:
    """The merged track from the database, with what the watch added."""
    out: list[GpsPoint] = []
    for r in conn.execute("SELECT ts_utc, lat, lon, alt_dem_m, hr_bpm, source, "
                          "activity_id FROM gps_points ORDER BY ts_utc"):
        try:
            ts = datetime.fromisoformat(str(r["ts_utc"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append(GpsPoint(ts, r["lat"], r["lon"], r["alt_dem_m"], r["source"],
                            None, r["hr_bpm"], r["activity_id"]))
    return out


def trek_span(bounds: Sequence) -> tuple[datetime, datetime] | None:
    """From the start of Act 2 to the end of the last act. Day 1 is the
    first day of Act 2; planning and after-trek material carry no day."""
    trek = [b for b in bounds if int(b.act) >= 2]
    if not trek:
        return None
    return (min(b.start_utc for b in trek), max(b.end_utc for b in trek))


def day_of(ts: datetime | None, span: tuple[datetime, datetime] | None) -> int | None:
    if ts is None or span is None or not (span[0] <= ts <= span[1]):
        return None
    return gps_mod.day_index(ts, span[0])


def describe(lat: float, lon: float, *, srtm: dem_mod.Srtm | None,
             gazetteer, ele: float | None = None
             ) -> tuple[float | None, str, str | None]:
    """Altitude (barometric first, then DEM), where it came from, and the
    nearest narratively useful place."""
    dem = srtm.elevation(lat, lon) if srtm is not None else None
    alt, src = dem_mod.resolve_altitude(dem, ele, None)
    pl = gazetteer.nearest(lat, lon) if gazetteer is not None else None
    return alt, src, (pl.name if pl else None)


def place_at(track: Sequence[GpsPoint], ts: datetime | None, *, srtm, gazetteer,
             max_gap_s: float, span: tuple[datetime, datetime] | None) -> Placement:
    """Everything the film wants to know about one moment."""
    day = day_of(ts, span)
    pt = gps_mod.interpolate_point(track, ts, max_gap_s=max_gap_s) \
        if (ts is not None and track) else None
    if pt is None:
        return Placement(None, None, None, "none", None, day)
    alt, src, name = describe(pt.lat, pt.lon, srtm=srtm, gazetteer=gazetteer, ele=pt.ele)
    return Placement(pt.lat, pt.lon, alt, src, name, day)
```

- [ ] **Step 5: The S03 sub-step**

In `src/nepal/stages/s03_process.py`:

Add `from nepal.spine import place as place_mod` to the imports (next to the existing `spine` imports), then in `detect_shots` replace the per-row loop written in Task 1 with:

```python
        for row in rows:
            ts = rec_start + timedelta(seconds=row["start_s"]) if rec_start else None
            row["start_utc"] = ts.isoformat() if ts else None
            row["act"] = acts_mod.act_for(ts, bounds) if (ts and bounds) else None
            # Position, altitude, place and day are the `place` sub-step's
            # job, run right after this one and again whenever S02 moves.
            # Every row carries every column: db.upsert refuses rows that
            # differ, and a first row without `lat` once cost every video
            # shot its position.
            row["lat"] = row["lon"] = row["alt_dem_m"] = row["place_name"] = None
            row["day_index"] = None
```

remove the now-unused `track`/`max_gap` locals and the `placed` count from `detect_shots` (and its `"n_positioned"` report key and the `%d positioned` in the log line). In `build_photo_shots`, in the `_measure` return dict add `"day_index": None,` after `"place_name": item["place_name"],`.

Add the sub-step after `build_photo_shots`:

```python
def place_shots(cfg: Config, conn) -> dict[str, Any]:
    """S03 `place` -- position, altitude, place name and trek day for every
    shot, from its moment against the track.

    Always re-run: it is a pure function of the track and the act
    boundaries, both of which move when S02 does, and it costs seconds. A
    photograph keeps its own fix -- a phone's GPS at the moment of the shot
    beats an interpolation -- and gains only what the fix lacks.
    """
    bounds = _bounds(conn)
    span = place_mod.trek_span(bounds) if bounds else None
    track = place_mod.load_track(conn)
    max_gap = float(cfg.get("spine.max_interp_gap_s"))
    srtm = dem_mod.Srtm(cfg.srtm_dir)
    gaz = geo_mod.Gazetteer(geo_mod.load_geonames(cfg.geonames_path))

    rows = [dict(r) for r in conn.execute(
        "SELECT shot_id, media_kind, start_utc, lat, lon FROM shots")]
    updates: list[tuple] = []
    positioned = dayed = named = 0
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        ts = _dt(r["start_utc"])
        if r["media_kind"] == "photo" and r["lat"] is not None and r["lon"] is not None:
            alt, _src, name = place_mod.describe(r["lat"], r["lon"], srtm=srtm, gazetteer=gaz)
            p = place_mod.Placement(r["lat"], r["lon"], alt, _src, name,
                                    place_mod.day_of(ts, span))
        else:
            p = place_mod.place_at(track, ts, srtm=srtm, gazetteer=gaz,
                                   max_gap_s=max_gap, span=span)
        updates.append((p.lat, p.lon, p.alt_m, p.place_name, p.day_index, r["shot_id"]))
        tally = by_kind.setdefault(r["media_kind"], [0, 0, 0])
        tally[0] += 1
        if p.lat is not None:
            positioned += 1
            tally[1] += 1
        if p.day_index is not None:
            dayed += 1
            tally[2] += 1
        if p.place_name:
            named += 1
    conn.executemany("UPDATE shots SET lat=?, lon=?, alt_dem_m=?, place_name=?, "
                     "day_index=? WHERE shot_id=?", updates)
    conn.commit()
    log.info("S03 place: %d shot(s): %d positioned, %d named, %d on a trek day; "
             "by kind %s", len(rows), positioned, named, dayed,
             {k: f"{v[1]}/{v[0]} positioned, {v[2]} dayed" for k, v in by_kind.items()})
    if rows and not positioned:
        log.warning("S03 place: no shot could be positioned -- is gps_points empty, "
                    "or every start_utc outside the track?")
    return {"n_shots": len(rows), "n_positioned": positioned, "n_named": named,
            "n_dayed": dayed, "n_track_points": len(track),
            "by_kind": {k: {"n": v[0], "positioned": v[1], "dayed": v[2]}
                        for k, v in by_kind.items()}}
```

In `run`, insert `("place", lambda: place_shots(cfg, conn)),` right after the `("photos", ...)` entry, and make it always run: `always = {"gate", "place"} | set(redo or ())`. Update the `--redo` help in `cli.py` to `proxies,shots,photos,place,metrics,audio,asr,faces,recluster,gate`.

- [ ] **Step 6: `nepal diagnose` shows the result**

In `src/nepal/diagnose.py::run`, after the `hr("clock decisions")` block add:

```python
    hr("shots placed, per kind -- the context score reads these")
    for r in conn.execute(
        "SELECT media_kind, COUNT(*) n, SUM(lat IS NOT NULL) pos, "
        "SUM(alt_dem_m IS NOT NULL) alt, SUM(place_name IS NOT NULL) named, "
        "SUM(day_index IS NOT NULL) dayed FROM shots WHERE status <> 'rejected' "
        "GROUP BY media_kind ORDER BY media_kind"
    ):
        print(f"  {r['media_kind']:<6} {r['n']:>5} surviving: {r['pos']:>5} positioned, "
              f"{r['alt']:>5} with altitude, {r['named']:>5} named, {r['dayed']:>5} on a trek day")
        if r["n"] and not r["pos"]:
            print("     ^ none positioned: run `nepal s03 --redo place` after `nepal s02`")
```

- [ ] **Step 7: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_place.py tests/test_gps.py tests/test_shots.py tests/test_s03_photos.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/nepal/spine/place.py src/nepal/spine/gps.py src/nepal/stages/s03_process.py src/nepal/diagnose.py src/nepal/cli.py tests/test_place.py tests/test_gps.py
git commit -m "Every shot gets a position, an altitude, a place and a trek day

Video shots never had any of them: S03.0 filled them for photographs from
the photo's own fix, S03.2 wrote lat/lon only when interpolation happened
to succeed, and nothing wrote altitude, place or day at all. The context
score therefore favoured stills, and Act 2 came out as a slideshow. The
new `place` sub-step is pure, costs seconds, and always re-runs, because
the track and the act boundaries move whenever S02 does.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `has_face` is derived at gate time

S03.6 wrote `has_face`, S03.2's re-detection wiped it, and the recluster restored only `face_cluster`. The flag joins `stability`, `has_speech` and `wind` as something derived from a stored measurement whenever the gate runs.

**Files:**
- Modify: `src/nepal/process/gate.py` (add `has_face`)
- Modify: `src/nepal/stages/s03_process.py::apply_gate` (SELECT and UPDATE)
- Test: `tests/test_gate.py` (extend)

**Interfaces:**
- Produces: `gate.has_face(face_score, face_cluster, *, min_det_score) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gate.py`:

```python
from nepal.process.gate import has_face


@pytest.mark.parametrize("score,cluster,want", [
    (0.83, "keller", 1),
    (0.83, None, 1),        # detected, not attributed to a known person
    (0.30, None, 0),        # below the detector floor: a buckle, lichen
    (0.0, None, 0),         # looked at, nothing there
    (None, None, 0),        # never looked at
    (None, "kulikov", 1),   # re-clustered from stored embeddings after a re-detect
])
def test_has_face_is_derived_from_what_was_measured(score, cluster, want):
    assert has_face(score, cluster, min_det_score=0.55) == want
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'has_face'`.

- [ ] **Step 3: Implement**

In `src/nepal/process/gate.py`, after the constants add:

```python
FACE_MIN_DET_SCORE = 0.55


def has_face(face_score: float | None, face_cluster: str | None, *,
             min_det_score: float = FACE_MIN_DET_SCORE) -> int:
    """Whether the shot shows a face, from what S03.6 stored.

    Derived at gate time rather than trusted from detection time, like
    stability and the audio flags: S03.2's re-detection replaces shot rows
    and the recluster step restores only the cluster label from the stored
    embeddings, so the flag written at detection time did not survive -- on
    the first draft `has_face` was 0 on every row while 347 carried a
    cluster, and the subject constraint and the context term were dead.
    """
    if face_cluster:
        return 1
    if face_score is None:
        return 0
    return int(float(face_score) >= float(min_det_score))
```

In `src/nepal/stages/s03_process.py::apply_gate`, add `s.face_score, s.face_cluster, ` to the SELECT column list, and after `refresh_audio_flags(cfg, conn)` add:

```python
    # Faces, the same way: a fact derived from the stored score and label.
    min_det = float(cfg.get("process.face_min_det_score", gate_mod.FACE_MIN_DET_SCORE))
    conn.executemany("UPDATE shots SET has_face=? WHERE shot_id=?", [
        (gate_mod.has_face(r["face_score"], r["face_cluster"], min_det_score=min_det),
         r["shot_id"]) for r in rows])
    n_faces = sum(1 for r in rows if gate_mod.has_face(
        r["face_score"], r["face_cluster"], min_det_score=min_det))
    log.info("S03.7 %d shot(s) show a face", n_faces)
```

and add `"n_with_face": n_faces` to the returned dict.

- [ ] **Step 4: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_gate.py tests/test_faces.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nepal/process/gate.py src/nepal/stages/s03_process.py tests/test_gate.py
git commit -m "has_face is derived at gate time from the stored face score

S03.6 wrote it, S03.2's re-detection replaced the rows, and the recluster
restored only the cluster label. Every row read 0 while 347 carried a
cluster, so the subject constraint and the context term were dead. It now
joins stability and the audio flags as something re-derived whenever the
gate runs.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: The draft renders at one frame rate

The concat of 15 fps proxies, 25 fps stills and 30/60 fps phone clips produced a 120 fps file. Every segment is resampled and the output rate is fixed.

**Files:**
- Modify: `src/nepal/process/render.py:27`, `:74-103`, `:111-178`
- Modify: `src/nepal/stages/s05_cut.py:222-224`
- Test: `tests/test_render.py` (extend)

**Interfaces:**
- Produces: `render.DRAFT_FPS = 30`; `segment_filters(row, index, *, width, height, overlay, fps=DRAFT_FPS)`; `build_command(..., fps=DRAFT_FPS)`; config `render.fps`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py`:

```python
def test_every_segment_is_resampled_and_the_output_rate_is_fixed():
    """15 fps proxies, 25 fps stills and 60 fps phone clips concatenated
    without a rate came out as a 120 fps file."""
    rows = [{"shot_id": "v1", "media_kind": "video", "t_in": 0.0, "t_out": 2.0,
             "src_in": 5.0, "src_out": 7.0},
            {"shot_id": "p1", "media_kind": "photo", "t_in": 2.0, "t_out": 5.0}]
    cmd = render.build_command(rows, sources={"v1": pathlib.Path("/m/v.mp4"),
                                              "p1": pathlib.Path("/m/a.jpg")},
                               out_path=pathlib.Path("/o.mp4"), fps=30)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert fc.count("fps=30") == 2
    assert cmd[cmd.index("-r") + 1] == "30"
    assert "fps=25" not in fc
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_render.py -v -k resampled`
Expected: FAIL with `unexpected keyword argument 'fps'`.

- [ ] **Step 3: Implement**

In `src/nepal/process/render.py`: `DRAFT_W, DRAFT_H, DRAFT_CRF, DRAFT_FPS = 960, 540, 23, 30`.

`segment_filters` gains `fps: int = DRAFT_FPS` and builds:

```python
    chain = []
    if is_still(row):
        dur = float(row["t_out"]) - float(row["t_in"])
        chain += [f"loop=loop=-1:size=1:start=0", f"fps={fps}", f"trim=duration={dur:.3f}"]
    # Every leg of the concat must share a rate: the proxies are 15 fps, the
    # phones 30 or 60, and concat of mixed rates produced a 120 fps draft.
    chain += ["setpts=PTS-STARTPTS", f"fps={fps}",
             f"scale={width}:{height}:force_original_aspect_ratio=decrease"
             f":force_divisible_by=2",
             f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
             "setsar=1"]
```

`build_command` gains `fps: int = DRAFT_FPS`, passes `fps=fps` into `segment_filters`, and emits `"-r", str(fps)` immediately before `"-c:v"`. In `s05_cut.render_draft` pass `fps=int(cfg.get("render.fps", render_mod.DRAFT_FPS))`.

- [ ] **Step 4: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_render.py -v`
Expected: all PASS (the two `ffmpeg_*` tests are slow-marked and deselected by default).

- [ ] **Step 5: Commit**

```bash
git add src/nepal/process/render.py src/nepal/stages/s05_cut.py tests/test_render.py
git commit -m "The draft renders at one frame rate

Fifteen-fps proxies, twenty-five-fps stills and sixty-fps phone clips went
into one concat with no rate, and ffmpeg chose 120. Every leg is now
resampled and the output rate is a config value.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Config, docs, and the cheap real-corpus re-run

Every key the previous tasks read gets its place in `config/pipeline.yaml`, the docs record what changed, and the parts of the pipeline that run in seconds are re-run on the real corpus to confirm the fixes with numbers.

**Files:**
- Modify: `config/pipeline.yaml` (inside the existing `probe:`, `spine:`, `render:` blocks; new `effort:` block)
- Modify: `docs/STATE.md`, `README.md`, `CLAUDE.md`
- Test: `tests/test_config_paths.py` (extend with one loader check)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_paths.py`:

```python
def test_the_shipped_config_declares_every_step1_key():
    from nepal.config import Config
    cfg = Config.load(pathlib.Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml")
    assert cfg.get("probe.exclude_dirs") == ["strava"]
    assert "*.zip" in cfg.get("probe.ignore_globs")
    assert cfg.get("probe.capture_time_overrides") == {}
    assert cfg.get("spine.strava_dir") == "strava"
    assert cfg.get("spine.strava_sample_s") == 5
    assert cfg.get("spine.named_peaks")[0]["name"] == "Manaslu"
    assert cfg.get("effort.hr_rest_bpm") == 60 and cfg.get("effort.hr_max_bpm") == 170
    assert cfg.get("render.fps") == 30
```

(add `import pathlib` at the top of that file if it is not already imported).

- [ ] **Step 2: Run it to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_config_paths.py -v -k step1`
Expected: FAIL with `KeyError: missing config key: probe.exclude_dirs`.

- [ ] **Step 3: Add the keys**

Inside the existing `probe:` block (after `max_chapter_gap_s: 1.0`):

```yaml
  # Directories under data_root that are not media. The Strava export is read
  # by S02.1 directly; walked by the manifest it would become 12 "assets".
  exclude_dirs: [strava]
  # Files that are not media wherever they sit: an installer, the thumbnail
  # Telegram writes beside every PDF, a spreadsheet lock file.
  ignore_globs: ["*.zip", "*_thumb.jpg", ".~lock*"]
  # A capture time the operator knows and the file does not, by file name.
  # Three phone clips on this corpus carry only an export date (2025-11-22),
  # and no clock offset can recover a time that was never recorded.
  #   video_1_902aaa.mp4: "2024-05-12T09:30:00+05:45"
  capture_time_overrides: {}
```

Inside the existing `spine:` block (after `whisper_language`):

```yaml
  # S02.1 Strava. One FIT file per trekking day from an Apple Watch: a fix a
  # second, barometric altitude, heart rate. Sampled to this spacing on the
  # way in -- one fix in five seconds is 56k points for the trek, which the
  # interpolation handles in milliseconds; one a second is five times that
  # for nothing the film can see.
  strava_dir: strava
  strava_sample_s: 5
  # Summits the film may label in the frame (section 13.3). Coordinates from
  # GeoNames; altitude in metres.
  named_peaks:
    - {name: Manaslu, lat: 28.5497, lon: 84.5597, alt_m: 8163}
```

A new top-level block (check first with `grep -n '^effort:' config/pipeline.yaml` that none exists):

```yaml
# Effort from the watch. The band that maps heart rate onto 0..1: rest is a
# fit adult at breakfast, the top is what the watch recorded on day one of
# this trek. Read by spine/effort.py and, later, by the S06 scene targets.
effort:
  hr_rest_bpm: 60
  hr_max_bpm: 170
```

Inside the existing `render:` block (after `draft:`):

```yaml
  # One rate for every leg of the concat. The proxies are 15 fps, the phones
  # 30 or 60, and a concat of mixed rates produced a 120 fps draft.
  fps: 30
```

- [ ] **Step 4: Run the whole fast suite**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: all pass (the count rises by roughly thirty over the 827 baseline), 0 failures.

- [ ] **Step 5: The cheap real-corpus re-run, in order**

Each of these runs in seconds to a couple of minutes on the local machine. Record every number printed; they go into `STATE.md` in the next step.

```bash
cd /data/projects/nepalvideo/.claude/worktrees/film-v2-design
N=/data/projects/nepalvideo/.venv/bin/nepal
$N doctor | head -12                          # must say "editable -- this checkout is what runs"
$N prune --dry-run                            # expect phone_kulikov_IMG, phone_keller_IMG, telegram_IMG? -- read the reasons
$N prune                                      # then for real
$N s02 --redo gps_track,geotag,acts           # Strava in; expect "n_from_strava" > 10,000 in the report
$N s03 --redo place                           # the gate runs too; expect video positioned > 800 of 930
$N diagnose | sed -n '/shots placed/,/^$/p'   # the new section
$N cut --redo score,timeline                  # NOT draft: the render is 8 min and waits for the remote box
```

Then measure what the cut looks like now, without rendering it:

```bash
sqlite3 /data/projects/nepal_work/db/nepal.sqlite "
SELECT t.act, COUNT(*) slots, SUM(s.media_kind='photo') stills, SUM(s.has_face) faces,
       SUM(s.place_name IS NOT NULL) placed, ROUND(SUM(t.t_out-t.t_in)) secs
FROM timeline t JOIN shots s ON s.shot_id=t.shot_id GROUP BY t.act;
SELECT COUNT(*) FROM shots WHERE recording_id LIKE '%_IMG';
SELECT source, COUNT(*) FROM gps_points GROUP BY source;
SELECT name, start_utc, hr_max, n_points FROM activities ORDER BY start_utc;"
```

Expected direction: no `_IMG` shots remain; `gps_points` has a `strava` row count above 10,000; Act 2's stills count falls well below its slot count (the photo *budget* is step 4, so stills may still be over 10%, but the ctx term no longer favours them); faces are non-zero in every trek act; Act 1 no longer contains Larke Pass footage.

If `nepal s02 --redo acts` moves the act boundaries (the Strava altitudes are barometric and finer than the DEM), say so in `STATE.md` with the old and new boundaries from `nepal decisions`.

- [ ] **Step 6: Docs**

`docs/STATE.md`: add a section **"Step 1 of Film v2 (date)"** under "The last full run" with: the prune result (recordings, shots, MB), the Strava ingest (activities, points, heart-rate range), the placement counts per kind from `diagnose`, the per-act table from the query above, and the sentence "The draft on disk is now older than the timeline; the next render runs on the remote box (step 2)." Move issues 3, 6, 9 and 11 of "Current issues" to a **Resolved** list with one line each on what fixed them. Add the new commands to "Do these next".

`README.md`: in "Running it" add `nepal prune [--dry-run]`, `nepal s01 --redo manifest,chapters,fov,clock`, `nepal s02 --redo gps_track,...`; in the S03 list add the `place` sub-step; in "S02 — Spine" add one paragraph on Strava as the spine; in the deviations table add a row "Photo EXIF as the geolocation spine → Strava first, photos fill the gaps".

`CLAUDE.md`, under "Traps": add

```
- **Every row of an upsert must carry every column.** `db.upsert` takes its
  columns from the first row. A row that lacked `lat` when the first row had
  it once dropped the column for the whole batch and left 930 video shots
  unpositioned; the upsert now refuses mismatched rows, so write `None`
  rather than omitting a key.
- **A flag written by one sub-step and wiped by another must be derived, not
  stored.** `has_face` was set by S03.6 and lost when S03.2 replaced the
  rows; it now joins `stability`, `has_speech` and `wind` as something the
  gate re-derives from the stored measurement every run.
```

- [ ] **Step 7: Run the slow suite once**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest -m slow -q 2>&1 | tail -3`
Expected: all pass in about two minutes (under the five-minute local limit). If a fixture test fails on the new schema, fix the fixture rather than the schema.

- [ ] **Step 8: Commit and push**

```bash
git add config/pipeline.yaml docs/STATE.md README.md CLAUDE.md tests/test_config_paths.py
git commit -m "Step 1 of Film v2 on the real corpus: pruned, Strava in, every shot placed

Config keys for the ignore list, the capture-time overrides, the Strava
export, the named peaks, the heart-rate band and the output frame rate.
STATE.md records what the cheap re-run measured: the ghosts gone, the
watch track in, positions on video shots, faces in the cut. The draft
render itself waits for the remote box.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin worktree-film-v2-design
```

---

## Self-review

**Spec coverage (step 1 scope, spec §11 item 1 and §8):** §8.1 prune → Task 2. §8.2 ignore list → Task 3. §8.3 `has_face` → Task 8. §8.4 video position, `diagnose` → Task 7. §8.5 `day_index` → Task 7. §8.8 frame rate → Task 9. §8.10 `--redo fov` → Task 4. §8.11 capture-time overrides → Task 4. §8.13 Strava ingestion → Task 5. §8.14 heading tags → Task 3. §8.15 named peaks → Task 10 (config). §13.1 data half (columns, activities, effort HR term) → Tasks 5, 6. Deferred to later plans by design: §8.6–7 (segment times, hallucination filter: step 3), §8.9 (constraints wiring: step 4), §8.12 (`usable_as_card`: step 3), the six-act boundary and day naming from activities (step 7), the manifest re-probe and the draft render on the real corpus (need the remote box: step 2).

**Placeholder scan:** none; every step carries its code and its command.

**Type consistency:** `GpsPoint` positional order `(ts, lat, lon, ele, source, accuracy_m, hr, activity_id)` is used identically in Tasks 5, 6, 7 and their tests. `Placement` fields `(lat, lon, alt_m, alt_source, place_name, day_index)` match between `place.py` and `place_shots`. `gate.has_face(face_score, face_cluster, *, min_det_score)` matches its call. `render.build_command(..., fps=)` matches `s05_cut`. `s01_probe.apply_offsets(conn, offsets, overrides)` matches both call sites in `run`.
