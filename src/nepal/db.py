"""SQLite state.

The schema is the spec's section 5 DDL verbatim, plus two additions the spec
requires in prose but does not tabulate:

  * ``stage_units``  -- section 6 ("Every stage records completion per unit of
    work in the DB so a Spot interruption re-runs only what is missing").
  * indices on the columns every downstream join actually uses.

Nothing else deviates. Column names match the spec exactly so the DDL can be
diffed against the document.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
-- Physical files as delivered
CREATE TABLE IF NOT EXISTS assets (
  asset_id      TEXT PRIMARY KEY,       -- sha256 of file
  s3_key        TEXT NOT NULL,
  source        TEXT NOT NULL,          -- camera | phone_keller | phone_kulikov | telegram
  kind          TEXT NOT NULL,          -- video360 | video_flat | photo | audio | document
  container     TEXT,                   -- insv | lrv | mp4 | jpg | heic ...
  bytes         INTEGER,
  width         INTEGER,
  height        INTEGER,
  fps           REAL,
  duration_s    REAL,
  created_at    TEXT,                   -- raw device timestamp, NOT corrected
  created_at_utc TEXT,                  -- after clock offset applied
  recording_id  TEXT,                   -- groups chapter-split files
  chapter_index INTEGER,
  has_gps       INTEGER DEFAULT 0,
  lat           REAL,
  lon           REAL,
  alt_dem_m     REAL,
  place_name    TEXT,
  quality_curve TEXT DEFAULT 'camera',  -- camera | phone | telegram
  probe_json    TEXT
);

-- Logical recordings (chapters merged)
CREATE TABLE IF NOT EXISTS recordings (
  recording_id  TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  is_360        INTEGER NOT NULL,
  start_utc     TEXT,
  duration_s    REAL,
  asset_count   INTEGER
);

-- Detected shots -- the unit of selection
CREATE TABLE IF NOT EXISTS shots (
  shot_id       TEXT PRIMARY KEY,
  recording_id  TEXT NOT NULL REFERENCES recordings(recording_id),
  start_s       REAL NOT NULL,
  end_s         REAL NOT NULL,
  start_utc     TEXT,
  day_index     INTEGER,
  act           INTEGER,
  lat REAL, lon REAL, alt_dem_m REAL, place_name TEXT,
  sharpness     REAL,
  exposure_pen  REAL,
  stability     REAL,
  motion_mag    REAL,
  audio_lufs    REAL,
  has_speech    INTEGER DEFAULT 0,
  has_face      INTEGER DEFAULT 0,
  face_cluster  TEXT,
  chosen_yaw    REAL,
  view_kind     TEXT,
  caption       TEXT,
  vlm_interest  REAL,
  transcript    TEXT,
  score_tech    REAL,
  score_sem     REAL,
  score_ctx     REAL,
  score_total   REAL,
  vote          INTEGER,
  tag_levity    INTEGER DEFAULT 0,
  status        TEXT DEFAULT 'candidate'
);

CREATE TABLE IF NOT EXISTS gps_points (
  ts_utc  TEXT PRIMARY KEY,
  lat REAL, lon REAL, alt_dem_m REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  msg_id      TEXT PRIMARY KEY,
  ts_utc      TEXT NOT NULL,
  author      TEXT,
  text        TEXT,
  media_asset TEXT REFERENCES assets(asset_id),
  phase       TEXT,
  is_notable  INTEGER DEFAULT 0,
  usable_as_card INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS music_tracks (
  track_id    TEXT PRIMARY KEY,
  s3_key      TEXT,
  title       TEXT,
  duration_s  REAL,
  tempo_bpm   REAL,
  key_est     TEXT,
  energy_mean REAL,
  energy_p95  REAL,
  energy_p10  REAL,
  centroid    REAL,
  onset_rate  REAL,
  assigned_act INTEGER
);

CREATE TABLE IF NOT EXISTS music_sections (
  section_id  TEXT PRIMARY KEY,
  track_id    TEXT REFERENCES music_tracks(track_id),
  start_s     REAL, end_s REAL,
  energy      REAL,
  is_swell    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS beats (
  track_id TEXT, t_s REAL, is_downbeat INTEGER
);

CREATE TABLE IF NOT EXISTS timeline (
  slot_index  INTEGER PRIMARY KEY,
  act         INTEGER,
  shot_id     TEXT REFERENCES shots(shot_id),
  msg_id      TEXT REFERENCES messages(msg_id),
  t_in        REAL,
  t_out       REAL,
  src_in      REAL,
  src_out     REAL,
  yaw         REAL,
  transition  TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
  key TEXT PRIMARY KEY, value TEXT, confidence REAL, method TEXT
);

-- ADDITION (spec section 6): per-unit completion ledger for resumability.
CREATE TABLE IF NOT EXISTS stage_units (
  stage       TEXT NOT NULL,
  unit_id     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'done',  -- done | failed
  detail      TEXT,
  updated_at  TEXT,
  PRIMARY KEY (stage, unit_id)
);

CREATE INDEX IF NOT EXISTS idx_assets_recording ON assets(recording_id);
CREATE INDEX IF NOT EXISTS idx_assets_source    ON assets(source);
CREATE INDEX IF NOT EXISTS idx_assets_utc       ON assets(created_at_utc);
CREATE INDEX IF NOT EXISTS idx_shots_recording  ON shots(recording_id);
CREATE INDEX IF NOT EXISTS idx_shots_status     ON shots(status);
CREATE INDEX IF NOT EXISTS idx_shots_act        ON shots(act);
CREATE INDEX IF NOT EXISTS idx_shots_utc        ON shots(start_utc);
CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages(ts_utc);
CREATE INDEX IF NOT EXISTS idx_beats_track      ON beats(track_id);
CREATE INDEX IF NOT EXISTS idx_sections_track   ON music_sections(track_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# Columns added after the first release. SQLite cannot add a column
# conditionally in DDL, so they are applied as idempotent migrations -- a
# database created by an earlier run must not have to be rebuilt. A database
# that already carries a column no longer in SCHEMA keeps it, harmlessly.
MIGRATIONS: tuple[tuple[str, str, str], ...] = ()


def init(db_path: str | Path) -> sqlite3.Connection:
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    for table, column, coltype in MIGRATIONS:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    conn.commit()
    return conn


# -- decisions ---------------------------------------------------------

def set_decision(conn: sqlite3.Connection, key: str, value: Any,
                 confidence: float | None = None, method: str = "") -> None:
    conn.execute(
        "INSERT INTO decisions(key, value, confidence, method) VALUES (?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
        "confidence=excluded.confidence, method=excluded.method",
        (key, str(value), confidence, method),
    )
    conn.commit()


def get_decision(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM decisions WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def get_decision_float(conn: sqlite3.Connection, key: str, default: float | None = None) -> float | None:
    v = get_decision(conn, key)
    if v is None or v == "None":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def all_decisions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT key, value, confidence, method FROM decisions ORDER BY key")]


# -- resumability ------------------------------------------------------

def mark_unit(conn: sqlite3.Connection, stage: str, unit_id: str,
              status: str = "done", detail: str = "") -> None:
    conn.execute(
        "INSERT INTO stage_units(stage, unit_id, status, detail, updated_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(stage, unit_id) DO UPDATE SET "
        "status=excluded.status, detail=excluded.detail, updated_at=excluded.updated_at",
        (stage, unit_id, status, detail, utcnow()),
    )
    conn.commit()


def done_units(conn: sqlite3.Connection, stage: str) -> set[str]:
    return {r["unit_id"] for r in conn.execute(
        "SELECT unit_id FROM stage_units WHERE stage=? AND status='done'", (stage,))}


def pending(conn: sqlite3.Connection, stage: str, unit_ids: Iterable[str]) -> list[str]:
    done = done_units(conn, stage)
    return [u for u in unit_ids if u not in done]


# -- generic upsert ----------------------------------------------------

def upsert(conn: sqlite3.Connection, table: str, key_cols: Sequence[str],
           rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in key_cols)
    sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT({','.join(key_cols)}) DO UPDATE SET {updates}"
           if updates else
           f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})")
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return len(rows)
