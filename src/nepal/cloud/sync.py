"""What travels between the machine, the bucket and the box.

The bucket is the hub. Local pushes the working set once and new material
as it lands; the box pulls before a run and pushes what a run may have
changed; local pulls those to look at them. Nothing before conform reads
the camera originals, so they are not in the plan.
"""
from __future__ import annotations

from pathlib import Path

# The WAL and shm beside a sqlite file are never a copy's business: a WAL
# carries frames for the database it was open against, and sqlite validates
# them by their own checksums, not against the file they land next to. One
# pushed from the box at boot (left by a killed job) was pulled next to a
# newer database on 2026-10-07 and applied: "database disk image is
# malformed", and a draft cut from the result. Only the checkpointed file
# travels.
EXCLUDE = r".*\.~lock.*|.*\.DS_Store|.*Thumbs\.db|.*\.sqlite-(wal|shm)$"

# (kind, local subdir, bucket path); kind says which root the subdir is under
PUSH = (
    ("data", "media_from_phones", "raw/media_from_phones"),
    ("data", "chat_export", "raw/chat_export"),
    ("data", "music", "raw/music"),
    ("data", "strava", "raw/strava"),
    ("data", "title", "raw/title"),
    # No work/ entry: the local work/ is a pull mirror and the box owns the
    # bucket's. A `nepal remote push` for the title clip (2026-10-07 07:54)
    # also pushed the September proxies over the rebuilt ones, and the next
    # draft had its portrait clips squeezed again.
    ("ref", "data", "ref/data"),
)

# work/ subdirs a run may change and local wants back
PULL = ("db", "reports", "semantic", "faces", "transcripts", "gates", "overlays", "beats",
        "status")


def rsync_args(src: str, dst: str, *, exclude: str = EXCLUDE) -> list[str]:
    return ["storage", "rsync", "--recursive", f"--exclude={exclude}", src, dst]


def _bucket(cfg) -> str:
    return str(cfg.get("cloud.gcp.bucket")).rstrip("/")


def push_plan(cfg) -> list[tuple[Path, str]]:
    roots = {"data": cfg.data_root, "work": cfg.work_root, "ref": cfg.base_dir}
    return [(roots[kind] / sub, f"{_bucket(cfg)}/{dst}") for kind, sub, dst in PUSH]


def pull_plan(cfg) -> list[tuple[str, Path]]:
    return [(f"{_bucket(cfg)}/work/{sub}", cfg.work_root / sub) for sub in PULL]
