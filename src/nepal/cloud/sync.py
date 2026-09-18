"""What travels between the machine, the bucket and the box.

The bucket is the hub. Local pushes the working set once and new material
as it lands; the box pulls before a run and pushes what a run may have
changed; local pulls those to look at them. Nothing before conform reads
the camera originals, so they are not in the plan.
"""
from __future__ import annotations

from pathlib import Path

EXCLUDE = r".*\.~lock.*|.*\.DS_Store|.*Thumbs\.db"

# (kind, local subdir, bucket path); kind says which root the subdir is under
PUSH = (
    ("data", "media_from_phones", "raw/media_from_phones"),
    ("data", "chat_export", "raw/chat_export"),
    ("data", "music", "raw/music"),
    ("data", "strava", "raw/strava"),
    ("work", "db", "work/db"),
    ("work", "reports", "work/reports"),
    ("work", "proxies", "work/proxies"),
    ("work", "audio", "work/audio"),
    ("work", "transcripts", "work/transcripts"),
    ("work", "faces", "work/faces"),
    ("work", "music", "work/music"),
    ("work", "gpx", "work/gpx"),
    ("work", "vocab", "work/vocab"),
    ("work", "semantic", "work/semantic"),
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
