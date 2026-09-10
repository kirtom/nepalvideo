"""S01.2 -- group chapter-split camera files into logical recordings.

Insta360 splits long recordings at ~4 GB. If each chunk is treated as its own
recording, every split becomes a false shot boundary in S03 and the telemetry
shows a gap. Grouping has to happen before scene detection.

Naming conventions vary by device and firmware, so this module recognises the
common Insta360 layouts and falls back to a generic rule rather than assuming
one. Whatever it decides is written to the S01 report for the operator to
eyeball, and the section S01.2 acceptance check (no inter-chapter gap greater
than 1 s) runs over the result.

Recognised forms::

    VID_20231015_143022_00_001.insv   -> key 20231015_143022, chapter 1
    LRV_20231015_143022_01_001.lrv    -> key 20231015_143022, chapter 1
    VID_20231015_143022_00_002.insv   -> key 20231015_143022, chapter 2

The ``.lrv`` proxy is deliberately grouped with the ``.insv`` it proxies: they
are the same content, and S03 prefers the ``.lrv`` as its decode source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# VID_<date>_<time>_<stream>_<chapter>.<ext>  (also LRV_, PRO_, and _NN suffixes)
INSTA360_RE = re.compile(
    r"^(?P<prefix>[A-Z]{3})_(?P<date>\d{8})_(?P<time>\d{6})"
    r"(?:_(?P<stream>\d{2}))?_(?P<chapter>\d{3})(?P<tail>.*)$",
    re.IGNORECASE,
)

# GoPro-style chaptering, in case any flat mp4 came from a second camera:
# GH010123.MP4 -> chapter 01, recording 0123
GOPRO_RE = re.compile(r"^(?P<prefix>G[HXP])(?P<chapter>\d{2})(?P<rec>\d{4})$", re.IGNORECASE)

# Generic: anything ending in a separator plus digits, e.g. clip_003
GENERIC_RE = re.compile(r"^(?P<base>.+?)[._-](?P<chapter>\d{1,4})$")


@dataclass
class ChapterKey:
    recording_key: str
    chapter_index: int
    method: str


def parse_chapter(filename: str) -> ChapterKey | None:
    """Extract (recording key, chapter index) from a filename, or None."""
    stem = Path(filename).stem

    m = INSTA360_RE.match(stem)
    if m:
        return ChapterKey(f"{m.group('date')}_{m.group('time')}",
                          int(m.group("chapter")), "insta360")

    m = GOPRO_RE.match(stem)
    if m:
        return ChapterKey(f"gopro_{m.group('rec')}", int(m.group("chapter")), "gopro")

    m = GENERIC_RE.match(stem)
    if m:
        return ChapterKey(m.group("base"), int(m.group("chapter")), "generic")

    return None


@dataclass
class Recording:
    recording_id: str
    source: str
    is_360: bool
    asset_ids: list[str] = field(default_factory=list)
    start_utc: str | None = None
    duration_s: float = 0.0
    grouping_method: str = "singleton"

    @property
    def asset_count(self) -> int:
        return len(self.asset_ids)


def _sort_key(a: dict[str, Any]) -> tuple:
    ck = parse_chapter(a.get("filename") or Path(a["s3_key"]).name)
    return (ck.chapter_index if ck else 0, a.get("created_at") or "", a["asset_id"])


def group_recordings(assets: Sequence[dict[str, Any]]) -> list[Recording]:
    """Group video assets into recordings.

    ``assets`` are dict rows with at least: asset_id, s3_key, source, kind,
    duration_s, created_at, and optionally filename.

    Photos and documents get no recording. Every video asset ends up in exactly
    one recording, singleton if nothing groups with it.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    methods: dict[str, str] = {}

    for a in assets:
        if a.get("kind") not in ("video360", "video_flat"):
            continue
        name = a.get("filename") or Path(a["s3_key"]).name
        ck = parse_chapter(name)
        if ck and ck.method in ("insta360", "gopro"):
            key = f"{a['source']}:{ck.recording_key}"
            methods[key] = ck.method
        elif ck and ck.method == "generic":
            key = f"{a['source']}:{ck.recording_key}"
            methods[key] = "generic"
        else:
            key = f"{a['source']}:{Path(name).stem}"
            methods[key] = "singleton"
        buckets.setdefault(key, []).append(a)

    recordings: list[Recording] = []
    for key, group in sorted(buckets.items()):
        group.sort(key=_sort_key)
        # A recording is 360 if any member is; .lrv proxies and .insv originals
        # of the same take live together and are all 360.
        is_360 = any(a.get("kind") == "video360" for a in group)
        # Duration: the .lrv duplicates the .insv, so summing every asset would
        # double-count. Sum one representative per chapter index instead.
        by_chapter: dict[int, dict[str, Any]] = {}
        for a in group:
            name = a.get("filename") or Path(a["s3_key"]).name
            ck = parse_chapter(name)
            idx = ck.chapter_index if ck else 0
            prev = by_chapter.get(idx)
            # prefer the original over the proxy for duration truth
            if prev is None or _is_proxy(prev) and not _is_proxy(a):
                by_chapter[idx] = a
        duration = sum(float(a.get("duration_s") or 0.0) for a in by_chapter.values())
        starts = [a.get("created_at_utc") or a.get("created_at") for a in group]
        starts = sorted(s for s in starts if s)
        rec = Recording(
            recording_id=key.replace(":", "_"),
            source=group[0]["source"],
            is_360=is_360,
            asset_ids=[a["asset_id"] for a in group],
            start_utc=starts[0] if starts else None,
            duration_s=round(duration, 3),
            grouping_method=methods.get(key, "singleton"),
        )
        recordings.append(rec)
    return recordings


def _is_proxy(asset: dict[str, Any]) -> bool:
    return (asset.get("container") or "").lower() == "lrv"


def check_continuity(recording: Recording, assets_by_id: dict[str, dict[str, Any]],
                     max_gap_s: float = 1.0) -> list[str]:
    """Section S01.2 acceptance: no gap between chapter N end and chapter N+1 start
    greater than ``max_gap_s``.

    Returns a list of human-readable violations; empty means the grouping is
    consistent with the timestamps. Chapters whose timestamps are missing or
    identical (some firmwares stamp every chunk with the take's start time) are
    reported separately rather than counted as violations, because that is a
    metadata quirk, not a grouping error.
    """
    from datetime import datetime

    chapters = []
    for aid in recording.asset_ids:
        a = assets_by_id.get(aid)
        if not a or _is_proxy(a):
            continue
        name = a.get("filename") or Path(a["s3_key"]).name
        ck = parse_chapter(name)
        ts = a.get("created_at_utc") or a.get("created_at")
        if ts is None:
            continue
        chapters.append((ck.chapter_index if ck else 0, ts, float(a.get("duration_s") or 0.0), name))

    if len(chapters) < 2:
        return []

    chapters.sort()
    problems: list[str] = []
    identical = len({c[1] for c in chapters}) == 1
    if identical:
        return [f"{recording.recording_id}: all {len(chapters)} chapters share one "
                f"timestamp ({chapters[0][1]}) -- continuity unverifiable from metadata"]

    for (i0, t0, d0, n0), (i1, t1, _d1, n1) in zip(chapters, chapters[1:]):
        try:
            a = datetime.fromisoformat(str(t0).replace("Z", "+00:00"))
            b = datetime.fromisoformat(str(t1).replace("Z", "+00:00"))
        except ValueError:
            continue
        gap = (b - a).total_seconds() - d0
        if abs(gap) > max_gap_s:
            problems.append(
                f"{recording.recording_id}: gap {gap:+.2f}s between chapter "
                f"{i0} ({n0}) and {i1} ({n1}) exceeds {max_gap_s}s"
            )
    return problems
