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
from typing import Any, Iterable, Iterator

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
