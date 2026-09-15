"""Is a recorded result the output of the code that is installed now?

``stage_units`` exists so a Spot interruption re-runs only what is missing. It
answers "did this finish?" -- not "did this finish under the code running
today", which is a different question and the one that matters after a fix.

A checkout writes every file it updates, so the newest .py mtime in the
installed package is the moment the current code arrived. A unit that completed
before then produced its output under the previous version.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def code_mtime() -> datetime | None:
    """When the installed package was last written, or None if unreadable."""
    import nepal
    root = Path(nepal.__file__).resolve().parent
    try:
        newest = max((f.stat().st_mtime for f in root.rglob("*.py")), default=None)
    except OSError:
        return None
    return datetime.fromtimestamp(newest, timezone.utc) if newest else None


def stale_units(conn, stage: str | None = None) -> list[tuple[str, str]]:
    """Completed units that finished before the installed code was written.

    Returns ``(stage, unit_id)`` pairs, sorted. Failed units are excluded: they
    are a different problem with a different fix, and folding them in here
    would bury it.
    """
    code = code_mtime()
    if code is None:
        return []
    sql = "SELECT stage, unit_id, updated_at FROM stage_units WHERE status='done'"
    args: tuple = ()
    if stage is not None:
        sql += " AND stage=?"
        args = (stage,)
    out = []
    for r in conn.execute(sql, args):
        raw = r["updated_at"]
        if not raw:
            continue
        try:
            ran = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        if ran.tzinfo is None:
            ran = ran.replace(tzinfo=timezone.utc)
        if ran < code:
            out.append((str(r["stage"]), str(r["unit_id"])))
    return sorted(out)


# Units that only a full --force re-run redoes; the cheap manifest-only re-run
# skips exactly these, so naming it for them sends the operator in a circle.
EXPENSIVE = {"S01": {"fov", "clock"}}


def rerun_command(stale: list[tuple[str, str]]) -> str:
    """The command that actually redoes these units.

    The cheap path is `nepal s01 --force --skip-fov --skip-clock`, which exists
    because a manifest fix should not cost an hour of GCC-PHAT. But if the FOV
    or clock solve is itself stale, that command skips the very work being asked
    for -- so it must not be the advice given.
    """
    parts = []
    s01 = {u for st, u in stale if st == "S01"}
    if s01:
        parts.append("nepal s01 --force" if s01 & EXPENSIVE["S01"]
                     else "nepal s01 --force --skip-fov --skip-clock")
    if any(st == "S02" for st, _ in stale):
        parts.append("nepal s02 --force")
    return " && ".join(parts)


def warn_if_stale(log, conn, stage: str, *, force: bool, rerun_hint: str) -> list[str]:
    """Say plainly that a re-run without --force will skip the stale work.

    Without this, `nepal s01` after a fix prints a full report and changes
    nothing: every unit is already marked done, so every unit is skipped, and
    the output is indistinguishable from a run that did the work.
    """
    if force:
        return []
    stale = [u for s, u in stale_units(conn, stage)]
    if stale:
        log.warning(
            "%s %s already completed under an OLDER version of the code and will "
            "be SKIPPED -- this run will not change them. To redo them: %s",
            stage, _summarise_units(stale), rerun_hint)
    return stale


def _summarise_units(units: list[str], *, max_listed: int = 12) -> str:
    """Name the top-level units and count the per-item ones.

    S03 records one unit per proxy, so a stale list is 480 entries and the
    warning became a single 20 KB line nobody reads. The reader wants to
    know WHICH steps are stale, not which of the 480 recordings.
    """
    from collections import Counter
    plain = [u for u in units if ":" not in u]
    grouped = Counter(u.split(":", 1)[0] for u in units if ":" in u)
    parts = sorted(plain)[:max_listed]
    if len(plain) > max_listed:
        parts.append(f"and {len(plain) - max_listed} more")
    parts += [f"{k}:* ({n} units)" for k, n in sorted(grouped.items())]
    return ", ".join(parts)
