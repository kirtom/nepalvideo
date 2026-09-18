"""What the project has paid for, and the line it must not cross.

One file per entry under work/reports/spend/, written by whatever spends:
`remote down` with the VM hours, the API wrappers with their `usage`. The
ceiling is a refusal, not a warning -- the GCP budget already warns.

Files rather than one JSON, because two hosts write this ledger and the
bucket carries it between them by rsync in both directions. One file that
both hosts rewrite is a file each host clobbers with its own older copy:
the box's first ledger read 0.00 while this machine's said 0.15, and the
next push would have made that permanent. A file that is written once and
never changed survives any number of rsyncs in any direction, so the
total is always the union of what every host has recorded. The old
`spend.json` beside the directory is still read, never written.
"""
from __future__ import annotations

import json
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

LEGACY_NAME = "spend.json"


@dataclass(frozen=True)
class Entry:
    when: str
    what: str
    usd: float
    detail: str = ""


class SpendCeiling(RuntimeError):
    def __init__(self, total: float, estimate: float, ceiling: float):
        self.total, self.estimate, self.ceiling = total, estimate, ceiling
        super().__init__(
            f"spend ceiling: {total:.2f} USD spent, this step is estimated at "
            f"{estimate:.2f}, and the ceiling is {ceiling:.2f}. Raise "
            f"cloud.spend_ceiling_usd deliberately or skip the step.")


def _slug(what: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", what.lower()).strip("-") or "x"


class Ledger:
    def __init__(self, path: Path):
        """``path`` is the entries directory; a legacy single file beside it
        (``spend.json``) is read too."""
        self.path = Path(path)
        self.entries: list[Entry] = []
        legacy = self.path.parent / LEGACY_NAME
        files = ([legacy] if legacy.exists() else []) + \
            (sorted(self.path.glob("*.json")) if self.path.is_dir() else [])
        for f in files:
            try:
                raw = json.loads(f.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"{f} is not valid JSON: {exc}") from exc
            items = raw.get("entries", []) if isinstance(raw, dict) and "entries" in raw \
                else [raw]
            self.entries += [Entry(**e) for e in items]
        self.entries.sort(key=lambda e: e.when)

    def total(self) -> float:
        return float(sum(e.usd for e in self.entries))

    def record(self, what: str, usd: float, detail: str = "") -> Entry:
        # Microseconds, so two entries in one second keep the order they
        # were recorded in when the directory is read back.
        e = Entry(datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                  what, round(float(usd), 4), detail)
        self.entries.append(e)
        self.path.mkdir(parents=True, exist_ok=True)
        name = f"{e.when.replace(':', '').replace('+0000', 'Z')}_{_slug(what)}_{secrets.token_hex(3)}.json"
        tmp = self.path / (name + ".tmp")
        tmp.write_text(json.dumps(asdict(e), indent=2))
        tmp.replace(self.path / name)
        return e

    def guard(self, estimate_usd: float, *, ceiling_usd: float) -> None:
        if self.total() + float(estimate_usd) > float(ceiling_usd):
            raise SpendCeiling(self.total(), float(estimate_usd), float(ceiling_usd))


def ledger(cfg) -> Ledger:
    return Ledger(cfg.work_root / "reports" / "spend")
