"""What the project has paid for, and the line it must not cross.

One JSON file, appended by whatever spends: `remote down` with the VM hours,
the API wrappers with their `usage`. The ceiling is a refusal, not a
warning -- the GCP budget already warns. Kept as a file rather than a
`decisions` row so it survives a database that is rebuilt or pulled from
the bucket, and so a person can read it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[Entry] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path} is not valid JSON: {exc}") from exc
            self.entries = [Entry(**e) for e in raw.get("entries", [])]

    def total(self) -> float:
        return float(sum(e.usd for e in self.entries))

    def record(self, what: str, usd: float, detail: str = "") -> Entry:
        e = Entry(datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  what, round(float(usd), 4), detail)
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"entries": [asdict(x) for x in self.entries],
                                   "total_usd": round(self.total(), 4)}, indent=2))
        tmp.replace(self.path)
        return e

    def guard(self, estimate_usd: float, *, ceiling_usd: float) -> None:
        if self.total() + float(estimate_usd) > float(ceiling_usd):
            raise SpendCeiling(self.total(), float(estimate_usd), float(ceiling_usd))


def ledger(cfg) -> Ledger:
    return Ledger(cfg.work_root / "reports" / "spend.json")
