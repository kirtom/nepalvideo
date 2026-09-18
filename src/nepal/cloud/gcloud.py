"""The one place that runs `gcloud`."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)


class GcloudFailed(RuntimeError):
    pass


class Gcloud:
    def __init__(self, binary: str | Path):
        self.binary = str(Path(binary).expanduser())

    def run(self, args: Sequence[str], *, check: bool = True,
            timeout: float | None = None) -> subprocess.CompletedProcess:
        cmd = [self.binary, *args]
        log.debug("gcloud %s", " ".join(args))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise GcloudFailed(f"gcloud {' '.join(args[:4])} failed ({proc.returncode}): "
                               f"{(proc.stderr or proc.stdout)[-800:]}")
        return proc

    def json(self, args: Sequence[str]) -> Any:
        proc = self.run(args, check=False)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout or "null")

    def stream(self, args: Sequence[str]) -> int:
        """Run with the terminal attached: the operator watches the log."""
        return subprocess.call([self.binary, *args])
