"""External binary wrappers.

Design rule for this codebase: every module that shells out keeps the shelling
in a thin function, and the decision logic that consumes its output stays pure
(lists/arrays in, values out). That is what makes the FOV solver, the clock
solver and the chapter grouper unit-testable without ffmpeg installed.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)


class ToolMissing(RuntimeError):
    pass


class ToolFailed(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str):
        self.cmd, self.returncode, self.stderr = list(cmd), returncode, stderr
        tail = stderr.strip().splitlines()[-6:]
        super().__init__(f"{cmd[0]} exited {returncode}\n" + "\n".join(tail))


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def require(*tools: str) -> None:
    missing = [t for t in tools if not have(t)]
    if missing:
        raise ToolMissing(
            f"required binaries not on PATH: {', '.join(missing)}. "
            f"Install with: apt-get install -y {' '.join(_apt_pkg(t) for t in missing)}"
        )


def _apt_pkg(tool: str) -> str:
    return {"ffmpeg": "ffmpeg", "ffprobe": "ffmpeg", "exiftool": "libimage-exiftool-perl"}.get(tool, tool)


def run(cmd: Sequence[str], *, check: bool = True, timeout: float | None = None,
        capture: bool = True) -> subprocess.CompletedProcess:
    log.debug("exec: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=capture, text=True, timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise ToolFailed(cmd, proc.returncode, proc.stderr or "")
    return proc


# -- ffprobe -----------------------------------------------------------

def ffprobe(path: str | Path) -> dict[str, Any]:
    """Full stream+format JSON for one file."""
    require("ffprobe")
    proc = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    return json.loads(proc.stdout or "{}")


def probe_summary(path: str | Path) -> dict[str, Any]:
    """The handful of fields the ``assets`` table wants."""
    data = ffprobe(path)
    fmt = data.get("format", {})
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {
        "width": _int(video.get("width")),
        "height": _int(video.get("height")),
        "fps": _fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "duration_s": _float(fmt.get("duration") or video.get("duration")),
        "has_audio": audio is not None,
        "probe_json": json.dumps(data, separators=(",", ":")),
    }


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fps(rate: str | None) -> float | None:
    """'30000/1001' -> 29.97. Returns None for '0/0'."""
    if not rate or "/" not in str(rate):
        return _float(rate)
    num, _, den = str(rate).partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    return round(n / d, 6) if d else None


# -- ffmpeg ------------------------------------------------------------

def ffmpeg(args: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    require("ffmpeg")
    return run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *args],
               timeout=timeout)


def extract_frame(src: str | Path, t_s: float, dest: str | Path,
                  vf: str | None = None) -> Path:
    """Single frame at t_s, optionally through a filter chain."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["-ss", f"{t_s:.3f}", "-i", str(src), "-frames:v", "1"]
    if vf:
        args += ["-vf", vf]
    args += ["-y", str(dest)]
    ffmpeg(args)
    return dest


def extract_audio(src: str | Path, dest: str | Path, *, sample_rate: int = 16000,
                  start_s: float | None = None, duration_s: float | None = None) -> Path:
    """16 kHz mono PCM WAV -- the format both the clock solver and ASR want."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    if start_s is not None:
        args += ["-ss", f"{start_s:.3f}"]
    args += ["-i", str(src)]
    if duration_s is not None:
        args += ["-t", f"{duration_s:.3f}"]
    args += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-y", str(dest)]
    ffmpeg(args)
    return dest


# -- exiftool ----------------------------------------------------------

def _capture_tag_args() -> list[str]:
    """Request exactly the capture-time tags the manifest knows how to read.

    A tag this scan does not ask for does not exist as far as the manifest is
    concerned, however carefully asset_datetime() ranks it. That has now bitten
    twice: OffsetTimeOriginal was absent and every Nepal photo landed 5h45m
    out, then CreationDate was absent and 276 clips kept the export date their
    QuickTime stamp had been rewritten to. Deriving the request from
    CAPTURE_TAGS is what stops it happening a third time -- the reader and the
    request cannot drift apart if only one of them is written by hand.
    """
    from nepal.probe.manifest import CAPTURE_TAGS, HEADING_TAGS
    # The heading and lens tags ride the same derivation: parse_heading()
    # reads them, so the request must carry them.
    return [f"-{tag}" for tag in (*CAPTURE_TAGS, *HEADING_TAGS)]


EXIF_TAGS = [
    "-FileName", "-Directory", "-FileSize", "-FileType", "-MIMEType", "-ImageSize",
    "-Duration", "-VideoFrameRate",
    *_capture_tag_args(),
    "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-Make", "-Model",
    # EXIF keeps a timestamp's UTC offset in a SEPARATE tag. Omitting these
    # from the request makes asset_datetime()'s timezone handling dead code and
    # silently shifts every Nepal photo by 5h45m onto the wrong day.
    "-OffsetTimeOriginal", "-OffsetTime", "-OffsetTimeDigitized",
]


def exiftool_recursive(root: str | Path, extra: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One exiftool pass over a whole tree (spec S01.1).

    -n forces numeric GPS output, which is the difference between parsing
    '27 deg 59\\' 17.00" N' and reading 27.988056.
    """
    require("exiftool")
    proc = run([
        "exiftool", "-r", "-j", "-ee", "-G1", "-n", "-api", "largefilesupport=1",
        *EXIF_TAGS, *extra, str(root),
    ], check=False)
    if not proc.stdout.strip():
        if proc.returncode != 0:
            raise ToolFailed(["exiftool"], proc.returncode, proc.stderr or "")
        return []
    return json.loads(proc.stdout)


def exiftool_one(path: str | Path, extra: Sequence[str] = ()) -> dict[str, Any]:
    require("exiftool")
    proc = run(["exiftool", "-j", "-ee", "-G1", "-n", *extra, str(path)], check=False)
    rows = json.loads(proc.stdout) if proc.stdout.strip() else []
    return rows[0] if rows else {}
