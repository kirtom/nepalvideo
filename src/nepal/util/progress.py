"""Progress reporting for the stages that run for minutes without speaking.

S01.1 walks the tree, hands 1364 files to exiftool, and hashes every one of
them -- twenty minutes during which it printed two lines. There is no way to
tell a working pipeline from a hung one, which makes every long run an act of
faith.

Two shapes cover everything here:

  * ``Progress`` for a loop with a known count -- hashing files, decoding
    photographs, sweeping FOV candidates.
  * ``heartbeat`` for one long opaque call, where there are no steps to count
    and the only honest thing to report is that it is still running.

Both write to stderr, leaving stdout clean for the reports that get piped and
grepped. Neither assumes a terminal: piped to a file or a CI log, they emit
periodic lines instead of redrawing one, because a log full of escape codes is
worse than no progress at all.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

# The line currently being redrawn, so a log record can wipe it before writing
# and not leave half a bar embedded in the message.
_active: "Progress | None" = None


def _is_tty(stream) -> bool:
    if os.environ.get("NEPAL_NO_PROGRESS"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def fmt_duration(seconds: float) -> str:
    """h:mm:ss, or m:ss below an hour -- never '0:00:07.3421'."""
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def render_bar(label: str, done: int, total: int | None, elapsed_s: float,
               *, width: int = 80, note: str = "") -> str:
    """One progress line. Pure, so its arithmetic is testable without a tty."""
    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    speed = f"{rate:.1f}/s" if rate and rate < 1000 else ""
    if total:
        frac = min(1.0, done / total)
        eta = (total - done) / rate if rate > 0 else float("inf")
        bar_w = max(6, min(24, width - len(label) - 42))
        filled = int(round(frac * bar_w))
        bar = "#" * filled + "." * (bar_w - filled)
        line = (f"{label} [{bar}] {done}/{total} {frac * 100:3.0f}% "
                f"{speed} eta {fmt_duration(eta)}")
    else:
        line = f"{label} {done} in {fmt_duration(elapsed_s)} {speed}"
    if note:
        line += f"  {note}"
    return line[:max(20, width - 1)]


class Progress:
    """A counter that redraws one line on a terminal and logs otherwise."""

    def __init__(self, label: str, total: int | None = None, *,
                 stream=None, min_interval_s: float = 0.25,
                 log_every_s: float = 15.0, logger: logging.Logger | None = None):
        self.label = label
        self.total = int(total) if total else None
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval_s = float(min_interval_s)
        self.log_every_s = float(log_every_s)
        self.log = logger or logging.getLogger("nepal.progress")
        self.done = 0
        self.started = time.monotonic()
        self._last_draw = 0.0
        self._last_log = self.started
        self._tty = _is_tty(self.stream)
        self._width = shutil.get_terminal_size((100, 24)).columns
        self._closed = False

    # -- drawing ------------------------------------------------------
    def _line(self, note: str = "") -> str:
        return render_bar(self.label, self.done, self.total,
                          time.monotonic() - self.started,
                          width=self._width, note=note)

    def _draw(self, note: str = "") -> None:
        text = self._line(note)
        self.stream.write("\r\033[2K" + text)
        self.stream.flush()

    def clear(self) -> None:
        """Wipe the drawn line so something else can write cleanly."""
        if self._tty and not self._closed:
            self.stream.write("\r\033[2K")
            self.stream.flush()

    def redraw(self) -> None:
        if self._tty and not self._closed:
            self._draw()

    # -- use ----------------------------------------------------------
    def step(self, n: int = 1, note: str = "") -> None:
        self.done += n
        now = time.monotonic()
        if self._tty:
            if now - self._last_draw >= self.min_interval_s:
                self._last_draw = now
                self._draw(note)
        elif now - self._last_log >= self.log_every_s:
            self._last_log = now
            self.log.info("%s", self._line(note))

    def close(self, note: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        global _active
        if _active is self:
            _active = None
        elapsed = time.monotonic() - self.started
        if self._tty:
            self.stream.write("\r\033[2K")
            self.stream.flush()
        # Always state the finished total: on a terminal the bar is transient,
        # and a run whose output is scrolled back should still say what it did.
        self.log.info("%s: %d%s in %s%s", self.label, self.done,
                      f"/{self.total}" if self.total else "",
                      fmt_duration(elapsed), f"  {note}" if note else "")

    def __enter__(self) -> "Progress":
        global _active
        _active = self
        if self._tty:
            self._draw()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def track(items: Iterable[T], label: str, total: int | None = None,
          **kw) -> Iterator[T]:
    """Wrap an iterable in a Progress."""
    if total is None:
        try:
            total = len(items)            # type: ignore[arg-type]
        except TypeError:
            total = None
    with Progress(label, total, **kw) as p:
        for item in items:
            yield item
            p.step()


@contextmanager
def heartbeat(label: str, *, interval_s: float = 2.0, stream=None,
              logger: logging.Logger | None = None,
              log_every_s: float = 30.0) -> Iterator[None]:
    """Report elapsed time while one long opaque call runs.

    exiftool over the whole tree takes fifteen minutes and offers nothing to
    count. Saying "still running, 6:12 elapsed" is not progress, but it is the
    difference between waiting and wondering.
    """
    stream = stream if stream is not None else sys.stderr
    log = logger or logging.getLogger("nepal.progress")
    tty = _is_tty(stream)
    start = time.monotonic()
    stop = threading.Event()

    def tick() -> None:
        last_log = start
        while not stop.wait(interval_s if tty else 1.0):
            now = time.monotonic()
            if tty:
                stream.write(f"\r\033[2K{label} ... {fmt_duration(now - start)}")
                stream.flush()
            elif now - last_log >= log_every_s:
                last_log = now
                log.info("%s ... %s elapsed", label, fmt_duration(now - start))

    t = threading.Thread(target=tick, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)
        if tty:
            stream.write("\r\033[2K")
            stream.flush()
        log.info("%s: %s", label, fmt_duration(time.monotonic() - start))


class ProgressAwareHandler(logging.StreamHandler):
    """A stderr handler that does not write through a progress line."""

    def emit(self, record: logging.LogRecord) -> None:
        p = _active
        if p is not None:
            p.clear()
        try:
            super().emit(record)
        finally:
            if p is not None:
                p.redraw()
