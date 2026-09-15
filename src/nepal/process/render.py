"""S07 -- the draft render.

Conformed from the proxies, never the originals: the draft exists to be
watched and argued with at Gate 3, and re-reading 5.7K H.265 for a 960x540
preview would cost hours to look identical at that size.

The command is built as data and executed by a thin wrapper, so the parts that
are easy to get wrong -- the filter graph, the trims, the mix -- are testable
without rendering anything.

One thing here is not a mechanical translation of the spec. The spec asks for
an overlay of shot_id and timecode "so Gate 3 feedback can reference specific
moments". That is the whole purpose of the draft, so the overlay is drawn from
the timeline rather than from ffmpeg's frame counter: a viewer who says "the
shot at 4:12 is wrong" must be naming a row someone can then find.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

DRAFT_W, DRAFT_H, DRAFT_CRF = 960, 540, 23

_HAS_DRAWTEXT: bool | None = None


def has_drawtext(*, refresh: bool = False) -> bool:
    """Whether this ffmpeg can burn text into a frame.

    ``drawtext`` needs libfreetype at build time and a great many distribution
    builds omit it -- the machine this was developed on among them. Asking the
    binary is the only reliable answer: the filter's absence is not visible
    until the graph is assembled, at which point the whole render dies with
    "No such filter" after the encoder has already started.

    Probed once and cached, because it cannot change while the process runs.
    """
    global _HAS_DRAWTEXT
    if _HAS_DRAWTEXT is None or refresh:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=30).stdout
            _HAS_DRAWTEXT = any(line.split()[1:2] == ["drawtext"]
                                for line in out.splitlines() if line.strip())
        except (OSError, subprocess.SubprocessError):
            _HAS_DRAWTEXT = False
    return bool(_HAS_DRAWTEXT)


def timecode(seconds: float) -> str:
    """``M:SS`` -- what a person says out loud when they name a moment."""
    s = max(0.0, float(seconds))
    return f"{int(s // 60)}:{int(s % 60):02d}"


def escape_drawtext(text: str) -> str:
    """ffmpeg's drawtext eats colons, backslashes and quotes.

    A shot_id contains a '#', and a place name can contain an apostrophe;
    both arrive here from the database rather than from a literal, so the
    escaping is not optional.
    """
    out = str(text)
    for a, b in (("\\", r"\\"), (":", r"\:"), ("'", r"\'"), ("%", r"\%")):
        out = out.replace(a, b)
    return out


def segment_filters(row: Mapping[str, Any], index: int, *,
                    width: int = DRAFT_W, height: int = DRAFT_H,
                    overlay: bool = True) -> str:
    """The per-shot video chain: reset timestamps, scale, label.

    The trim is NOT here. A ``trim`` filter runs after the decoder, so
    reaching a shot twenty minutes into a recording means decoding twenty
    minutes of video to throw away -- measured, that produced no output frames
    at all in three minutes across this timeline's 220 slots. Seeking is done
    at the input instead (``-ss`` before ``-i``), which jumps by keyframe.

    ``setpts=PTS-STARTPTS`` still matters: a seeked stream keeps its source
    timestamps, and concat would otherwise leave the cut sitting at the
    source's time rather than at zero.
    """
    chain = []
    if is_still(row):
        dur = float(row["t_out"]) - float(row["t_in"])
        chain += [f"loop=loop=-1:size=1:start=0", "fps=25", f"trim=duration={dur:.3f}"]
    chain += ["setpts=PTS-STARTPTS",
             f"scale={width}:{height}:force_original_aspect_ratio=decrease"
             f":force_divisible_by=2",
             f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
             "setsar=1"]
    if overlay and has_drawtext():
        label = escape_drawtext(f"{row['shot_id']}  {timecode(row['t_in'])}")
        chain.append(
            f"drawtext=text='{label}':x=8:y=h-24:fontsize=14:fontcolor=white"
            f":box=1:boxcolor=black@0.5:boxborderw=4")
    return ",".join(chain)


def is_still(row: Mapping[str, Any]) -> bool:
    """Whether this slot is a photograph rather than a piece of video."""
    return str(row.get("media_kind") or "video") == "photo"


def build_command(rows: Sequence[Mapping[str, Any]], *, sources: Mapping[str, Path],
                  out_path: Path, music_path: Path | None = None,
                  width: int = DRAFT_W, height: int = DRAFT_H,
                  crf: int = DRAFT_CRF, music_lufs: float = -14.0,
                  overlay: bool = True) -> list[str]:
    """One ffmpeg invocation that renders the whole draft.

    Every shot is an input; the filter graph trims each, concatenates, and
    mixes. One process rather than render-then-concat: a per-shot file would
    be a second generation of H.264 on material that is already a proxy.
    """
    if not rows:
        raise ValueError("nothing to render: the timeline is empty")
    if overlay and not has_drawtext():
        # Spec S07 asks for this overlay so Gate 3 feedback can name a moment.
        # Losing it is a real loss, not a cosmetic one -- but a draft nobody
        # can watch is worse than one whose shots are unlabelled.
        log.warning("S07 this ffmpeg has no drawtext filter (built without "
                    "libfreetype), so the draft carries no shot_id/timecode "
                    "overlay. Gate 3 notes will have to cite wall-clock times. "
                    "Install an ffmpeg with libfreetype to restore it.")
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    inputs: list[str] = []
    for r in rows:
        src = sources.get(str(r["shot_id"]))
        if src is None:
            raise KeyError(f"no source for shot {r['shot_id']}")
        inputs.append(str(src))
    for src, r in zip(inputs, rows):
        dur = float(r["t_out"]) - float(r["t_in"])
        if is_still(r):
            # No -loop here: it is a demuxer-private option that image2 has and
            # the ISO-BMFF demuxer (HEIC) does not, so it fails with "Option
            # loop not found" on exactly the fourteen iPhone stills in this
            # timeline. The hold is done in the filter graph instead, which
            # works on decoded frames whatever read them.
            cmd += ["-i", src]
        else:
            # -ss and -t BEFORE -i: input seeking, so the decoder starts near
            # the shot instead of at the head of the recording.
            cmd += ["-ss", f"{float(r.get('src_in') or 0.0):.3f}",
                    "-t", f"{dur:.3f}", "-i", src]
    music_idx = None
    if music_path is not None:
        music_idx = len(inputs)
        cmd += ["-i", str(music_path)]

    parts: list[str] = []
    labels: list[str] = []
    for i, r in enumerate(rows):
        parts.append(f"[{i}:v]{segment_filters(r, i, width=width, height=height, overlay=overlay)}[v{i}]")
        labels.append(f"[v{i}]")
    parts.append("".join(labels) + f"concat=n={len(rows)}:v=1:a=0[vout]")

    maps = ["-map", "[vout]"]
    if music_idx is not None:
        # The bed is normalised to the spec's target rather than assumed to be
        # there already: the music library is whatever the operator had.
        parts.append(f"[{music_idx}:a]loudnorm=I={music_lufs:g}:TP=-1.5:LRA=11[aout]")
        maps += ["-map", "[aout]", "-shortest"]

    cmd += ["-filter_complex", ";".join(parts), *maps,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if music_idx is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd.append(str(out_path))
    return cmd


def describe(cmd: Sequence[str]) -> str:
    """The command as a copyable line, for the log and for a bug report."""
    return " ".join(shlex.quote(c) for c in cmd)
