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

import json
import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

DRAFT_W, DRAFT_H, DRAFT_CRF, DRAFT_FPS = 960, 540, 23, 30

# A card slot (the cold-open title) has no plate of its own to show, and the
# spec asks for nothing fancier than a caption over black.
CARD_COLOR = "black"
# Read at a glance while the card holds the frame, unlike the small per-shot
# debug label -- the two are different text at different distances.
CARD_FONTSIZE = 36

# The speech pass's own ceiling and range. -1.5 dBTP is the ceiling the
# final pass takes from the config, applied to speech early so that no
# single loud cue is the thing that trips the final limiter; LRA 11 is
# loudnorm's own default, written out so both passes visibly ask for the
# same range.
SPEECH_TP_DB, LRA = -1.5, 11
# The film's delivery rate. loudnorm upsamples to 192 kHz and hands that
# on, and left to itself the encoder negotiated 96 kHz -- the highest it
# takes -- for a draft whose every source is 48.
AUDIO_RATE = 48000

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
                    overlay: bool = True, fps: int = DRAFT_FPS,
                    secondary_index: int | None = None,
                    card_text: str | None = None) -> str:
    """The per-slot video chain -- what sits between ``[index:v]`` and the
    leg's output pad: reset timestamps, scale, label.

    The trim is NOT here. A ``trim`` filter runs after the decoder, so
    reaching a shot twenty minutes into a recording means decoding twenty
    minutes of video to throw away -- measured, that produced no output frames
    at all in three minutes across this timeline's 220 slots. Seeking is done
    at the input instead (``-ss`` before ``-i``), which jumps by keyframe.

    ``setpts=PTS-STARTPTS`` still matters: a seeked stream keeps its source
    timestamps, and concat would otherwise leave the cut sitting at the
    source's time rather than at zero.

    A split (``secondary_index`` given) is a fragment rather than one chain:
    the primary's half, the secondary's half from its own input pad, and the
    ``hstack`` that joins them -- laid out so the caller's ``[index:v]``
    prefix and output-pad suffix still bracket it like any other leg.
    """
    if is_card(row):
        # The lavfi ``color`` input build_command gives this row is already
        # exactly width x height at ``fps``, so there is no scale/pad step --
        # only the caption, and only when this ffmpeg can burn one in.
        chain = ["setpts=PTS-STARTPTS", "setsar=1"]
        text = card_text if card_text is not None else _card_text(row)
        # This caption is the card's whole content, not the shot_id/timecode
        # debug label -- it is deliberately independent of the `overlay` flag.
        if has_drawtext() and text:
            chain.append(
                f"drawtext=text='{escape_drawtext(text)}'"
                f":x=(w-text_w)/2:y=(h-text_h)/2:fontsize={CARD_FONTSIZE}"
                f":fontcolor=white")
        return ",".join(chain)
    if secondary_index is not None:
        # Each phone fills its half: scaled to cover and cropped, not
        # letterboxed -- a portrait clip letterboxed into a half-width frame
        # would be a stamp in a black field.
        half = ",".join([f"fps={fps}", "setpts=PTS-STARTPTS",
                         f"scale={width // 2}:{height}:force_original_aspect_ratio=increase"
                         f":force_divisible_by=2",
                         f"crop={width // 2}:{height}", "setsar=1"])
        chain = [f"{half}[h{index}a];[{secondary_index}:v]{half}[h{index}b];"
                 f"[h{index}a][h{index}b]hstack=inputs=2"]
    else:
        chain = []
        if is_still(row):
            dur = float(row["t_out"]) - float(row["t_in"])
            chain += [f"loop=loop=-1:size=1:start=0", f"fps={fps}", f"trim=duration={dur:.3f}"]
        else:
            # Every leg of the concat must share a rate: the proxies are 15
            # fps, the phones 30 or 60, and a concat of mixed rates produced
            # a 120 fps draft.
            chain += [f"fps={fps}"]
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


def is_card(row: Mapping[str, Any]) -> bool:
    """Whether this slot is a caption card (the cold-open title) rather than
    footage -- it has no shot and no source file to read."""
    return str(row.get("kind") or "") == "card"


def _card_text(row: Mapping[str, Any]) -> str:
    """The caption ``build_timeline`` wrote as JSON into ``motion`` for a
    card slot -- e.g. ``{"type": "card", "text": "..."}``. Malformed or
    missing text reads as no caption rather than as an error: a card with
    nothing to say still holds the timeline's length as plain black."""
    try:
        payload = json.loads(row.get("motion") or "{}")
    except (TypeError, ValueError):
        payload = {}
    return str(payload.get("text") or "")


def _cue_len(cue: Mapping[str, Any]) -> float:
    return float(cue["t_out"]) - float(cue["t_in"])


def _ms(seconds: Any) -> int:
    return int(round(float(seconds) * 1000))


def audio_filters(tracks: Mapping[str, Sequence[tuple[int, Mapping[str, Any]]]], *,
                  envelopes: Mapping[str, str], levels: Mapping[str, float],
                  film_len: float, measured: Mapping[str, float] | None = None,
                  measure_only: bool = False) -> list[str]:
    """The three tracks and their mix, as graph parts ending in ``[aout]``.

    ``tracks`` maps each track to its cues in time order, each with the
    index of the input that carries it. A cue is placed on film time with
    ``adelay`` -- one delay per channel of the stereo the mix ends in; a
    mono cue takes the first and ignores the rest -- and a track's cues are
    summed without rescaling (``normalize=0``): amix's default halves
    everything while two inputs are live, which would dip a cue every time
    another began.

    Speech is normalised per cue, because every recording sits at its own
    level, and faded over ``cue_fade_s`` so a cut into a word does not
    click. Location is set against its full level. Music is placed the
    same way, cue by cue at its own ``t_in``: chaining the cues with
    ``acrossfade`` instead collapsed every gap and shortened the run by one
    crossfade per join, so every cue after the first hole landed early
    under a picture cut. The crossfade is made by overlap -- each cue's
    input is read ``music_xfade_s`` longer than its slot and fades out
    over that extension under the cue that follows, which fades in when it
    abuts (starts within ``music_xfade_s`` of the previous cue's end) and
    starts clean after a real gap. Both tracks are then shaped by the
    mix's envelope -- decided elsewhere, arriving as an opaque ``volume``
    expression on film time. It carries commas, which would otherwise end
    the option, so it is quoted; the mixer promises it holds no quotes of
    its own. A track with no cues is silence for the film's length so the
    final mix always has its three inputs.

    The final normalisation is two-pass. Single-pass loudnorm re-levels
    what the envelope designed: through three seconds of silence its gain
    rode up and the returning bed came back 3.1 dB hot. So ``measure_only``
    has it print what it hears, and ``measured`` -- those numbers -- has it
    apply one static gain and only limit true peaks (``linear=true``).
    """
    fade = float(levels["cue_fade_s"])
    parts: list[str] = []

    def summed(prefix: str, n: int, tail: str, label: str) -> str:
        if not n:
            return f"anullsrc,atrim=duration={film_len:.3f}{label}"
        return "".join(f"[{prefix}{k}]" for k in range(n)) + f"amix=inputs={n}:normalize=0{tail}{label}"

    speech = tracks.get("speech") or ()
    for k, (i, c) in enumerate(speech):
        ms = _ms(c["t_in"])
        parts.append(f"[{i}:a]loudnorm=I={float(levels['speech_lufs']):g}:TP={SPEECH_TP_DB:g}:LRA={LRA},"
                     f"afade=t=in:d={fade:g},afade=t=out:st={max(0.0, _cue_len(c) - fade):.3f}:d={fade:g},"
                     f"adelay={ms}|{ms}[sp{k}]")
    parts.append(summed("sp", len(speech), "", "[speech]"))

    location = tracks.get("location") or ()
    full = float(levels["location_full_lufs"])
    for k, (i, c) in enumerate(location):
        ms = _ms(c["t_in"])
        parts.append(f"[{i}:a]volume={float(c['gain_lufs']) - full:g}dB,adelay={ms}|{ms}[lo{k}]")
    parts.append(summed("lo", len(location),
                        f",volume='{envelopes['location']}':eval=frame" if location else "", "[loc]"))

    music = tracks.get("music") or ()
    xfade = float(levels["music_xfade_s"])
    prev_out: float | None = None
    for k, (i, c) in enumerate(music):
        ms = _ms(c["t_in"])
        chain = [f"afade=t=in:d={xfade:g}"] if (
            prev_out is not None and float(c["t_in"]) - prev_out < xfade) else []
        chain += [f"afade=t=out:st={_cue_len(c):.3f}:d={xfade:g}", f"adelay={ms}|{ms}"]
        parts.append(f"[{i}:a]" + ",".join(chain) + f"[mu{k}]")
        prev_out = float(c["t_out"])
    parts.append(summed("mu", len(music),
                        f",volume='{envelopes['music']}':eval=frame" if music else "", "[mus]"))

    target_i, target_tp = float(levels["final_lufs"]), float(levels["true_peak_db"])
    final = f"loudnorm=I={target_i:g}:TP={target_tp:g}:LRA={LRA}"
    if measure_only:
        final += ":print_format=json"
    elif measured:
        m = {k: float(measured[k]) for k in ("input_i", "input_lra", "input_tp", "input_thresh")}
        gain = target_i - m["input_i"]
        if m["input_lra"] > LRA or m["input_tp"] + gain > target_tp:
            # loudnorm's own rule: a source range wider than the target's,
            # or a gain that would push a true peak past the ceiling, and
            # it quietly reverts to dynamic mode -- the second pass then
            # buys nothing, which nobody would hear until Gate 3.
            log.warning("S07 measured LRA %.1f against a target of %d, true peak %.1f dBTP "
                        "after %.1f dB of gain: loudnorm will revert to dynamic normalisation "
                        "and re-level the mix", m["input_lra"], LRA, m["input_tp"] + gain, gain)
        final += (f":measured_I={m['input_i']:g}:measured_LRA={m['input_lra']:g}"
                  f":measured_TP={m['input_tp']:g}:measured_thresh={m['input_thresh']:g}:linear=true")
    parts.append(f"[speech][loc][mus]amix=inputs=3:normalize=0,{final}[aout]")
    return parts


def build_command(rows: Sequence[Mapping[str, Any]], *, sources: Mapping[str, Path],
                  out_path: Path, cues: Sequence[Mapping[str, Any]] = (),
                  audio_sources: Mapping[str, Path] | None = None,
                  music_sources: Mapping[str, Path] | None = None,
                  envelopes: Mapping[str, str] | None = None,
                  width: int = DRAFT_W, height: int = DRAFT_H,
                  crf: int = DRAFT_CRF, fps: int = DRAFT_FPS, overlay: bool = True,
                  levels: Mapping[str, float] | None = None,
                  script_path: Path | None = None,
                  loudnorm_measured: Mapping[str, float] | None = None,
                  measure_only: bool = False) -> list[str]:
    """One ffmpeg invocation that renders the whole draft.

    Every shot is an input; the filter graph trims each, concatenates, and
    mixes. One process rather than render-then-concat: a per-shot file would
    be a second generation of H.264 on material that is already a proxy.

    The inputs are, in order: one per row (a split's secondary right after
    its primary), then one per speech, location and music cue, each track
    in time order, a music cue read ``music_xfade_s`` longer for its
    crossfade. Every pad index in the graph follows from that order.
    ``levels`` is the config's ``render`` block; ``envelopes`` the mix's
    per-track ``volume`` expressions. Without cues this is the silent draft
    it always was.

    The final normalisation is two-pass, the first pass audio-only so it
    costs seconds rather than a second encode: ``measure_only`` builds it
    -- the same cue inputs, now from index zero, no picture at all, output
    to null, loudnorm printing what it hears -- and ``loudnorm_measured``,
    what ``parse_loudnorm_json`` read from that, makes the render apply
    one static gain instead of re-levelling the mix as it goes.
    """
    if not rows:
        raise ValueError("nothing to render: the timeline is empty")
    if measure_only and not cues:
        raise ValueError("nothing to measure: no cues")
    if not measure_only and overlay and not has_drawtext():
        # Spec S07 asks for this overlay so Gate 3 feedback can name a moment.
        # Losing it is a real loss, not a cosmetic one -- but a draft nobody
        # can watch is worse than one whose shots are unlabelled.
        log.warning("S07 this ffmpeg has no drawtext filter (built without "
                    "libfreetype), so the draft carries no shot_id/timecode "
                    "overlay. Gate 3 notes will have to cite wall-clock times. "
                    "Install an ffmpeg with libfreetype to restore it.")
    if not measure_only and any(is_card(r) for r in rows) and not has_drawtext():
        # Same tradeoff as the overlay above, for card slots: no libfreetype
        # means no caption burned in, not a failed render.
        log.warning("S07 this ffmpeg has no drawtext filter, so card slot(s) "
                    "render as plain black without their caption.")
    # loudnorm prints its measurement at INFO, which -loglevel error would
    # swallow; -nostats keeps the progress line out of what gets parsed.
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "info" if measure_only else "error",
                      "-nostdin", "-y"] + (["-nostats"] if measure_only else [])
    legs: list[tuple[Mapping[str, Any], int, int | None]] = []  # row, its input, its secondary's
    n = 0
    if not measure_only:
        for r in rows:
            if not is_card(r) and sources.get(str(r["shot_id"])) is None:
                raise KeyError(f"no source for shot {r['shot_id']}")
        for r in rows:
            dur = float(r["t_out"]) - float(r["t_in"])
            if is_card(r):
                # No file backs a card: ffmpeg synthesises the frame from a
                # lavfi source instead of decoding one, at the row's own length
                # -- so the draft's length still matches the timeline's.
                cmd += ["-f", "lavfi", "-i",
                        f"color=c={CARD_COLOR}:s={width}x{height}:d={dur:.3f}:r={fps}"]
            elif is_still(r):
                # No -loop here: it is a demuxer-private option that image2 has
                # and the ISO-BMFF demuxer (HEIC) does not, so it fails with
                # "Option loop not found" on exactly the fourteen iPhone stills
                # in this timeline. The hold is done in the filter graph
                # instead, which works on decoded frames whatever read them.
                cmd += ["-i", str(sources[str(r["shot_id"])])]
            else:
                # -ss and -t BEFORE -i: input seeking, so the decoder starts
                # near the shot instead of at the head of the recording.
                cmd += ["-ss", f"{float(r.get('src_in') or 0.0):.3f}",
                        "-t", f"{dur:.3f}", "-i", str(sources[str(r["shot_id"])])]
            primary, secondary = n, None
            n += 1
            other = r.get("secondary_shot_id")
            if other and sources.get(str(other)) is not None:
                # The other phone's clip, right after its primary, seeked to
                # the instant the pair was aligned on, for the slot's length.
                cmd += ["-ss", f"{float(r.get('secondary_src_in') or 0.0):.3f}",
                        "-t", f"{dur:.3f}", "-i", str(sources[str(other)])]
                secondary, n = n, n + 1
            elif other:
                # The same policy as a slot with no media: a split whose other
                # clip nobody supplied shows one phone, loudly, rather than
                # killing the draft or vanishing from it.
                log.warning("S07 split slot %s has no source for its secondary %s; "
                            "rendering the primary alone", r.get("shot_id"), other)
            legs.append((r, primary, secondary))

    tracks: dict[str, list[tuple[int, Mapping[str, Any]]]] = {"speech": [], "location": [], "music": []}
    unknown = {str(c["track"]) for c in cues} - tracks.keys()
    if unknown:
        raise ValueError(f"cue(s) on unknown track(s): {sorted(unknown)}")
    files = {"speech": audio_sources or {}, "location": audio_sources or {},
             "music": music_sources or {}}
    levels = levels or {}
    for track, placed in tracks.items():
        for c in sorted((c for c in cues if c["track"] == track), key=lambda c: float(c["t_in"])):
            src = files[track].get(str(c["source"]))
            if src is None:
                raise KeyError(f"no audio source for cue {c.get('cue_id')} ({track}: {c['source']})")
            # Seeked at the input like a shot, for the cue's length on the
            # film -- plus the crossfade for music, which is made by overlap:
            # the extension plays out under the next cue (see audio_filters).
            # ffmpeg simply stops where the track ends if there is less.
            length = _cue_len(c) + (float(levels["music_xfade_s"]) if track == "music" else 0.0)
            cmd += ["-ss", f"{float(c['src_in']):.3f}", "-t", f"{length:.3f}", "-i", str(src)]
            placed.append((n, c))
            n += 1

    parts: list[str] = []
    if not measure_only:
        labels: list[str] = []
        for k, (r, i, si) in enumerate(legs):
            parts.append(f"[{i}:v]{segment_filters(r, i, width=width, height=height, overlay=overlay, fps=fps, secondary_index=si)}[v{k}]")
            labels.append(f"[v{k}]")
        parts.append("".join(labels) + f"concat=n={len(legs)}:v=1:a=0[vout]")
    if cues:
        parts += audio_filters(tracks, envelopes=envelopes or {}, levels=levels,
                               film_len=max(float(r["t_out"]) for r in rows),
                               measured=loudnorm_measured, measure_only=measure_only)

    # A 564-slot draft put this whole graph past Linux's MAX_ARG_STRLEN
    # (128 KiB) as a single -filter_complex argument, and subprocess.run
    # raised OSError: Argument list too long -- the real film runs 500-800
    # slots. ffmpeg reads the identical graph from a file just as well, so
    # it goes on disk instead of on the command line. Written through a
    # temporary name and renamed, per this pipeline's checkpoint rule.
    graph = ";".join(parts)
    filters_path = script_path or out_path.with_suffix(".measure.filters" if measure_only else ".filters")
    tmp_filters_path = filters_path.with_name(filters_path.name + ".tmp")
    tmp_filters_path.write_text(graph)
    tmp_filters_path.rename(filters_path)

    if measure_only:
        # Audio only, to nowhere: the product is what loudnorm prints.
        cmd += ["-filter_complex_script", str(filters_path), "-map", "[aout]", "-vn", "-f", "null", "-"]
        return cmd
    maps = ["-map", "[vout]"]
    if cues:
        # No -shortest: the mix may run a hair past the picture, which is
        # harmless, but a mix that ran short would cut the picture with it.
        maps += ["-map", "[aout]"]
    cmd += ["-filter_complex_script", str(filters_path), *maps, "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p"]
    if cues:
        cmd += ["-c:a", "aac", "-b:a", f"{int(levels['audio_bitrate_k'])}k", "-ar", str(AUDIO_RATE)]
    cmd.append(str(out_path))
    return cmd


def describe(cmd: Sequence[str]) -> str:
    """The command as a copyable line, for the log and for a bug report.

    The graph itself now lives in a script file, not in ``cmd`` -- a bare
    path tells a DEBUG log nothing, so this inlines the file's content in
    place of the path, exactly where ``-filter_complex`` used to show it.
    Falls back to the path if the file is gone by the time this runs.
    """
    cmd = list(cmd)
    try:
        i = cmd.index("-filter_complex_script")
        graph = Path(cmd[i + 1]).read_text()
        cmd[i], cmd[i + 1] = "-filter_complex", graph
    except (ValueError, OSError):
        pass
    return " ".join(shlex.quote(c) for c in cmd)


def probe_loudness(path: Path, *, t_in: float, t_out: float) -> float:
    """Mean level in dB of one window of a file's audio, as ``volumedetect``
    measures it (-91 for digital silence). The slow test's ear: the one
    thing here that runs ffmpeg for an answer rather than building one."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-ss", f"{t_in:.3f}", "-t", f"{t_out - t_in:.3f}",
         "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", proc.stderr)
    if not m:
        raise RuntimeError(f"volumedetect reported no mean_volume for {path}: {proc.stderr[-400:]}")
    return float(m.group(1))


def parse_loudnorm_json(stderr: str) -> dict[str, float]:
    """The four measurements the measuring pass prints, as ffmpeg names
    them -- the JSON block loudnorm writes at the end of stderr, its
    values quoted as strings. Anything short of all four is an error that
    says which, because a render with half a measurement would quietly be
    a single-pass render."""
    blocks = re.findall(r"\{[^{}]*\}", stderr)
    if not blocks:
        raise ValueError("no loudnorm JSON block in ffmpeg's output -- was the pass run "
                         "with -loglevel info? tail: " + stderr[-300:].strip())
    try:
        data = json.loads(blocks[-1])
    except ValueError as e:
        raise ValueError(f"loudnorm JSON block does not parse: {e}") from None
    out: dict[str, float] = {}
    for key in ("input_i", "input_lra", "input_tp", "input_thresh"):
        try:
            out[key] = float(data[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"loudnorm JSON block has no usable {key}: {blocks[-1]}") from None
    return out
