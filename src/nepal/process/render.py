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
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

DRAFT_W, DRAFT_H, DRAFT_CRF, DRAFT_FPS = 960, 540, 23, 30

# How far SHORT of the timeline the rendered draft may fall before
# ``run_render`` calls it a failed render. Only short: every leg is resampled
# to one rate and so rounds up to a whole frame, which on the real film put
# the draft 5.489 s over a 2145.944 s timeline -- arithmetic, not loss. A
# render that stopped part-way is out by slots. The stage passes
# ``render.draft_tol_s``; this is the fallback for a caller with no config.
DRAFT_TOL_S = 5.0

# The checks below ask the same question cues.py asks of its own edges --
# do these two times coincide? -- and must answer it the same way. The
# value is repeated rather than imported because it is private over there;
# a test asserts the two cannot drift apart.
EDGE_TOL_S = 1e-3

# A card slot (the cold-open title) has no plate of its own to show, and the
# spec asks for nothing fancier than a caption over black.
CARD_COLOR = "black"
# Read at a glance while the card holds the frame, unlike the small per-shot
# debug label -- the two are different text at different distances.
CARD_FONTSIZE = 36

# The speech pass's own ceiling and range. -1.5 dBTP is the ceiling the
# final pass takes from the config, applied to speech early so that no
# single loud cue is the thing that trips the final limiter; LRA 11 is
# loudnorm's own default, written out so every loudnorm in this graph
# visibly asks for the same range.
SPEECH_TP_DB, LRA = -1.5, 11
# The film's delivery rate. loudnorm upsamples to 192 kHz and hands that
# on, and left to itself the encoder negotiated 96 kHz -- the highest it
# takes -- for a draft whose every source is 48.
AUDIO_RATE = 48000
# concat refuses pieces that disagree on rate, sample format or layout, and
# the location recordings agree on none of the three. aformat states the
# constraint and lets the graph insert the resampler wherever a piece needs
# one -- amix did that conversion itself, which is why the old path could
# ignore it.
CONCAT_AFORMAT = (f"aformat=sample_fmts=fltp:sample_rates={AUDIO_RATE}"
                  ":channel_layouts=stereo")

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


def _fades(cue: Mapping[str, Any], levels: Mapping[str, float]) -> tuple[float, float]:
    """The row's own fades -- cues.py writes the cut fade at cuts, the
    window fade at the silence's edges, the crossfade on the cue after it
    -- or the config's cut fade where the row has none."""
    default = float(levels["cue_fade_s"])
    fade_in, fade_out = (default if cue.get(k) is None else float(cue[k])
                         for k in ("fade_in_s", "fade_out_s"))
    return fade_in, fade_out


def _afades(fade_in: float, fade_out: float, *, end: float) -> list[str]:
    """The afade filters for a cue that ends at ``end``. A zero fade is no
    filter at all: afade with d=0 falls back to its default of 44100
    samples, a one-second fade nobody asked for."""
    out = []
    if fade_in > 0:
        out.append(f"afade=t=in:d={fade_in:g}")
    if fade_out > 0:
        out.append(f"afade=t=out:st={max(0.0, end - fade_out):.3f}:d={fade_out:g}")
    return out


def _location_pieces(location: Sequence[tuple[int, Mapping[str, Any], float]], *,
                     full: float, levels: Mapping[str, float]) -> tuple[list[str], list[str]]:
    """The location track's cues laid end to end, as graph parts and the
    labels to hand ``concat``, in time order.

    The cues tile the timeline: each is a slot's span, and a photo or card
    slot's cue continues the previous recording from its ``src_out``. So the
    track can be built by laying them end to end, which is what it is --
    one film's length of sound, ~1800 s. Placing each one with ``adelay``
    instead and summing 735 of them with ``amix`` asked amix to add 735
    streams whose lengths average half the film, of the order of 3e10
    samples: measured on the real draft, 19.5 minutes at 100 % of one core
    for the audio-only measurement pass alone, and the same graph again
    under the render -- twice what the picture costs. Timed both ways on
    the box against one synthetic timeline, 700 cues over 600 s, the
    measurement pass alone: 397.4 s summed against 57.1 s laid end to
    end, 7.0x. A measured 3x is worth taking.

    Where no cue covers a stretch of film -- a card that opens an act is
    intended dead air -- concat can only say so with a piece of silence.
    """
    parts: list[str] = []
    labels: list[str] = []
    played = 0.0
    for k, (i, c, length) in enumerate(location):
        t_in = float(c["t_in"])
        if t_in - played > EDGE_TOL_S:
            parts.append(f"anullsrc=r={AUDIO_RATE}:cl=stereo,"
                         f"atrim=duration={t_in - played:.3f},{CONCAT_AFORMAT}[log{k}]")
            labels.append(f"[log{k}]")
        chain = [f"volume={float(c['gain_lufs']) - full:g}dB",
                 *_afades(*_fades(c, levels), end=length),
                 # Input -t stops at a packet boundary, not at the sample,
                 # and a source that ends early simply ends. amix did not
                 # care -- every cue was pinned to its own t_in by adelay --
                 # but in a concat one piece's overshoot pushes every later
                 # cue late and an undershoot pulls them early, cue after
                 # cue. So each piece is held to exactly its span. Both
                 # filters read the cue's own span because an input seek
                 # rebases the stream to zero (checked: `-ss 5` delivers
                 # pts_time 0), which is also why the fades above are
                 # written relative.
                 f"atrim=duration={length:.3f}", f"apad=whole_dur={length:.3f}",
                 # apad can give a piece its length but not give those samples
                 # a time. A cue seeked at or past the end of its recording --
                 # the ambience held on under a still, after the recording it
                 # is held from has run out -- decodes nothing at all, and the
                 # padding apad then invents carries no pts. concat adds each
                 # piece's end time to the delta it shifts every later piece
                 # by, so one such cue puts the whole rest of the track at a
                 # nonsense time: the graph dies mid-film with "Invalid data
                 # found when processing input" and ffmpeg still writes a
                 # valid trailer and exits 0. Measured on the real corpus --
                 # one cue at 132.2 s left 138.7 s of a 2145.9 s draft.
                 # Counting a piece's timestamps off its own samples cannot
                 # depend on what the input did or did not deliver.
                 "asetpts=N/SR/TB",
                 CONCAT_AFORMAT]
        parts.append(f"[{i}:a]" + ",".join(chain) + f"[lo{k}]")
        labels.append(f"[lo{k}]")
        played = float(c["t_out"])
    return parts, labels


def audio_filters(tracks: Mapping[str, Sequence[tuple[int, Mapping[str, Any], float]]], *,
                  envelopes: Mapping[str, str], levels: Mapping[str, float],
                  film_len: float, measured: Mapping[str, float] | None = None,
                  measure_only: bool = False) -> list[str]:
    """The three tracks and their mix, as graph parts ending in ``[aout]``.

    ``tracks`` maps each track to its cues in time order, each with the
    index of the input that carries it and how much of it is read. A speech
    or music cue is placed on film time with ``adelay`` -- one delay per
    channel of the stereo the mix ends in; a mono cue takes the first and
    ignores the rest -- and those tracks are summed without rescaling
    (``normalize=0``): amix's default halves everything while two inputs
    are live, which would dip a cue every time another began. They overlap
    -- crossfades, a replay under a card -- and there are 47 of them in the
    real film. The 735 location cues do not overlap, so they are laid end
    to end with ``concat`` instead (see ``_location_pieces``); only cues
    that do overlap send that track back to ``adelay`` and ``amix``.

    Every cue fades at its edges as its row says (``_fades``): without
    that, every change of recording is a butt-splice of two unrelated
    waveforms -- hundreds of clicks at picture cuts. Speech is normalised
    per cue first, because every recording sits at its own level.
    Location is set against its full level. Music is placed the same way,
    cue by cue at its own ``t_in``: chaining the cues with ``acrossfade``
    instead collapsed every gap and shortened the run by one crossfade
    per join, so every cue after the first hole landed early under a
    picture cut. The crossfade is made by overlap -- each cue is read its
    own fade-out longer than its slot and fades out over that extension
    under the cue that follows, which fades in as its row says. The row
    is the only source of that length: ``cues.py`` writes ``music_xfade_s``
    where a cue hands over to the next and the shorter window fade where
    it runs into the silence, and a renderer that overrode it with the
    flat crossfade threw the distinction away.
    A music cue is cut from a decoded-from-start track with ``atrim``
    rather than seeked (see build_command), so its timestamps are reset
    the way a seek would have. Location and music are then shaped by the
    mix's envelope -- decided elsewhere, arriving as an opaque ``volume``
    expression on film time. It carries commas, which would otherwise end
    the option, so it is quoted; the mixer promises it holds no quotes of
    its own. A track with no cues is silence for the film's length, at
    the film's rate and layout, so the final mix always has its three
    inputs.

    The final normalisation is two-pass. Single-pass loudnorm re-levels
    what the envelope designed: through three seconds of silence its gain
    rode up and the returning bed came back 3.1 dB hot. So ``measure_only``
    has it print what it hears, and ``measured`` -- those numbers -- turns
    the final stage into what linear mode is made of: one static gain to
    the target, then a true-peak limiter to keep the ceiling.
    """
    parts: list[str] = []

    def summed(prefix: str, n: int, tail: str, label: str) -> str:
        if not n:
            return f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=duration={film_len:.3f}{label}"
        return "".join(f"[{prefix}{k}]" for k in range(n)) + f"amix=inputs={n}:normalize=0{tail}{label}"

    speech = tracks.get("speech") or ()
    for k, (i, c, length) in enumerate(speech):
        ms = _ms(c["t_in"])
        chain = [f"loudnorm=I={float(levels['speech_lufs']):g}:TP={SPEECH_TP_DB:g}:LRA={LRA}",
                 *_afades(*_fades(c, levels), end=length), f"adelay={ms}|{ms}"]
        parts.append(f"[{i}:a]" + ",".join(chain) + f"[sp{k}]")
    parts.append(summed("sp", len(speech), "", "[speech]"))

    # concat lays its pieces down in the order it is given them, so the
    # graph's order has to be film order. build_command already sorts every
    # track by t_in; sorting here is what makes the tiling test below mean
    # what it says rather than rest on that promise.
    location = sorted(tracks.get("location") or (), key=lambda p: float(p[1]["t_in"]))
    full = float(levels["location_full_lufs"])
    envelope = f",volume='{envelopes['location']}':eval=frame" if location else ""
    overlap = next((p for p in zip(location, location[1:])
                    if float(p[1][1]["t_in"]) < float(p[0][1]["t_out"]) - EDGE_TOL_S), None)
    if location and overlap is None:
        pieces, labels = _location_pieces(location, full=full, levels=levels)
        parts += pieces
        parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1{envelope}[loc]")
    else:
        if overlap is not None:
            # Two cues that sound at once cannot be laid end to end, and
            # getting the film's ambience right beats getting it fast.
            (_, a, _), (_, b, _) = overlap
            log.warning("S07 location cues %s (%.3f-%.3f) and %s (%.3f-%.3f) overlap, so the "
                        "track cannot be concatenated; falling back to adelay+amix, which is "
                        "correct but costs ~20 min of one core per pass on a film-length mix.",
                        a.get("cue_id"), float(a["t_in"]), float(a["t_out"]),
                        b.get("cue_id"), float(b["t_in"]), float(b["t_out"]))
        for k, (i, c, length) in enumerate(location):
            ms = _ms(c["t_in"])
            chain = [f"volume={float(c['gain_lufs']) - full:g}dB",
                     *_afades(*_fades(c, levels), end=length), f"adelay={ms}|{ms}"]
            parts.append(f"[{i}:a]" + ",".join(chain) + f"[lo{k}]")
        parts.append(summed("lo", len(location), envelope, "[loc]"))

    music = tracks.get("music") or ()
    for k, (i, c, length) in enumerate(music):
        ms, src_in = _ms(c["t_in"]), float(c["src_in"])
        chain = [f"atrim=start={src_in:.3f}:end={src_in + length:.3f}", "asetpts=PTS-STARTPTS",
                 *_afades(*_fades(c, levels), end=length), f"adelay={ms}|{ms}"]
        parts.append(f"[{i}:a]" + ",".join(chain) + f"[mu{k}]")
    parts.append(summed("mu", len(music),
                        f",volume='{envelopes['music']}':eval=frame" if music else "", "[mus]"))

    target_i, target_tp = float(levels["final_lufs"]), float(levels["true_peak_db"])
    if measure_only:
        final = f"loudnorm=I={target_i:g}:TP={target_tp:g}:LRA={LRA}:print_format=json"
    elif measured:
        # Not loudnorm's linear mode but what linear mode is: one gain and a
        # ceiling. ``linear=true`` is honoured only while measured_LRA <=
        # LRA (af_loudnorm.c, init), and that option stops at 20 LU -- the
        # film is a designed three-second silence and half an hour of
        # envelope, wider than that by construction, so loudnorm warned and
        # reverted to dynamic, re-levelling the very envelope this second
        # pass exists to preserve. A gain has no range precondition.
        gain = target_i - float(measured["input_i"])
        # alimiter's limit is linear amplitude, not dB; its attack and
        # release defaults are left alone, and level=false stops it
        # normalising the output afterwards, which would undo the gain.
        final = (f"volume={gain:g}dB,"
                 f"alimiter=limit={10 ** (target_tp / 20):g}:level=false")
    else:
        final = f"loudnorm=I={target_i:g}:TP={target_tp:g}:LRA={LRA}"
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
    in time order, a music cue read its own fade-out longer for its
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

    tracks: dict[str, list[tuple[int, Mapping[str, Any], float]]] = {"speech": [], "location": [], "music": []}
    unknown = {str(c["track"]) for c in cues} - tracks.keys()
    if unknown:
        raise ValueError(f"cue(s) on unknown track(s): {sorted(unknown)}")
    files = {"speech": audio_sources or {}, "location": audio_sources or {},
             "music": music_sources or {}}
    levels = levels or {}
    film_len = max(float(r["t_out"]) for r in rows)
    for track, placed in tracks.items():
        for c in sorted((c for c in cues if c["track"] == track), key=lambda c: float(c["t_in"])):
            src = files[track].get(str(c["source"]))
            if src is None:
                raise KeyError(f"no audio source for cue {c.get('cue_id')} ({track}: {c['source']})")
            if track == "music":
                # A music cue reads its slot plus its own fade-out, the
                # extension that plays out under the next cue (see
                # audio_filters) -- clamped to the film's end so the mix
                # never outlasts the picture. The row's value and not the
                # flat crossfade: a cue that hands over reads the whole
                # crossfade, one that runs into the silence window reads
                # only the shorter fade cues.py gave it and stops there.
                # No -ss: the library is VBR MP3, and an input seek
                # on MP3 without a table of contents goes by a bitrate
                # estimate that can land seconds off, which the overlap
                # arithmetic cannot survive. (An -ss after the -i is no
                # answer: between two -i it belongs to the next input.) The
                # track is decoded from its start, bounded by -t, and cut
                # to the sample in the graph. ffmpeg simply stops where the
                # track ends if there is less.
                length = max(0.0, min(_cue_len(c) + _fades(c, levels)[1],
                                      film_len - float(c["t_in"])))
                cmd += ["-t", f"{float(c['src_in']) + length:.3f}", "-i", str(src)]
            else:
                # Seeked at the input like a shot, for the cue's length on
                # the film: a WAV seeks exactly.
                length = _cue_len(c)
                cmd += ["-ss", f"{float(c['src_in']):.3f}", "-t", f"{length:.3f}", "-i", str(src)]
            placed.append((n, c, length))
            n += 1

    parts: list[str] = []
    if not measure_only:
        labels: list[str] = []
        for k, (r, i, si) in enumerate(legs):
            parts.append(f"[{i}:v]{segment_filters(r, i, width=width, height=height, overlay=overlay, fps=fps, secondary_index=si)}[v{k}]")
            labels.append(f"[v{k}]")
        parts.append("".join(labels) + f"concat=n={len(legs)}:v=1:a=0[vout]")
    if cues:
        parts += audio_filters(tracks, envelopes=envelopes or {}, levels=levels, film_len=film_len,
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
    # Explicitly UTF-8: the card captions are Cyrillic, and the locale of
    # whatever runs this is not a thing to rely on.
    tmp_filters_path.write_text(graph, encoding="utf-8")
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
    cmd.append(str(part_path(out_path)))
    return cmd


def part_path(out_path: Path) -> Path:
    """Where the render actually writes, until it has finished writing.

    The extension is kept because ffmpeg picks its muxer from it -- a bare
    ``.part`` would fail at "Unable to find a suitable output format".
    """
    return out_path.with_suffix(".part" + out_path.suffix)


def media_seconds(path: Path) -> float | None:
    """The video stream's length as ffprobe reads it back, or None if it
    cannot be read at all."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, check=True).stdout
        return round(float(out.strip()), 3)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        log.warning("S07 could not probe the length of %s: %s", path, e)
        return None


def run_render(cmd: Sequence[str], out_path: Path, *, expect_s: float | None = None,
               tol_s: float = DRAFT_TOL_S) -> subprocess.CompletedProcess:
    """Run the render and publish its output only if ffmpeg succeeded and
    the file is as long as it was asked to be.

    A render measured in tens of minutes that is killed, preempted or
    rsynced over leaves a truncated file where the good draft was, and
    everything downstream -- the Gate 3 page, the measured length, whoever
    watches it -- reads that as the draft. Written through a temporary name
    and renamed, which is atomic on one filesystem, so draft.mp4 is either
    the last complete render or nothing at all.

    A zero exit is not the whole verdict. ffmpeg breaks out of its transcode
    loop on a mid-graph error, then flushes its encoders, writes a valid
    trailer and returns 0 -- so a render that died at 138.7 s of a 2145.9 s
    film came back "successful", with a playable file that was 6 % of the
    cut. The length is the verdict ffmpeg will not give: ``expect_s`` is
    what the timeline says the picture is, and a file short of that by more
    than ``tol_s`` is not published and comes back as the failure it is.
    Short only: a finished draft is always a shade longer than its timeline,
    because every leg rounds up to a whole frame at the draft's rate.
    The short file is kept at its temporary name, because the thing to look
    at when this fires is where it stopped.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    part = part_path(out_path)
    if proc.returncode != 0 or not part.exists():
        return proc
    got = media_seconds(part) if expect_s is not None else None
    if got is not None and expect_s is not None and got > expect_s:
        # Longer is arithmetic, not failure, and it only goes this way: every
        # leg is resampled to one rate and so rounds up to a whole frame.
        # Measured on the real film, 512 slots at 30 fps: 2151.433 s against a
        # 2145.944 s timeline, +10.7 ms a slot. Said out loud, never fatal --
        # only a draft that is *short* has lost something.
        log.info("S07 the draft runs %.3f s against the timeline's %.3f s (+%.3f s): each leg "
                 "rounds up to a whole frame at the draft's rate.", got, expect_s, got - expect_s)
    if expect_s is None or (got is not None and expect_s - got <= tol_s):
        part.replace(out_path)
        return proc
    msg = (f"ffmpeg exited 0 but wrote {'a file whose length cannot be read' if got is None else f'{got:.3f} s'} "
           f"where the timeline is {expect_s:.3f} s (tolerance {tol_s:g} s short): the render "
           f"stopped part-way through the film. {out_path} is left as it was; the short file is at {part}.")
    log.error("S07 %s", msg)
    return subprocess.CompletedProcess(proc.args, 1, proc.stdout, (proc.stderr or "") + "\n" + msg)


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
        graph = Path(cmd[i + 1]).read_text(encoding="utf-8")
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
        if not math.isfinite(out[key]):
            # A silent mix measures -inf, which loudnorm refuses at graph
            # init -- better said here, with the number, than there.
            raise ValueError(f"loudnorm measured a non-finite {key} ({data[key]}): a silent mix?")
    return out
